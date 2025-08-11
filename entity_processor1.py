import json
import requests
import pandas as pd
import spacy
from scispacy.linking import EntityLinker
import dspy
from typing import List, Dict, Any, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
import logging
import numpy as np
import ast
import re
import datetime
from collections import defaultdict
from embedding_pipeline import BioMedBERTEmbedder
from summarize_entities import EntitySummarizer


# Optimize spacy for speed
spacy.prefer_gpu()  # Use GPU if available

# Configure enhanced logging with file output
def setup_logging():
    """Setup enhanced logging with both console and file output."""
    # Create logs directory if it doesn't exist
    os.makedirs('./logs', exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./logs/entity_processing_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger

logger = setup_logging()

class BatchEntityProcessingSignature(dspy.Signature):
    """DSPy signature for batch processing entities and their relationships."""
    
    paragraph_content = dspy.InputField(desc="""The full paragraph content. Example: They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve. Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly.
""")
    entity_list = dspy.InputField(desc="JSON list of entities detected by SciSpacy with their names and labels")
    
    processed_entities = dspy.OutputField(desc="""You are a JSON-only assistant. Your goal is to generate a JSON object with entities and relationships from the given text.
ALWAYS respond with nothing but valid JSON that conforms to this schema:
{
    "processed_entities": {
        "entities": [
            {
                "entity_name": "entity name, typically 1-2 words.",
                "entity_description": "comprehensive description of what this entity is and its role, and using the entity_name within the description"
            }
        ],
        "relationships": [
            {
                "source_entity": "name of source entity in 1-2 words",
                "target_entity": "name of target entity in 1-2 words",
                "relationship_description": "Description of the relationship between the source entity and the target entity in 20 words or less.",
                "relationship_strength":  "numeric score indicating strength of the relationship between the source entity and target entity" 
            }
        ]
    }
}
REMEMBER: For all chunks with more than 1 entity, all entities should have at least one relationship to another entity as either a source or target entity.
IMPORTANT: All relationships must have a relationship_strength between 5 and 10. Relationships with strength below 5 should not be included.
IMPORTANT: You can output ONLY JSON.
""")

class EntityProcessor:
    """Process entities from chunked JSON using SciSpacy for detection and DSPy/Ollama for descriptions."""
    
    # Class-level cache for SciSpacy models
    _nlp_cache = {}
    _linker_cache = {}
    
    def __init__(self, spacy_model: str = "en_core_sci_md", ollama_model: str = None, 
                 ollama_endpoint: str = "http://localhost:11434",
                 chunk_size: int = 25,
                 max_workers: Optional[int] = None,
                 progress_interval: int = 20,
                 cache_path: str = "./cache/entity_desc_cache.json",
                 relationship_fallback: bool = False):
        """
        Initialize the entity processor.
        
        Args:
            spacy_model: SciSpacy model to use (default: en_core_sci_md)
            ollama_model: Ollama or HuggingFace model to use for descriptions 
                         (e.g., "llama3.1:8b" for Ollama or "meta-llama/Llama-3.2-1B-Instruct" for HF)
            ollama_endpoint: Ollama endpoint URL (ignored for HuggingFace models)
        """
        self.spacy_model = spacy_model
        self.ollama_model = ollama_model
        self.ollama_endpoint = ollama_endpoint
        # Performance / config parameters
        self.chunk_size = chunk_size
        self.max_workers = max_workers if max_workers is not None else os.cpu_count() or 4
        self.progress_interval = progress_interval
        self.cache_path = cache_path
        self.relationship_fallback = relationship_fallback

        # Detect if this is a HuggingFace model (contains "/" or starts with common HF prefixes)
        self.use_huggingface = False
        self.hf_pipeline = None
        if ollama_model and ('/' in ollama_model or any(ollama_model.startswith(prefix) for prefix in ['meta-', 'google/', 'microsoft/', 'facebook/'])):
            self.use_huggingface = True
            logger.info(f"Detected HuggingFace model: {ollama_model}")
            logger.info("Note: If HuggingFace download fails, the system will automatically try Ollama as fallback")

        self.nlp = None
        self.linker = None
        self.lm = None
        self.batch_predictor = None
        self.semantic_types = []
        # Ensure attributes exist even if initialization fails later
        self.embedder = None
        self.summarizer = None

        # Global description cache and lock
        self.desc_cache: Dict[str, str] = {}
        self.cache_lock = threading.Lock()
        self._load_description_cache()
        
        # Initialize SciSpacy with entity linker (with caching)
        try:
            # Check if model is already cached
            if spacy_model not in self._nlp_cache:
                logger.info(f"Loading SciSpacy model: {spacy_model} (this may take a moment...)")
                # Use faster loading with disable components we don't need
                self._nlp_cache[spacy_model] = spacy.load(spacy_model, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
                
                # Add entity linker to the pipeline
                if "scispacy_linker" not in self._nlp_cache[spacy_model].pipe_names:
                    self._nlp_cache[spacy_model].add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
                
                logger.info(f"✓ SciSpacy model cached: {spacy_model}")
            else:
                logger.info(f"✓ Using cached SciSpacy model: {spacy_model}")
            
            self.nlp = self._nlp_cache[spacy_model]
            self.linker = self.nlp.get_pipe("scispacy_linker")
            logger.info(f"✓ SciSpacy initialized with model: {spacy_model} and entity linker")
        except Exception as e:
            logger.error(f"Failed to initialize SciSpacy with model {spacy_model}: {e}")
            raise RuntimeError(f"Could not initialize SciSpacy with model {spacy_model}")
        
        # Initialize LLM backend (HuggingFace or Ollama)
        if self.ollama_model:
            if self.use_huggingface:
                # Initialize HuggingFace model with retries and better error handling
                max_retries = 3
                retry_delay = 5
                
                for retry in range(max_retries):
                    try:
                        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                        import torch
                        import time
                        
                        logger.info(f"Initializing HuggingFace model: {self.ollama_model} (attempt {retry + 1}/{max_retries})")
                        
                        # Try to load tokenizer first (smaller download)
                        try:
                            # First try local files only
                            try:
                                tokenizer = AutoTokenizer.from_pretrained(
                                    self.ollama_model, 
                                    trust_remote_code=True,
                                    local_files_only=True  # Try local first
                                )
                                logger.info("Using locally cached tokenizer")
                            except Exception:
                                # If not available locally, download
                                logger.info("Downloading tokenizer from HuggingFace...")
                                tokenizer = AutoTokenizer.from_pretrained(
                                    self.ollama_model, 
                                    trust_remote_code=True,
                                    local_files_only=False,
                                    resume_download=True  # Resume partial downloads
                                )
                        except Exception as tok_err:
                            logger.warning(f"Failed to load tokenizer: {tok_err}")
                            # Try with a smaller, fallback model
                            if retry < max_retries - 1:
                                logger.info(f"Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                continue
                            else:
                                raise
                        
                        # Set padding token if not present
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                        
                        # Determine device and dtype
                        device = 0 if torch.cuda.is_available() else -1
                        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                        
                        logger.info("Loading model weights (this may take a while for large models)...")
                        
                        # Try loading model with different strategies
                        try:
                            # First try local files only
                            try:
                                model = AutoModelForCausalLM.from_pretrained(
                                    self.ollama_model,
                                    torch_dtype=dtype,
                                    device_map="auto" if torch.cuda.is_available() else None,
                                    trust_remote_code=True,
                                    local_files_only=True,  # Try local first
                                    low_cpu_mem_usage=True
                                )
                                logger.info("Using locally cached model")
                            except Exception:
                                # If not available locally, try different download strategies
                                logger.info("Downloading model from HuggingFace (this may take a while)...")
                                logger.info("Note: Large models may fail due to connection issues. Consider using Ollama instead.")
                                
                                # Try different strategies for downloading
                                download_strategies = [
                                    # Strategy 1: Standard download with resume
                                    {"resume_download": True, "force_download": False},
                                    # Strategy 2: Force fresh download
                                    {"resume_download": False, "force_download": True},
                                    # Strategy 3: Use different connection settings
                                    {"resume_download": True, "force_download": False, "use_auth_token": False}
                                ]
                                
                                model = None
                                for i, strategy in enumerate(download_strategies):
                                    try:
                                        logger.info(f"Trying download strategy {i+1}/{len(download_strategies)}")
                                        model = AutoModelForCausalLM.from_pretrained(
                                            self.ollama_model,
                                            torch_dtype=dtype,
                                            device_map="auto" if torch.cuda.is_available() else None,
                                            trust_remote_code=True,
                                            local_files_only=False,
                                            low_cpu_mem_usage=True,
                                            **strategy
                                        )
                                        logger.info(f"Download strategy {i+1} succeeded")
                                        break
                                    except Exception as strategy_err:
                                        logger.warning(f"Download strategy {i+1} failed: {strategy_err}")
                                        if i < len(download_strategies) - 1:
                                            logger.info("Trying next strategy...")
                                            continue
                                        else:
                                            raise strategy_err
                        except Exception as model_err:
                            if "connection" in str(model_err).lower() or "timeout" in str(model_err).lower():
                                logger.warning(f"Connection error loading model: {model_err}")
                                if retry < max_retries - 1:
                                    logger.info(f"Retrying in {retry_delay} seconds...")
                                    time.sleep(retry_delay)
                                    continue
                            else:
                                raise
                        
                        # Create pipeline with more diverse generation settings
                        self.hf_pipeline = pipeline(
                            "text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            device=device if device >= 0 else None,
                            max_new_tokens=2048,
                            # Use default temperature/sampling here, override per call
                            return_full_text=False
                        )
                        
                        logger.info(f"✓ HuggingFace model initialized: {self.ollama_model}")
                        break  # Success, exit retry loop
                        
                    except ImportError as e:
                        logger.error(f"Missing required libraries for HuggingFace: {e}")
                        logger.info("Install with: pip install transformers torch accelerate")
                        self.use_huggingface = False
                        self.hf_pipeline = None
                        break
                        
                    except Exception as e:
                        logger.error(f"Failed to initialize HuggingFace model (attempt {retry + 1}): {e}")
                        
                        if retry == max_retries - 1:
                            logger.warning("All attempts failed. Checking for Ollama fallback...")
                            
                            # Try to fallback to a similar Ollama model if available
                            ollama_fallback_models = ["llama3.1:8b", "llama3.2:3b", "llama2:7b"]
                            
                            for fallback_model in ollama_fallback_models:
                                try:
                                    # Test if Ollama is available and the model exists
                                    import requests
                                    test_url = self.ollama_endpoint.rstrip('/') + "/api/generate"
                                    test_payload = {
                                        "model": fallback_model,
                                        "prompt": "test",
                                        "stream": False,
                                        "options": {"num_predict": 1}
                                    }
                                    resp = requests.post(test_url, json=test_payload, timeout=10)
                                    if resp.status_code == 200:
                                        logger.info(f"Found Ollama fallback model: {fallback_model}")
                                        self.ollama_model = fallback_model
                                        self.use_huggingface = False
                                        self.hf_pipeline = None
                                        logger.info(f"Successfully switched to Ollama model: {fallback_model}")
                                        break
                                except:
                                    continue
                            else:
                                # No Ollama fallback available
                                logger.warning("No Ollama fallback available. Continuing without LLM.")
                                logger.info("Tip: You can try downloading the HF model manually first with:")
                                logger.info(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
                                logger.info(f"  AutoTokenizer.from_pretrained('{self.ollama_model}')")
                                logger.info(f"  AutoModelForCausalLM.from_pretrained('{self.ollama_model}')")
                                logger.info("Or install Ollama with a local model as fallback.")
                                self.use_huggingface = False
                                self.hf_pipeline = None
                                self.ollama_model = None  # Disable LLM completely
                        else:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
            else:
                logger.info(f"ℹ️ Using direct Ollama HTTP calls for LLM generation with model: {self.ollama_model}")
        else:
            logger.info("ℹ️ LLM model not provided; skipping LLM-based descriptions and relationships")

        # Initialize BioMedBERT embedder for entity embeddings
        try:
            self.embedder = BioMedBERTEmbedder(batch_size=32)  # Smaller batch size for memory efficiency
            logger.info(f"✓ BioMedBERT embedder initialized")
        except Exception as e:
            logger.error(f"Failed to initialize BioMedBERT embedder: {e}")
            logger.warning("Continuing without embedding functionality")
            self.embedder = None

        # Initialize EntitySummarizer (medical tag generator)
        try:
            # Pass the model info to the summarizer
            if self.use_huggingface:
                self.summarizer = EntitySummarizer(
                    provider="hf",
                    model_id=self.ollama_model
                )
            else:
                self.summarizer = EntitySummarizer(
                    ollama_model=self.ollama_model if self.ollama_model else "llama3.1:8b",
                    ollama_endpoint=self.ollama_endpoint
                )
            logger.info("✓ EntitySummarizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize EntitySummarizer: {e}")
            self.summarizer = None

    def _ollama_generate_processed_entities(self, context_text: str, entity_input_list: list[dict], 
                                          max_retries: int = 3) -> dict:
        """Call LLM (Ollama or HuggingFace) to generate JSON of entities and relationships for the given context and entities."""
        if not self.ollama_model:
            logger.warning("No LLM model specified, returning empty relationships")
            return {"entities": [], "relationships": []}

        for attempt in range(max_retries):
            try:
                logger.debug(f"LLM generation attempt {attempt + 1}/{max_retries}")
                
                # Build a simpler, clearer prompt for relationship generation
                entity_names = [e["entity_name"] for e in entity_input_list]
                
                # Create different prompts for HuggingFace vs Ollama to optimize for each model type
                if self.use_huggingface and self.hf_pipeline:
                    # More detailed prompt for HuggingFace to encourage specific descriptions
                    prompt = f"""You are a medical expert analyzing relationships between entities in clinical text.

Text: {context_text[:1000]}...

Entities found: {', '.join(entity_names)}

Your task: Identify SPECIFIC medical relationships between these entities. Each relationship should be unique and descriptive.

Requirements:
- Be SPECIFIC about how entities relate (not just "is related to")
- Use medical terminology when appropriate
- Make each relationship description unique and meaningful
- Relationship strength: 5-10 (5=weak connection, 10=strong direct connection)

Return ONLY this JSON structure:
{{
  "processed_entities": {{
    "entities": [
      {{"entity_name": "entity1", "entity_description": "brief medical description"}}
    ],
    "relationships": [
      {{
        "source_entity": "entity1",
        "target_entity": "entity2", 
        "relationship_description": "specific medical relationship (e.g., 'surgery treats condition', 'medication manages symptoms', 'device monitors function')",
        "relationship_strength": 8
      }}
    ]
  }}
}}

Generate diverse, specific relationship descriptions. Avoid generic terms like "related to" or "associated with"."""
                else:
                    # Simpler prompt for Ollama (which already generates diverse outputs)
                    prompt = f"""Given this medical text and list of entities, identify relationships between entities.

Text: {context_text[:1000]}...

Entities: {', '.join(entity_names)}

Generate relationships between these entities. Return ONLY a JSON object with this exact structure:
{{
  "processed_entities": {{
    "entities": [
      {{"entity_name": "entity1", "entity_description": "brief description"}}
    ],
    "relationships": [
      {{
        "source_entity": "entity1",
        "target_entity": "entity2", 
        "relationship_description": "how entity1 relates to entity2",
        "relationship_strength": 7
      }}
    ]
  }}
}}

Rules:
- Create relationships between entities that are meaningfully connected in the text
- relationship_strength must be between 5-10 (5=weak, 10=strong)
- Include multiple relationships if entities are connected in different ways
- Focus on medical/clinical relationships
- Return ONLY the JSON, no other text"""

                # Call the appropriate LLM backend
                if self.use_huggingface and self.hf_pipeline:
                    # Use HuggingFace pipeline
                    logger.debug(f"Sending request to HuggingFace model: {len(entity_input_list)} entities")
                    
                    # Generate with HuggingFace using more diverse parameters to match Ollama
                    outputs = self.hf_pipeline(
                        prompt,
                        max_new_tokens=2048,
                        temperature=0.3,  # Higher temperature for more creativity (closer to Ollama's behavior)
                        do_sample=True,
                        top_p=0.9,  # Nucleus sampling for more diverse outputs
                        top_k=50,   # Top-k sampling to maintain quality while adding diversity
                        repetition_penalty=1.1,  # Reduce repetitive text
                        no_repeat_ngram_size=2,  # Prevent exact n-gram repetition
                        return_full_text=False,
                        pad_token_id=self.hf_pipeline.tokenizer.eos_token_id
                    )
                    
                    raw_text = outputs[0]['generated_text'] if outputs else ""
                    logger.debug(f"HuggingFace raw response (first 200 chars): {raw_text[:200]}")
                    
                else:
                    # Use Ollama HTTP API
                    url = self.ollama_endpoint.rstrip('/') + "/api/generate"
                    payload = {
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 8000}
                    }
                    
                    # Longer timeout for larger contexts
                    logger.debug(f"Sending request to Ollama: {len(entity_input_list)} entities")
                    resp = requests.post(url, json=payload, timeout=300)
                    resp.raise_for_status()
                    data = resp.json()
                    raw_text = data.get("response", "")
                    
                    logger.debug(f"Ollama raw response (first 200 chars): {raw_text[:200]}")
                
                result = self._parse_llm_response(raw_text)
                
                # Validate that we got meaningful relationships
                relationships = result.get("relationships", [])
                if len(entity_input_list) >= 2 and not relationships:
                    logger.warning(f"No relationships generated for {len(entity_input_list)} entities on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                
                # Filter relationships by minimum strength
                filtered_relationships = []
                for rel in relationships:
                    strength = rel.get("relationship_strength", 0)
                    if isinstance(strength, (int, float)) and strength >= 5:
                        filtered_relationships.append(rel)
                    else:
                        logger.debug(f"Filtered out relationship with strength {strength}: {rel}")
                
                result["relationships"] = filtered_relationships
                
                logger.debug(f"Successfully generated {len(result.get('entities', []))} entities and {len(filtered_relationships)} relationships")
                return result
                
            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP request failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} LLM attempts failed")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on attempt {attempt + 1}: {e}")
                if self.use_huggingface:
                    logger.error(f"Raw response: {raw_text[:500] if 'raw_text' in locals() else 'N/A'}")
                else:
                    logger.error(f"Raw response: {data.get('response', '')[:500] if 'data' in locals() else 'N/A'}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} JSON parsing attempts failed")
            except Exception as e:
                logger.error(f"Unexpected LLM error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} LLM attempts failed with unexpected errors")

        # Return empty result if all attempts failed
        logger.warning("LLM generation completely failed, returning empty result")
        return {"entities": [], "relationships": []}
    
    def load_semantic_types(self, txt_path: str) -> None:
        """
        Load semantic types from pipe-delimited text file and prepare for cosine similarity search.
        
        Args:
            txt_path: Path to the joined text file (pipe-delimited: cui|entity_name|entity_type|definition|TUI)
        """
        try:
            # Load the pipe-delimited text file
            semantic_types = []
            semantic_type_definitions = {}
            semantic_type_categories = {}
            semantic_type_cuis = {}
            semantic_type_tuis = {}
            
            line_count = 0
            successful_entries = 0
            
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    line = line.strip()
                    if line and '|' in line:
                        parts = line.split('|', 4)  # Split into 5 parts: cui, entity_name, entity_type, definition, TUI
                        if len(parts) >= 4:
                            cui = parts[0].strip()
                            entity_name = parts[1].strip()
                            entity_type = parts[2].strip()
                            definition = parts[3].strip() if len(parts) > 3 else ""
                            tui = parts[4].strip() if len(parts) > 4 else ""
                            
                            # Use entity_name as the key
                            semantic_types.append(entity_name)
                            semantic_type_definitions[entity_name] = definition
                            semantic_type_categories[entity_name] = entity_type
                            semantic_type_cuis[entity_name] = cui
                            semantic_type_tuis[entity_name] = tui
                            successful_entries += 1
                        else:
                            logger.debug(f"Skipping malformed line {line_count}: {line[:100]}...")
            
            self.semantic_types = semantic_types
            self.semantic_type_definitions = semantic_type_definitions
            self.semantic_type_categories = semantic_type_categories
            self.semantic_type_cuis = semantic_type_cuis
            self.semantic_type_tuis = semantic_type_tuis
            
            # Create reverse CUI lookup for faster access
            self.cui_to_semantic_name = {cui: name for name, cui in semantic_type_cuis.items()}
            
            logger.info(f"✓ Loaded {len(self.semantic_types)} semantic types from {successful_entries}/{line_count} lines")
            logger.info(f"✓ Created reverse CUI lookup with {len(self.cui_to_semantic_name)} entries")
            
            # Log some sample mappings for debugging
            if len(self.semantic_types) > 0:
                sample_names = list(self.semantic_types)[:5]
                logger.debug("Sample semantic type mappings:")
                for name in sample_names:
                    cui = self.semantic_type_cuis.get(name, "")
                    etype = self.semantic_type_categories.get(name, "")
                    logger.debug(f"  '{name}' -> CUI: {cui}, Type: {etype}")
                
                # Specific debug check for the problematic CUI
                test_cui = "C0011900"
                if test_cui in self.cui_to_semantic_name:
                    test_name = self.cui_to_semantic_name[test_cui]
                    test_type = self.semantic_type_categories.get(test_name, "NOT_FOUND")
                    logger.debug(f"Debug CUI {test_cui}: name='{test_name}', type='{test_type}'")
                else:
                    logger.debug(f"Debug CUI {test_cui}: NOT FOUND in cui_to_semantic_name mapping")
            
        except Exception as e:
            logger.error(f"Failed to load semantic types: {e}")
            raise
    
    def _chunk_list(self, data_list: List[Any], chunk_size: int) -> "Generator[List[Any], None, None]":
        """Utility to split a list into chunks of size chunk_size."""
        for i in range(0, len(data_list), chunk_size):
            yield data_list[i:i + chunk_size]

    def batch_process_entities(self, paragraph_content: str, detected_entities: List[Dict], chunk_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Process entities in a paragraph using LLM for descriptions only (no paragraph-level relationships).
        
        Args:
            paragraph_content: The full paragraph content
            detected_entities: List of entities detected by SciSpacy with their metadata
            
        Returns:
            Dictionary containing processed entities (relationships will be generated at chunk level)
        """
        try:
            if chunk_size is None:
                chunk_size = self.chunk_size

            # We'll aggregate entities across chunks
            aggregated_entities: List[Dict[str, Any]] = []

            # Pre-fill entity descriptions from cache
            entity_descriptions: Dict[str, str] = {
                ent_name: desc for ent_name, desc in self.desc_cache.items()
                if any(e["entity_name"] == ent_name for e in detected_entities)
            }

            # Process entities for descriptions only (no relationships at paragraph level)
            for chunk_index, entity_chunk in enumerate(self._chunk_list(detected_entities, chunk_size)):
                processed_data = {"entities": [], "relationships": []}
                if self.ollama_model:
                    # Prepare entity list for LLM - focus on descriptions
                    entity_input_list = [
                        {
                            "entity_name": ent["entity_name"],
                            "spacy_label": ent["spacy_label"]
                        } for ent in entity_chunk
                    ]

                    logger.debug(
                        f"[Batch {chunk_index+1}] Sending {len(entity_chunk)} entities to LLM for description generation"
                    )

                    processed_data = self._ollama_generate_processed_entities(
                        paragraph_content, entity_input_list
                    )

                # Process entities: use LLM descriptions for uncached entities, cache for cached ones
                for entity_info in processed_data.get("entities", []):
                    entity_name = entity_info["entity_name"]
                    llm_description = entity_info.get("entity_description", "")
                    
                    # Use cached description if available, otherwise use LLM description
                    if entity_name in entity_descriptions:
                        final_description = entity_descriptions[entity_name]
                    else:
                        final_description = llm_description
                        # Update cache with new description
                        entity_descriptions[entity_name] = final_description
                        with self.cache_lock:
                            self.desc_cache[entity_name] = final_description
                    
                    # Compute embeddings for the entity
                    entity_embedding = None
                    if self.embedder is not None:
                        try:
                            # Find the original entity to get entity_type
                            original_entity = next((e for e in detected_entities if e["entity_name"] == entity_name), None)
                            if original_entity:
                                entity_type = original_entity.get("entity_type", "Entity")
                                entity_embedding = self.embedder.encode_entity(entity_name, entity_type, final_description)
                        except Exception as e:
                            logger.warning(f"Failed to compute embedding for entity '{entity_name}': {e}")
                            entity_embedding = None
                    
                    # Add to aggregated entities with embedding (empty list if failed/unavailable)
                    aggregated_entities.append({
                        "entity_name": entity_name,
                        "entity_description": final_description,
                        "content_embedding": entity_embedding.tolist() if entity_embedding is not None else []
                    })

            # Ensure all entities have descriptions (fallback for any missing)
            for entity in detected_entities:
                entity_name = entity["entity_name"]
                if entity_name not in entity_descriptions or not entity_descriptions[entity_name]:
                    description = ""
                    if self.ollama_model:
                        # Fallback single-entity call to ensure description
                        single_data = self._ollama_generate_processed_entities(
                            paragraph_content,
                            [{"entity_name": entity_name, "spacy_label": entity["spacy_label"]}]
                        )
                        if single_data.get("entities"):
                            description = single_data["entities"][0].get("entity_description", "")
                    entity_descriptions[entity_name] = description
                    # Update cache
                    with self.cache_lock:
                        self.desc_cache[entity_name] = description
                    
                    # Add to aggregated entities if not already present
                    if not any(e["entity_name"] == entity_name for e in aggregated_entities):
                        # Compute embeddings for fallback entities
                        entity_embedding = None
                        if self.embedder is not None:
                            try:
                                entity_type = entity.get("entity_type", "Entity")
                                entity_embedding = self.embedder.encode_entity(entity_name, entity_type, description)
                            except Exception as e:
                                logger.warning(f"Failed to compute embedding for fallback entity '{entity_name}': {e}")
                                entity_embedding = None
                        
                        aggregated_entities.append({
                            "entity_name": entity_name,
                            "entity_description": description,
                            "content_embedding": entity_embedding.tolist() if entity_embedding is not None else []
                        })

            # Persist cache after updates
            self._save_description_cache()

            # Return processed entities with descriptions and embeddings (no paragraph-level relationships)
            return {
                "entity_descriptions": entity_descriptions,
                "entities_with_embeddings": aggregated_entities
            }
            
        except Exception as e:
            logger.warning(f"Failed to batch process entities: {e}")
            logger.warning(f"Exception type: {type(e).__name__}")
            logger.warning(f"Exception details: {str(e)}")
            # Fallback to empty results
            return {
                "entity_descriptions": {},
                "entities_with_embeddings": []
            }

    def _generate_chunk_relationships(self, chunk_text: str, all_entities: List[Dict[str, Any]], 
                                    max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Generate relationships between ALL entities within a chunk using Ollama with fallback strategies.
        
        Args:
            chunk_text: Combined text content of all paragraphs in the chunk
            all_entities: All entities from all paragraphs in the chunk
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of relationships between entities in the chunk
        """
        try:
            if len(all_entities) < 2:
                logger.debug("Chunk has fewer than 2 entities, no relationships possible")
                return []
            
            # Create unique entity list for the chunk
            unique_entities = {}
            for ent in all_entities:
                name = ent.get('entity_name')
                if name and name not in unique_entities:
                    unique_entities[name] = ent
            
            unique_entity_list = list(unique_entities.values())
            logger.info(f"Generating chunk-level relationships for {len(unique_entity_list)} unique entities")
            
            # Try Ollama first if available
            if self.ollama_model:
                try:
                    entity_input_list = [
                        {"entity_name": ent["entity_name"], "spacy_label": ent.get("spacy_label", "")}
                        for ent in unique_entity_list
                    ]
                    
                    logger.debug(f"Attempting Ollama relationship generation for chunk with {len(entity_input_list)} entities")
                    processed_data = self._ollama_generate_processed_entities(
                        chunk_text, entity_input_list, max_retries=max_retries
                    )
                    
                    relationships = processed_data.get('relationships', [])
                    
                    if relationships:
                        logger.info(f"Successfully generated {len(relationships)} chunk-level relationships via Ollama")
                        return relationships
                    else:
                        logger.warning("Ollama returned no relationships")
                        
                except Exception as e:
                    logger.error(f"Ollama chunk relationship generation failed: {e}")
                    logger.warning("LLM relationship generation failed")
            
            # Optional heuristic fallback
            if self.relationship_fallback:
                logger.info("Using heuristic fallback for chunk-level relationship generation (relationship_fallback=True)")
                relationships = self._generate_heuristic_relationships(unique_entity_list, chunk_text)
                # Filter by minimum strength
                filtered_relationships = [rel for rel in relationships if rel.get("relationship_strength", 0) >= 5]
                logger.info(f"Generated {len(filtered_relationships)} chunk-level relationships via heuristic method")
                return filtered_relationships

            # No heuristic fallback: prefer free-form LLM only
            logger.info("No relationships generated and heuristic fallback disabled; returning empty list")
            return []
            
        except Exception as e:
            logger.error(f"Failed to generate chunk relationships: {e}")
            return []

    def _generate_heuristic_relationships(self, entities: List[Dict[str, Any]], chunk_text: str) -> List[Dict[str, Any]]:
        """
        Generate relationships using heuristic methods based on entity types and co-occurrence.
        
        Args:
            entities: List of entities in the chunk
            chunk_text: Text content of the chunk
            
        Returns:
            List of heuristically generated relationships
        """
        relationships = []
        
        try:
            # Create entity type mapping
            entity_types = {}
            for ent in entities:
                name = ent.get("entity_name", "")
                etype = ent.get("entity_type", "").lower()
                spacy_label = ent.get("spacy_label", "").lower()
                entity_types[name] = {"type": etype, "label": spacy_label}
            
            entity_names = list(entity_types.keys())
            
            # Generate relationships based on medical domain knowledge
            for i in range(len(entity_names)):
                for j in range(i + 1, len(entity_names)):
                    source = entity_names[i]
                    target = entity_names[j]
                    
                    source_info = entity_types[source]
                    target_info = entity_types[target]
                    
                    # Calculate relationship strength and description
                    strength, description = self._calculate_relationship_strength(
                        source, target, source_info, target_info, chunk_text
                    )
                    
                    if strength >= 5:  # Only include meaningful relationships
                        relationships.append({
                            "source_entity": source,
                            "target_entity": target,
                            "relationship_description": description,
                            "relationship_strength": strength
                        })
            
            return relationships
            
        except Exception as e:
            logger.error(f"Heuristic relationship generation failed: {e}")
            return []

    def _calculate_relationship_strength(self, source: str, target: str, 
                                       source_info: Dict, target_info: Dict, 
                                       chunk_text: str) -> tuple[int, str]:
        """
        Calculate relationship strength and description based on entity types and context.
        
        Returns:
            Tuple of (strength, description)
        """
        try:
            # Base strength calculation
            strength = 5  # Minimum acceptable strength
            
            source_type = source_info.get("type", "").lower()
            target_type = target_info.get("type", "").lower()
            source_label = source_info.get("label", "").lower()
            target_label = target_info.get("label", "").lower()
            
            # Medical domain relationship patterns
            medical_relationships = {
                # High strength relationships (8-10)
                ("drug", "disease"): (9, "treats or is used for"),
                ("drug", "disorder"): (9, "treats or manages"),
                ("procedure", "disease"): (8, "is performed to diagnose or treat"),
                ("procedure", "anatomical"): (8, "is performed on"),
                ("symptom", "disease"): (8, "is a manifestation of"),
                
                # Medium-high strength relationships (7-8)
                ("anatomical", "disease"): (7, "is affected by"),
                ("drug", "anatomical"): (7, "acts on or affects"),
                ("laboratory", "disease"): (7, "is used to diagnose or monitor"),
                
                # Medium strength relationships (6-7)
                ("device", "procedure"): (6, "is used in"),
                ("device", "anatomical"): (6, "is applied to"),
                
                # Default medium-low strength (5-6)
                ("default", "default"): (5, "is related to")
            }
            
            # Check for specific relationship patterns
            relationship_key = None
            for pattern, (str_val, desc_template) in medical_relationships.items():
                if pattern == ("default", "default"):
                    continue
                    
                # Check both directions
                if (pattern[0] in source_type or pattern[0] in source_label) and \
                   (pattern[1] in target_type or pattern[1] in target_label):
                    relationship_key = pattern
                    break
                elif (pattern[1] in source_type or pattern[1] in source_label) and \
                     (pattern[0] in target_type or pattern[0] in target_label):
                    relationship_key = (pattern[1], pattern[0])
                    break
            
            if relationship_key and relationship_key in medical_relationships:
                strength, description_template = medical_relationships[relationship_key]
                description = f"{source} {description_template} {target}"
            else:
                # Default relationship
                strength = 5
                description = f"{source} is related to {target} within the same medical context"
            
            # Boost strength if entities appear close together in text
            if self._entities_appear_close(source, target, chunk_text):
                strength = min(10, strength + 1)
                description += " (co-located in text)"
            
            return strength, description
            
        except Exception as e:
            logger.warning(f"Failed to calculate relationship strength: {e}")
            return 5, f"{source} is related to {target}"

    def _map_spacy_label_to_type(self, spacy_label: str) -> str:
        """Map SciSpacy entity labels to more specific entity types."""
        label_mappings = {
            # Medical procedures and treatments
            "ENTITY": "Medical Entity",  # Generic fallback but better than "Entity"
            
            # Common SciSpacy labels for medical domain
            "PROCEDURE": "Diagnostic Procedure",
            "TREATMENT": "Therapeutic Procedure", 
            "SURGERY": "Surgical Procedure",
            "MEDICATION": "Pharmacologic Substance",
            "DRUG": "Pharmacologic Substance",
            "DISEASE": "Disease or Syndrome",
            "CONDITION": "Disease or Syndrome",
            "SYMPTOM": "Sign or Symptom",
            "ANATOMY": "Body Part, Organ, or Organ Component",
            "DEVICE": "Medical Device",
            "TEST": "Laboratory Procedure",
            "LAB": "Laboratory Procedure",
            "DIAGNOSTIC": "Diagnostic Procedure"
        }
        
        # Try exact match first
        if spacy_label in label_mappings:
            return label_mappings[spacy_label]
        
        # Try partial matching for complex labels
        spacy_lower = spacy_label.lower()
        for key, value in label_mappings.items():
            if key.lower() in spacy_lower or spacy_lower in key.lower():
                return value
                
        return "Medical Entity"  # Better fallback than just "Entity"
    
    def _entities_appear_close(self, entity1: str, entity2: str, text: str, 
                             window_size: int = 100) -> bool:
        """Check if two entities appear within a certain character window of each other."""
        try:
            text_lower = text.lower()
            entity1_lower = entity1.lower()
            entity2_lower = entity2.lower()
            
            # Find all positions of both entities
            positions1 = [i for i in range(len(text_lower)) if text_lower[i:].startswith(entity1_lower)]
            positions2 = [i for i in range(len(text_lower)) if text_lower[i:].startswith(entity2_lower)]
            
            # Check if any positions are within the window
            for pos1 in positions1:
                for pos2 in positions2:
                    if abs(pos1 - pos2) <= window_size:
                        return True
            
            return False
            
        except Exception:
            return False

    def _parse_llm_response(self, raw) -> dict:
        """
        Attempts to coerce the LLM output into a Python dict with enhanced error logging.
        """
        try:
            # 0. If already a dict we're done
            if isinstance(raw, dict):
                data = raw
            else:
                txt = str(raw).strip()
                logger.debug(f"Parsing LLM response (length: {len(txt)})")

                # Drop code fences or leading markdown
                txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.IGNORECASE).strip()

                # 1st attempt – vanilla JSON
                try:
                    data = json.loads(txt)
                    logger.debug("Successfully parsed response as JSON")
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parsing failed: {e}")
                    # 2nd attempt – replace single quotes
                    try:
                        data = json.loads(txt.replace("'", '"'))
                        logger.debug("Successfully parsed response after quote replacement")
                    except json.JSONDecodeError as e2:
                        logger.debug(f"JSON parsing with quote replacement failed: {e2}")
                        # 3rd attempt – Python literal
                        try:
                            data = ast.literal_eval(txt)
                            logger.debug("Successfully parsed response as Python literal")
                        except Exception as e3:
                            logger.warning(f"All parsing attempts failed: JSON={e}, Quote={e2}, Literal={e3}")
                            logger.warning(f"Raw response: {txt[:500]}")
                            data = {}

            # Handle case where LLM returns a list instead of dict
            if isinstance(data, list):
                logger.debug("LLM returned a list, attempting to extract relationships")
                # If it's a list of relationships, wrap it properly
                if data and isinstance(data[0], dict) and any(k in data[0] for k in ['source_entity', 'target_entity']):
                    data = {"entities": [], "relationships": data}
                else:
                    logger.warning("List format not recognized as relationships")
                    data = {"entities": [], "relationships": []}
            
            # Unwrap if the model kept the outer key
            if isinstance(data, dict):
                if "processed_entities" in data:
                    data = data["processed_entities"]
                
                # Guarantee keys exist
                result = {
                    "entities": data.get("entities", []),
                    "relationships": data.get("relationships", [])
                }
            else:
                result = {"entities": [], "relationships": []}
            
            logger.debug(f"Parsed result: {len(result['entities'])} entities, {len(result['relationships'])} relationships")
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in LLM response parsing: {e}")
            logger.error(f"Error details: {str(e)}")
            return {"entities": [], "relationships": []}

    # ------------------ Cache helpers ------------------
    def _load_description_cache(self) -> None:
        """Load entity description cache from disk if it exists."""
        try:
            cache_dir = os.path.dirname(self.cache_path)
            os.makedirs(cache_dir, exist_ok=True)
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.desc_cache = json.load(f)
                    logger.info(f"✓ Loaded description cache with {len(self.desc_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to load description cache: {e}")
            self.desc_cache = {}

    def _save_description_cache(self) -> None:
        """Persist the description cache to disk (thread-safe)."""
        try:
            with self.cache_lock:
                cache_dir = os.path.dirname(self.cache_path)
                os.makedirs(cache_dir, exist_ok=True)
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.desc_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save description cache: {e}")

    def _save_progress(self, results: Dict[str, Any], current_paragraph: int, total_paragraphs: int) -> None:
        """
        Save current processing progress to a JSON file.
        
        Args:
            results: Current processing results
            current_paragraph: Number of paragraphs processed so far
            total_paragraphs: Total number of paragraphs to process
        """
        try:
            # Create progress filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            progress_filename = f"entity_processing_progress_{timestamp}.json"
            
            # Add progress metadata
            progress_results = results.copy()
            progress_results['progress_metadata'] = {
                'paragraphs_processed': current_paragraph,
                'total_paragraphs': total_paragraphs,
                'completion_percentage': round((current_paragraph / total_paragraphs) * 100, 2),
                'saved_at': timestamp,
                'status': 'in_progress'
            }
            
            # Save to file
            with open(progress_filename, 'w', encoding='utf-8') as f:
                json.dump(progress_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Progress saved: {progress_filename} ({current_paragraph}/{total_paragraphs} paragraphs, {progress_results['progress_metadata']['completion_percentage']}% complete)")
            
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")
    
    def extract_entities_from_paragraph(self, paragraph_content: str) -> Dict[str, Any]:
        """
        Extract clinically relevant entities from a paragraph using SciSpacy and batch LLM processing.
        No longer generates paragraph-level relationships.
        
        Args:
            paragraph_content: The paragraph content to analyze
            
        Returns:
            Dictionary containing entities list only (no paragraph-level relationships)
        """
        try:
            # Process the paragraph with SciSpacy
            doc = self.nlp(paragraph_content)
            
            # First pass: Extract basic entity information from SciSpacy
            detected_entities = []
            for ent in doc.ents:
                # Skip entities that are too short
                if len(ent.text.strip()) < 2:
                    continue
                
                # Extract CUI using SciSpacy EntityLinker
                entity_cui = ""
                entity_score = 0.0
                if len(ent._.kb_ents) > 0:
                    entity_cui, entity_score = ent._.kb_ents[0]
                    logger.debug(f"Found CUI for '{ent.text}': {entity_cui} (score: {entity_score})")
                
                # Get entity type and UMLS definition from joined.txt using the CUI if available
                entity_type = "Entity"  # Default fallback
                umls_definition = ""  # Default empty
                if entity_cui:
                    # Method 1: Direct CUI lookup using pre-built reverse mapping
                    if entity_cui in self.cui_to_semantic_name:
                        entity_name_from_cui = self.cui_to_semantic_name[entity_cui]
                        entity_type = self.semantic_type_categories.get(entity_name_from_cui, "Entity")
                        umls_definition = self.semantic_type_definitions.get(entity_name_from_cui, "")
                        logger.debug(f"Found entity type via CUI lookup: '{ent.text}' (CUI: {entity_cui}) -> '{entity_name_from_cui}' -> {entity_type}")
                        
                        # Debug: Check if the mapping failed
                        if entity_type == "Entity":
                            logger.debug(f"CUI mapping failed: entity_name_from_cui='{entity_name_from_cui}' not found in semantic_type_categories")
                            # Try to infer a better entity type from SciSpacy label as fallback
                            spacy_label = ent.label_
                            if spacy_label:
                                fallback_type = self._map_spacy_label_to_type(spacy_label)
                                if fallback_type != "Entity":
                                    entity_type = fallback_type
                                    logger.debug(f"Using SciSpacy label fallback: '{spacy_label}' -> '{entity_type}'")
                        else:
                            logger.debug(f"CUI mapping succeeded: '{ent.text}' -> '{entity_name_from_cui}' -> '{entity_type}'")
                    else:
                        # Method 2: Try partial/fuzzy matching on entity name if CUI lookup fails
                        entity_text_lower = ent.text.strip().lower()
                        best_match = None
                        best_score = 0
                        
                        # Look for exact name match first
                        for semantic_name in self.semantic_types:
                            if semantic_name.lower() == entity_text_lower:
                                best_match = semantic_name
                                best_score = 1.0
                                break
                        
                        # If no exact match, try partial matching
                        if not best_match:
                            for semantic_name in self.semantic_types:
                                semantic_name_lower = semantic_name.lower()
                                # Check if entity text is contained in semantic name or vice versa
                                if (entity_text_lower in semantic_name_lower) or (semantic_name_lower in entity_text_lower):
                                    # Calculate simple overlap score
                                    overlap = len(set(entity_text_lower.split()) & set(semantic_name_lower.split()))
                                    total_words = len(set(entity_text_lower.split()) | set(semantic_name_lower.split()))
                                    score = overlap / max(total_words, 1) if total_words > 0 else 0
                                    
                                    if score > best_score and score > 0.3:  # Minimum threshold for partial match
                                        best_match = semantic_name
                                        best_score = score
                        
                        # Use best match if found
                        if best_match:
                            entity_type = self.semantic_type_categories.get(best_match, "Entity")
                            umls_definition = self.semantic_type_definitions.get(best_match, "")
                            logger.debug(f"Found entity type via name matching: '{ent.text}' -> '{best_match}' -> {entity_type} (score: {best_score:.2f})")
                        else:
                            logger.debug(f"No semantic type found for '{ent.text}' with CUI '{entity_cui}', using fallback")
                
                detected_entity = {
                    'entity_name': ent.text.strip(),
                    'entity_type': entity_type,
                    'umls_definition': umls_definition,
                    'cui': entity_cui,
                    'linker_score': entity_score,
                    'spacy_label': ent.label_,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                    'is_clinically_relevant': True
                }
                
                # Log when we're using fallback entity type for debugging
                if entity_type == "Entity" and entity_cui:
                    logger.warning(f"Entity '{ent.text}' has CUI '{entity_cui}' but using fallback type 'Entity' - check semantic types mapping")
                
                detected_entities.append(detected_entity)
            
            # Remove duplicates (keep first occurrence)
            seen = set()
            unique_detected_entities = []
            for entity in detected_entities:
                if entity['entity_name'].lower() not in seen:
                    seen.add(entity['entity_name'].lower())
                    unique_detected_entities.append(entity)
            
            # If no entities found, return empty structure
            if not unique_detected_entities:
                return {'entities': []}
            
            # Second pass: Use batch LLM processing for descriptions only
            batch_results = self.batch_process_entities(paragraph_content, unique_detected_entities)
            
            # Combine SciSpacy data with LLM results and embeddings
            final_entities = []
            entity_descriptions = batch_results.get("entity_descriptions", {})
            entities_with_embeddings = batch_results.get("entities_with_embeddings", [])
            
            # Create a lookup for entities with embeddings
            embedding_lookup = {e["entity_name"]: e for e in entities_with_embeddings}
            
            for entity in unique_detected_entities:
                entity_name = entity['entity_name']
                
                # Get description from batch processing or fallback
                entity_description = entity_descriptions.get(entity_name, "")
                
                # Get embedding from batch processing or compute it
                embedding = None
                if entity_name in embedding_lookup:
                    embedding = embedding_lookup[entity_name].get("content_embedding")
                
                # If no embedding from batch processing, try to compute it
                if not embedding and self.embedder is not None:
                    try:
                        entity_type = entity.get('entity_type', 'Entity')
                        entity_embedding = self.embedder.encode_entity(entity_name, entity_type, entity_description)
                        embedding = entity_embedding.tolist() if entity_embedding is not None else []
                    except Exception as e:
                        logger.warning(f"Failed to compute embedding for entity '{entity_name}': {e}")
                        embedding = []
                
                # Ensure embedding is always a list (empty if unavailable)
                if embedding is None:
                    embedding = []
                
                final_entity = {
                    'entity_name': entity_name,
                    'entity_description': entity_description,
                    'entity_type': entity['entity_type'],
                    'umls_definition': entity['umls_definition'],
                    'cui': entity['cui'],
                    'linker_score': entity['linker_score'],
                    'spacy_label': entity['spacy_label'],
                    'start_char': entity['start_char'],
                    'end_char': entity['end_char'],
                    'is_clinically_relevant': entity['is_clinically_relevant'],
                    'content_embedding': embedding
                }
                
                final_entities.append(final_entity)
            
            return {'entities': final_entities}
            
        except Exception as e:
            logger.error(f"Failed to extract entities from paragraph: {e}")
            return {'entities': []}
    
    def process_chunked_json(self, json_path: str, max_paragraphs: int = 20) -> Dict[str, Any]:
        """
        Process the chunked JSON file and extract entities from the first N paragraphs.
        Generate relationships at the chunk level only.
        
        Args:
            json_path: Path to the chunked JSON file
            max_paragraphs: Maximum number of paragraphs to process (default: 20)
            
        Returns:
            Dictionary containing the processing results with chunk-level relationships only
        """
        try:
            # Load the JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✓ Loaded JSON file with {len(data.get('chunks', []))} chunks")
            
            # Collect all paragraphs from chunks
            all_paragraphs = []
            for chunk in data.get('chunks', []):
                chunk_number = chunk.get('chunk_number')
                paragraphs = chunk.get('paragraphs', [])
                for idx, paragraph in enumerate(paragraphs, start=1):
                    if paragraph.get('type') == 'text' and paragraph.get('content', '').strip():
                        all_paragraphs.append({
                            'chunk_number': chunk_number,
                            'content': paragraph.get('content'),
                            'type': paragraph.get('type'),
                            'paragraph_index': idx
                        })
            
            # Limit to first N paragraphs
            paragraphs_to_process = all_paragraphs[:max_paragraphs]
            logger.info(f"Processing first {len(paragraphs_to_process)} paragraphs")
            
            # Process each paragraph for entity extraction only
            # Derive document name for metadata
            document_name = os.path.basename(json_path)

            results = {
                'metadata': {
                    'total_paragraphs_processed': len(paragraphs_to_process),
                    'spacy_model': self.spacy_model,
                    'ollama_model': self.ollama_model,
                    'semantic_types_loaded': len(self.semantic_types),
                    'document_name': document_name,
                    'embedding_model': getattr(self.embedder, 'model_name', None) if self.embedder is not None else None
                },
                'chunks': {}
            }

            # Process paragraphs for entities only (no medical summaries at paragraph level)
            def _process_single(idx_paragraph):
                idx, paragraph = idx_paragraph
                logger.info(f"[Thread] Processing paragraph {idx+1}/{len(paragraphs_to_process)}")

                extraction_results = self.extract_entities_from_paragraph(paragraph['content'])
                entities = extraction_results.get('entities', [])

                return idx, paragraph, entities

            # Process paragraphs in parallel
            per_paragraph_outputs: list[dict] = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for idx, paragraph, entities in executor.map(_process_single, enumerate(paragraphs_to_process)):
                    paragraph_result = {
                        'paragraph_index': paragraph.get('paragraph_index', idx + 1),
                        'chunk_number': paragraph['chunk_number'],
                        'content': paragraph['content'],
                        'entities': entities,
                        'entity_count': len(entities)
                    }
                    per_paragraph_outputs.append(paragraph_result)

                    # Log progress
                    if entities:
                        logger.info(f"  Found {len(entities)} entities in paragraph {idx+1}")
                    else:
                        logger.info(f"  No entities found in paragraph {idx+1}")

                    # Save progress every self.progress_interval paragraphs
                    if (idx + 1) % self.progress_interval == 0:
                        self._save_progress({'paragraphs': per_paragraph_outputs}, idx + 1, len(paragraphs_to_process))

            # Consolidate into chunk-level structure
            chunk_map: dict[int, dict] = {}
            for p in per_paragraph_outputs:
                cnum = p['chunk_number']
                if cnum not in chunk_map:
                    chunk_map[cnum] = {
                        'paragraphs': [],
                        'relationships_within_chunk': [],
                        'medical_summary_for_chunk': {},
                        'medical_summary_embeddings_for_chunk': {}
                    }
                
                # Use per-chunk paragraph index
                per_chunk_index = len(chunk_map[cnum]['paragraphs']) + 1
                p_struct = {
                    'paragraph_index': per_chunk_index,
                    'content': p['content'],
                    'entities': p['entities'],
                    'entity_count': p['entity_count']
                }
                chunk_map[cnum]['paragraphs'].append(p_struct)

            # Generate chunk-level relationships and summaries
            logger.info("Generating chunk-level relationships...")
            for cnum, cdata in chunk_map.items():
                try:
                    if cdata['paragraphs']:
                        # Concatenate content and aggregate entities for the chunk
                        chunk_text = "\n\n".join(p['content'] for p in cdata['paragraphs'] if p.get('content'))
                        aggregate_entities: list = []
                        for p in cdata['paragraphs']:
                            aggregate_entities.extend(p.get('entities', []))

                        # Generate chunk-level relationships for ALL entities in the chunk
                        if len(aggregate_entities) >= 2:
                            logger.info(f"Generating relationships for chunk {cnum} with {len(aggregate_entities)} entities")
                            chunk_relationships = self._generate_chunk_relationships(chunk_text, aggregate_entities)
                            cdata['relationships_within_chunk'] = chunk_relationships
                            logger.info(f"Generated {len(chunk_relationships)} relationships for chunk {cnum}")
                        else:
                            logger.debug(f"Chunk {cnum} has fewer than 2 entities, no relationships generated")
                            cdata['relationships_within_chunk'] = []

                        # Generate chunk-level medical summary and embeddings
                        if self.summarizer is not None:
                            logger.debug(f"Generating medical summary for chunk {cnum}")
                            chunk_summary = self.summarizer.summarize_chunk(
                                chunk_text, aggregate_entities, cdata['relationships_within_chunk']
                            ) or {}
                            cdata['medical_summary_for_chunk'] = chunk_summary

                            # Generate embeddings for the medical summary categories
                            if self.embedder is not None and chunk_summary:
                                try:
                                    medical_summary_embeddings = {}
                                    for category, summary_text in chunk_summary.items():
                                        if summary_text and str(summary_text).strip():
                                            # Generate embedding for each medical category
                                            category_embedding = self.embedder.encode([str(summary_text)])
                                            if len(category_embedding) > 0:
                                                medical_summary_embeddings[category] = category_embedding[0].tolist()
                                    
                                    cdata['medical_summary_embeddings_for_chunk'] = medical_summary_embeddings
                                    logger.debug(f"Generated embeddings for {len(medical_summary_embeddings)} medical categories in chunk {cnum}")
                                except Exception as e:
                                    logger.warning(f"Failed to generate medical summary embeddings for chunk {cnum}: {e}")
                                    cdata['medical_summary_embeddings_for_chunk'] = {}
                            else:
                                cdata['medical_summary_embeddings_for_chunk'] = {}
                        else:
                            cdata['medical_summary_for_chunk'] = {}
                            cdata['medical_summary_embeddings_for_chunk'] = {}
                    else:
                        cdata['relationships_within_chunk'] = []
                        cdata['medical_summary_for_chunk'] = {}
                        cdata['medical_summary_embeddings_for_chunk'] = {}
                        
                except Exception as e:
                    logger.error(f"Failed to process chunk {cnum}: {e}")
                    cdata['relationships_within_chunk'] = []
                    cdata['medical_summary_for_chunk'] = {}
                    cdata['medical_summary_embeddings_for_chunk'] = {}

            results['chunks'] = chunk_map
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process chunked JSON: {e}")
            raise

def entity_processor(json_path: str, txt_path: str, max_paragraphs: int = 20, 
                    spacy_model: str = "en_core_sci_md", ollama_model: str = None,
                    ollama_endpoint: str = "http://localhost:11434") -> Dict[str, Any]:
    """
    Main function to process entities from chunked JSON using SciSpacy for detection and DSPy/Ollama for descriptions.
    Generates relationships only at the chunk level.
    
    Args:
        json_path: Path to the chunked JSON file
        txt_path: Path to the semantic types text file (pipe-delimited)
        max_paragraphs: Maximum number of paragraphs to process (default: 20)
        spacy_model: SciSpacy model to use (default: en_core_sci_md)
        ollama_model: Ollama model to use for descriptions (default: llama3.1:8b)
        ollama_endpoint: Ollama endpoint URL
        
    Returns:
        Dictionary containing the processing results with chunk-level relationships only
    """
    try:
        # Initialize processor with heuristic fallback enabled
        processor = EntityProcessor(
            spacy_model=spacy_model, 
            ollama_model=ollama_model, 
            ollama_endpoint=ollama_endpoint,
            relationship_fallback=True  # Enable heuristic fallback for relationship generation
        )
        
        # Load semantic types
        processor.load_semantic_types(txt_path)
        
        # Process the JSON file
        results = processor.process_chunked_json(json_path, max_paragraphs)
        
        # Add completion metadata
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        total_entities_found = 0
        total_relationships_found = 0
        total_paragraphs_with_entities = 0
        chunks_with_relationships = 0
        
        for chunk in results.get('chunks', {}).values():
            chunk_relationships = len(chunk.get('relationships_within_chunk', []))
            if chunk_relationships > 0:
                chunks_with_relationships += 1
            total_relationships_found += chunk_relationships
            
            for p in chunk.get('paragraphs', []):
                ent_count = len(p.get('entities', []))
                total_entities_found += ent_count
                if ent_count > 0:
                    total_paragraphs_with_entities += 1

        results['completion_metadata'] = {
            'completed_at': timestamp,
            'status': 'completed',
            'total_entities_found': total_entities_found,
            'total_chunk_relationships_found': total_relationships_found,
            'total_chunks': len(results.get('chunks', {})),
            'chunks_with_relationships': chunks_with_relationships,
            'total_paragraphs_processed': results.get('metadata', {}).get('total_paragraphs_processed', 0),
            'paragraphs_with_entities': total_paragraphs_with_entities,
        }
        
        logger.info("✓ Entity processing completed successfully")
        logger.info(f"✓ Generated relationships for {chunks_with_relationships}/{len(results.get('chunks', {}))} chunks")
        return results
        
    except Exception as e:
        logger.error(f"Entity processing failed: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    # Defaults per request
    default_json = "UNMC PDF Transplant_chunked.json"
    default_txt = "joined.txt"
    default_max_paragraphs = 10
    default_spacy = "en_core_sci_md"
    default_llm = "llama3.1:8b"

    try:
        print("Press Enter to accept defaults shown in [brackets].")
        print()
        print("Model Selection:")
        print("- For Ollama models: llama3.1:8b, llama3.2:3b, etc.")  
        print("- For HuggingFace models: meta-llama/Llama-3.2-1B-Instruct, microsoft/DialoGPT-small, etc.")
        print("- Note: Large HuggingFace models may fail to download due to connection issues")
        print("- Recommendation: Use Ollama for reliable operation, or try smaller HF models first")
        print()
        
        json_path = input(f"JSON path [{default_json}]: ").strip() or default_json
        txt_path = input(f"Semantic types path [{default_txt}]: ").strip() or default_txt
        max_para_str = input(f"Max paragraphs [{default_max_paragraphs}]: ").strip()
        max_paragraphs = int(max_para_str) if max_para_str else default_max_paragraphs
        spacy_model = input(f"spaCy model [{default_spacy}]: ").strip() or default_spacy
        llm_model = input(f"LLM model [{default_llm}]: ").strip() or default_llm
        
        # Detect if it's a HuggingFace model
        if '/' in llm_model:
            print(f"Detected HuggingFace model: {llm_model}")
            ollama_model = llm_model
        else:
            print(f"Using Ollama model: {llm_model}")
            ollama_model = llm_model

        results = entity_processor(json_path, txt_path, max_paragraphs, spacy_model, ollama_model)

        # Save results to file
        output_file = "entity_processing_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✓ Results saved to {output_file}")
        total_paragraphs = results['metadata']['total_paragraphs_processed']
        total_chunks = len(results.get('chunks', {}))
        paragraphs_with_entities = results.get('completion_metadata', {}).get('paragraphs_with_entities', 0)
        chunks_with_relationships = results.get('completion_metadata', {}).get('chunks_with_relationships', 0)
        total_relationships = results.get('completion_metadata', {}).get('total_chunk_relationships_found', 0)

        print(f"Processed {total_paragraphs} paragraphs across {total_chunks} chunks")
        print(f"Found entities in {paragraphs_with_entities} paragraphs")
        print(f"Generated relationships in {chunks_with_relationships} chunks ({total_relationships} total relationships)")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)