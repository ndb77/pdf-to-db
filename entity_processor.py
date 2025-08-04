import json
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
from embedding_pipeline import BioMedBERTEmbedder


# Optimize spacy for speed
spacy.prefer_gpu()  # Use GPU if available

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchEntityProcessingSignature(dspy.Signature):
    """DSPy signature for batch processing entities and their relationships."""
    
    paragraph_content = dspy.InputField(desc="The full paragraph content")
    entity_list = dspy.InputField(desc="JSON list of entities detected by SciSpacy with their names and labels")
    
    processed_entities = dspy.OutputField(desc="""You are a JSON-only assistant. Your goal is to generate a JSON object with entities and relationships from the given text.


ALWAYS respond with nothing but valid JSON that conforms to this schema:
{
    "processed_entities": {
        "entities": [
            {
                "entity_name": "entity name",
                "entity_description": "comprehensive description of what this entity is and its role"
            }
        ],
        "relationships": [
            {
                "source_entity": "name of source entity",
                "target_entity": "name of target entity",
                "relationship_description": "explanation of how these entities are related",
                "relationship_strength": integer between 1 and 10
            }
        ]
    }
}
Example content:
                                          
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly.

Example output:
{
    "processed_entities": {
        "entities": [
            {
                "entity_name": "Operatives",
                "entity_description": "Operatives initially tasked with observing and reporting, later evolved into active participants and guardians of a cosmic threshold."
            },
            {
                "entity_name": "Guardians of a threshold",
                "entity_description": "A symbolic role assumed by the team, responsible for managing a message from a realm beyond ordinary human understanding."
            },
            {
                "entity_name": "Washington",
                "entity_description": "Refers to the U.S. political and military command center with which the team maintains communication."
            },
            {
                "entity_name": "Cosmos",
                "entity_description": "The greater universe, representing the unknown or extraterrestrial domain with which the team is interacting."
            },
            {
                "entity_name": "Mercer",
                "entity_description": "A person whose instincts influenced the team's decision to shift from observation to interaction and preparation."
            },
            {
                "entity_name": "Operation: Dulce",
                "entity_description": "A codename for the evolving mission undertaken by the team, marked by increased engagement with a cosmic warning."
            },
            {
                "entity_name": "Crystallizing warning",
                "entity_description": "An emergent, urgent signal or message from beyond that prompts the team to change their role."
            }
        ],
        "relationships": [
            {
                "source_entity": "Operatives",
                "target_entity": "Guardians of a threshold",
                "relationship_description": "The operatives transformed into guardians, taking on a more active and protective role.",
                "relationship_strength": 9
            },
            {
                "source_entity": "Washington",
                "target_entity": "Operatives",
                "relationship_description": "The operatives maintained communication with Washington, suggesting oversight or coordination.",
                "relationship_strength": 7
            },
            {
                "source_entity": "Mercer",
                "target_entity": "Team",
                "relationship_description": "Mercer’s instincts guided the team’s shift from passive observation to proactive engagement.",
                "relationship_strength": 8
            },
            {
                "source_entity": "Operation: Dulce",
                "target_entity": "Crystallizing warning",
                "relationship_description": "Operation: Dulce was reoriented to respond to the emerging cosmic warning.",
                "relationship_strength": 9
            },
            {
                "source_entity": "Team",
                "target_entity": "Cosmos",
                "relationship_description": "The team's actions could influence humanity's place in the cosmos, implying a high-stakes relationship.",
                "relationship_strength": 10
            }
        ]
    }
}

Remember: output ONLY JSON. For all chunks with more than 1 entity, all entities should have at least one relationship to another entity as either a source or target entity.""")

class EntityProcessor:
    """Process entities from chunked JSON using SciSpacy for detection and DSPy/Ollama for descriptions."""
    
    # Class-level cache for SciSpacy models
    _nlp_cache = {}
    _linker_cache = {}
    
    def __init__(self, spacy_model: str = "en_core_sci_md", ollama_model: str = "llama3.1:8b", 
                 ollama_endpoint: str = "http://localhost:11434",
                 chunk_size: int = 25,
                 max_workers: Optional[int] = None,
                 progress_interval: int = 20,
                 cache_path: str = "./cache/entity_desc_cache.json"):
        """
        Initialize the entity processor.
        
        Args:
            spacy_model: SciSpacy model to use (default: en_core_sci_md)
            ollama_model: Ollama model to use for descriptions (default: llama3.1:8b)
            ollama_endpoint: Ollama endpoint URL
        """
        self.spacy_model = spacy_model
        self.ollama_model = ollama_model
        self.ollama_endpoint = ollama_endpoint
        # Performance / config parameters
        self.chunk_size = chunk_size
        self.max_workers = max_workers if max_workers is not None else os.cpu_count() or 4
        self.progress_interval = progress_interval
        self.cache_path = cache_path

        self.nlp = None
        self.linker = None
        self.lm = None
        self.batch_predictor = None
        self.semantic_types = []

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
        
        # Initialize DSPy with Ollama for descriptions
        try:
            ollama_model_string = f"ollama/{self.ollama_model}"
            self.lm = dspy.LM(
                model=ollama_model_string, 
                api_base=self.ollama_endpoint,
                temperature=0.0,
                max_tokens=5500
            )
            dspy.settings.configure(lm=self.lm)
            self.batch_predictor = dspy.Predict(BatchEntityProcessingSignature)
            logger.info(f"✓ DSPy initialized with Ollama model: {self.ollama_model}")
        except Exception as e:
            logger.error(f"Failed to initialize DSPy with Ollama: {e}")
            raise RuntimeError(f"Could not initialize DSPy with Ollama model {ollama_model}")
        
        # Initialize BioMedBERT embedder for entity embeddings
        try:
            self.embedder = BioMedBERTEmbedder(batch_size=32)  # Smaller batch size for memory efficiency
            logger.info(f"✓ BioMedBERT embedder initialized")
        except Exception as e:
            logger.error(f"Failed to initialize BioMedBERT embedder: {e}")
            logger.warning("Continuing without embedding functionality")
            self.embedder = None
    
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
            
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
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
            
            self.semantic_types = semantic_types
            self.semantic_type_definitions = semantic_type_definitions
            self.semantic_type_categories = semantic_type_categories
            self.semantic_type_cuis = semantic_type_cuis
            self.semantic_type_tuis = semantic_type_tuis
            
            logger.info(f"✓ Loaded {len(self.semantic_types)} semantic types")
            
        except Exception as e:
            logger.error(f"Failed to load semantic types: {e}")
            raise
    
    def _chunk_list(self, data_list: List[Any], chunk_size: int) -> "Generator[List[Any], None, None]":
        """Utility to split a list into chunks of size chunk_size."""
        for i in range(0, len(data_list), chunk_size):
            yield data_list[i:i + chunk_size]

    def batch_process_entities(self, paragraph_content: str, detected_entities: List[Dict], chunk_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all entities in a paragraph using a single LLM call for descriptions and relationships.
        
        Args:
            paragraph_content: The full paragraph content
            detected_entities: List of entities detected by SciSpacy with their metadata
            
        Returns:
            Dictionary containing processed entities and relationships
        """
        try:
            if chunk_size is None:
                chunk_size = self.chunk_size

            # We'll aggregate results across chunks
            aggregated_entities: List[Dict[str, Any]] = []
            aggregated_relationships: List[Dict[str, Any]] = []

            # Pre-fill entity descriptions from cache
            entity_descriptions: Dict[str, str] = {
                ent_name: desc for ent_name, desc in self.desc_cache.items()
                if any(e["entity_name"] == ent_name for e in detected_entities)
            }

            # Always process ALL entities for relationships, but use cache for descriptions
            for chunk_index, entity_chunk in enumerate(self._chunk_list(detected_entities, chunk_size)):
                # Prepare entity list for LLM - include ALL entities for relationship generation
                entity_input_list = [
                    {
                        "entity_name": ent["entity_name"],
                        "spacy_label": ent["spacy_label"]
                    } for ent in entity_chunk
                ]

                entity_list_json = json.dumps(entity_input_list)
                logger.info(
                    f"[Batch {chunk_index+1}] Sending {len(entity_chunk)} entities to LLM for relationship generation (paragraph preview: {paragraph_content[:80]}...)"
                )

                result = self.batch_predictor(
                    paragraph_content=paragraph_content,
                    entity_list=entity_list_json
                )

                logger.debug(f"[Batch {chunk_index+1}] LLM raw response: {result.processed_entities}")

                processed_data = self._parse_llm_response(result.processed_entities)

                # Always collect relationships from LLM response
                aggregated_relationships.extend(processed_data.get("relationships", []))

                # Process entities: use LLM descriptions for uncached entities, cache for cached ones
                for entity_info in processed_data.get("entities", []):
                    entity_name = entity_info["entity_name"]
                    llm_description = entity_info.get("entity_description", "")
                    
                    # Use cached description if available, otherwise use LLM description
                    entity_type = entity_info.get("entity_type", "")
                    if entity_name in entity_descriptions:
                        final_description = entity_descriptions[entity_name]
                    else:
                        final_description = llm_description
                        # Update cache with new description
                        entity_descriptions[entity_name] = final_description
                        with self.cache_lock:
                            self.desc_cache[entity_name] = final_description
                    # Add to aggregated entities
                    aggregated_entities.append({
                        "entity_name": entity_name,
                        "entity_description": final_description
                    })
                    
                    # Compute embeddings for the entity if embedder is available
                    if self.embedder is not None:
                        try:
                            # Find the original entity to get entity_type
                            original_entity = next((e for e in detected_entities if e["entity_name"] == entity_name), None)
                            if original_entity:
                                entity_type = original_entity.get("entity_type", "Entity")
                                entity_embedding = self.embedder.encode_entity(entity_name, entity_type, final_description)
                                # Add the embedding to the aggregated entities
                                aggregated_entities[-1]["embedding"] = entity_embedding.tolist()
                        except Exception as e:
                            logger.warning(f"Failed to compute embedding for entity '{entity_name}': {e}")
                    else:
                        logger.debug("Embedder not available, skipping embedding computation")

            # Ensure all entities have descriptions (fallback for any missing)
            for entity in detected_entities:
                entity_name = entity["entity_name"]
                if entity_name not in entity_descriptions or not entity_descriptions[entity_name]:
                    # Fallback single-entity call to ensure description
                    single_entity_payload = json.dumps([
                        {"entity_name": entity_name, "spacy_label": entity["spacy_label"]}
                    ])
                    single_result = self.batch_predictor(
                        paragraph_content=paragraph_content,
                        entity_list=single_entity_payload
                    )
                    single_data = self._parse_llm_response(single_result.processed_entities)
                    if single_data.get("entities"):
                        description = single_data["entities"][0].get("entity_description", "")
                    else:
                        description = ""
                    entity_descriptions[entity_name] = description
                    # Update cache
                    with self.cache_lock:
                        self.desc_cache[entity_name] = description
                    
                    # Add to aggregated entities if not already present
                    if not any(e["entity_name"] == entity_name for e in aggregated_entities):
                        aggregated_entities.append({
                            "entity_name": entity_name,
                            "entity_description": description
                        })
                        
                        # Compute embeddings for fallback entities if embedder is available
                        if self.embedder is not None:
                            try:
                                entity_type = entity.get("entity_type", "Entity")
                                entity_embedding = self.embedder.encode_entity(entity_name, entity_type, description)
                                # Add the embedding to the aggregated entities
                                aggregated_entities[-1]["embedding"] = entity_embedding.tolist()
                            except Exception as e:
                                logger.warning(f"Failed to compute embedding for fallback entity '{entity_name}': {e}")

            # Persist cache after updates
            self._save_description_cache()

            # Return processed entities with descriptions, embeddings, and relationships
            return {
                "entity_descriptions": entity_descriptions,
                "entities_with_embeddings": aggregated_entities,
                "relationships": aggregated_relationships
            }
            
        except Exception as e:
            logger.warning(f"Failed to batch process entities: {e}")
            logger.warning(f"Exception type: {type(e).__name__}")
            logger.warning(f"Exception details: {str(e)}")
            # Fallback to empty results
            return {
                "entity_descriptions": {},
                "relationships": []
            }

    def _parse_llm_response(self, raw) -> dict:
        """
        Attempts to coerce the LLM output into a Python dict.
        1.  Accepts already-parsed dicts
        2.  Strips ``` fences and leading labels
        3.  First tries json.loads
        4.  Then fixes single→double quotes and retries json.loads
        5.  Finally falls back to ast.literal_eval
        6.  Always returns {'entities': [], 'relationships': []} shape
        """
        # 0. If DSPy already returned a dict we're done
        if isinstance(raw, dict):
            data = raw
        else:
            txt = str(raw).strip()

            # Drop code fences or leading markdown
            txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.IGNORECASE).strip()

            # 1st attempt – vanilla JSON
            try:
                data = json.loads(txt)
            except json.JSONDecodeError:
                # 2nd attempt – replace single quotes
                try:
                    data = json.loads(txt.replace("'", '"'))
                except json.JSONDecodeError:
                    # 3rd attempt – Python literal
                    try:
                        data = ast.literal_eval(txt)
                    except Exception:
                        logger.warning("Unable to parse LLM response, returning empty shell")
                        data = {}

        # Unwrap if the model kept the outer key
        if "processed_entities" in data:
            data = data["processed_entities"]

        # Guarantee keys exist
        return {
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", [])
        }

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
            import datetime
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
    
    def generate_entity_description(self, entity_name: str, context_content: str, spacy_label: str) -> str:
        """
        Generate a comprehensive description of the entity using DSPy/Ollama.
        
        Args:
            entity_name: Name of the entity
            context_content: The paragraph content where the entity was found
            spacy_label: The SciSpacy entity label
            
        Returns:
            Generated description of the entity
        """
        try:
            # Use DSPy to generate description
            result = self.description_predictor(
                entity_name=entity_name,
                context_content=context_content,
                spacy_label=spacy_label
            )
            
            description = result.entity_description.strip()
            
            # Handle empty or invalid responses
            if not description or description == "{}" or description == "[]":
                logger.warning(f"Empty description generated for entity '{entity_name}'")
                return f"A {spacy_label.lower().replace('_', ' ')} entity"
            
            return description
            
        except Exception as e:
            logger.warning(f"Failed to generate description for '{entity_name}': {e}")
            return f"A {spacy_label.lower().replace('_', ' ')} entity"
    
    def extract_entities_from_paragraph(self, paragraph_content: str) -> Dict[str, Any]:
        """
        Extract clinically relevant entities from a paragraph using SciSpacy and batch LLM processing.
        
        Args:
            paragraph_content: The paragraph content to analyze
            
        Returns:
            Dictionary containing entities list and relationships list
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
                    # Create reverse lookup dictionaries for CUI-based search
                    cui_to_name = {cui: name for name, cui in self.semantic_type_cuis.items()}
                    if entity_cui in cui_to_name:
                        entity_name_from_cui = cui_to_name[entity_cui]
                        entity_type = self.semantic_type_categories.get(entity_name_from_cui, "Entity")
                        umls_definition = self.semantic_type_definitions.get(entity_name_from_cui, "")
                
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
                return {
                    'entities': [],
                    'relationships': []
                }
            
            # Second pass: Use batch LLM processing for descriptions and relationships
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
                
                # Get embedding if available
                embedding = None
                if entity_name in embedding_lookup:
                    embedding = embedding_lookup[entity_name].get("embedding")
                
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
                    'is_clinically_relevant': entity['is_clinically_relevant']
                }
                
                # Add embedding if available
                if embedding is not None:
                    final_entity['embedding'] = embedding
                
                final_entities.append(final_entity)
            
            return {
                'entities': final_entities,
                'relationships': batch_results.get("relationships", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to extract entities from paragraph: {e}")
            return {
                'entities': [],
                'relationships': []
            }
    
    def process_chunked_json(self, json_path: str, max_paragraphs: int = 20) -> Dict[str, Any]:
        """
        Process the chunked JSON file and extract entities from the first N paragraphs.
        Saves progress every 10 paragraphs to prevent data loss.
        
        Args:
            json_path: Path to the chunked JSON file
            max_paragraphs: Maximum number of paragraphs to process
            
        Returns:
            Dictionary containing the processing results
        """
        try:
            # Load the JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✓ Loaded JSON file with {len(data.get('chunks', []))} chunks")
            
            # Collect all paragraphs from chunks
            all_paragraphs = []
            for chunk in data.get('chunks', []):
                for paragraph in chunk.get('paragraphs', []):
                    if paragraph.get('type') == 'text' and paragraph.get('content', '').strip():
                        all_paragraphs.append({
                            'chunk_number': chunk.get('chunk_number'),
                            'content': paragraph.get('content'),
                            'type': paragraph.get('type')
                        })
            
            # Limit to first N paragraphs
            paragraphs_to_process = all_paragraphs[:max_paragraphs]
            logger.info(f"Processing first {len(paragraphs_to_process)} paragraphs")
            
            # Process each paragraph
            results = {
                'metadata': {
                    'total_paragraphs_processed': len(paragraphs_to_process),
                    'spacy_model': self.spacy_model,
                    'ollama_model': self.ollama_model,
                    'semantic_types_loaded': len(self.semantic_types)
                },
                'paragraphs': []
            }
            
            def _process_single(idx_paragraph):
                idx, paragraph = idx_paragraph
                logger.info(f"[Thread] Processing paragraph {idx+1}/{len(paragraphs_to_process)}")
                extraction_results = self.extract_entities_from_paragraph(paragraph['content'])
                entities = extraction_results.get('entities', [])
                relationships = extraction_results.get('relationships', [])
                return idx, paragraph, entities, relationships

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for idx, paragraph, entities, relationships in executor.map(_process_single, enumerate(paragraphs_to_process)):
                    paragraph_result = {
                        'paragraph_index': idx + 1,
                        'chunk_number': paragraph['chunk_number'],
                        'content': paragraph['content'],
                        'entities': entities,
                        'entity_count': len(entities),
                        'relationships': relationships,
                        'relationship_count': len(relationships)
                    }
                    results['paragraphs'].append(paragraph_result)

                    # Log progress (main thread)
                    if entities:
                        logger.info(f"  Found {len(entities)} entities and {len(relationships)} relationships in paragraph {idx+1}")
                    else:
                        logger.info(f"  No entities found in paragraph {idx+1}")

                    # Save progress every self.progress_interval paragraphs
                    if (idx + 1) % self.progress_interval == 0:
                        self._save_progress(results, idx + 1, len(paragraphs_to_process))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process chunked JSON: {e}")
            raise

def entity_processor(json_path: str, txt_path: str, max_paragraphs: int = 20, 
                    spacy_model: str = "en_core_sci_md", ollama_model: str = "llama3.1:8b",
                    ollama_endpoint: str = "http://localhost:11434") -> Dict[str, Any]:
    """
    Main function to process entities from chunked JSON using SciSpacy for detection and DSPy/Ollama for descriptions.
    
    Args:
        json_path: Path to the chunked JSON file
        txt_path: Path to the semantic types text file (pipe-delimited)
        max_paragraphs: Maximum number of paragraphs to process (default: 20)
        spacy_model: SciSpacy model to use (default: en_core_sci_md)
        ollama_model: Ollama model to use for descriptions (default: llama3.1:8b)
        ollama_endpoint: Ollama endpoint URL
        
    Returns:
        Dictionary containing the processing results
    """
    try:
        # Initialize processor
        processor = EntityProcessor(
            spacy_model=spacy_model, 
            ollama_model=ollama_model, 
            ollama_endpoint=ollama_endpoint
        )
        
        # Load semantic types
        processor.load_semantic_types(txt_path)
        
        # Process the JSON file
        results = processor.process_chunked_json(json_path, max_paragraphs)
        
        # Add completion metadata
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results['completion_metadata'] = {
            'completed_at': timestamp,
            'status': 'completed',
            'total_entities_found': sum(len(p.get('entities', [])) for p in results.get('paragraphs', [])),
            'total_relationships_found': sum(len(p.get('relationships', [])) for p in results.get('paragraphs', []))
        }
        
        logger.info("✓ Entity processing completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Entity processing failed: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python entity_processor.py <json_path> <txt_path> [max_paragraphs] [spacy_model] [ollama_model]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    txt_path = sys.argv[2]
    max_paragraphs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    spacy_model = sys.argv[4] if len(sys.argv) > 4 else "en_core_sci_md"
    ollama_model = sys.argv[5] if len(sys.argv) > 5 else "llama3.1:8b"
    
    try:
        results = entity_processor(json_path, txt_path, max_paragraphs, spacy_model, ollama_model)
        
        # Save results to file
        output_file = "entity_processing_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to {output_file}")
        print(f"Processed {results['metadata']['total_paragraphs_processed']} paragraphs")
        print(f"Found entities in {sum(1 for p in results['paragraphs'] if p['entities'])} paragraphs")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 