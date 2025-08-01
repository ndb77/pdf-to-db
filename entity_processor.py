import json
import pandas as pd
import spacy
from scispacy.linking import EntityLinker
import dspy
from typing import List, Dict, Any, Optional
import logging
import numpy as np

# Optimize spacy for speed
spacy.prefer_gpu()  # Use GPU if available

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchEntityProcessingSignature(dspy.Signature):
    """DSPy signature for batch processing entities and their relationships."""
    
    paragraph_content = dspy.InputField(desc="The full paragraph content")
    entity_list = dspy.InputField(desc="JSON list of entities detected by SciSpacy with their names and labels")
    
    processed_entities = dspy.OutputField(desc="""Generate a JSON object with entities and relationships from the given text.

For each entity in the entity_list, create a description. Then identify relationships between entities.

Return the JSON wrapped in a "processed_entities" field like this:
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

Example output:
{
    "processed_entities": {
        "entities": [
            {
                "entity_name": "Operatives",
                "entity_description": "Operatives initially tasked with observing and reporting, later evolved into active participants and guardians of a cosmic threshold."
            }
        ],
        "relationships": [
            {
                "source_entity": "Operatives",
                "target_entity": "Guardians of a threshold",
                "relationship_description": "The operatives transformed into guardians, taking on a more active and protective role.",
                "relationship_strength": 9
            }
        ]
    }
}

Return only valid JSON with the processed_entities wrapper.""")

class EntityProcessor:
    """Process entities from chunked JSON using SciSpacy for detection and DSPy/Ollama for descriptions."""
    
    # Class-level cache for SciSpacy models
    _nlp_cache = {}
    _linker_cache = {}
    
    def __init__(self, spacy_model: str = "en_core_sci_md", ollama_model: str = "llama3.2", 
                 ollama_endpoint: str = "http://localhost:11434"):
        """
        Initialize the entity processor.
        
        Args:
            spacy_model: SciSpacy model to use (default: en_core_sci_md)
            ollama_model: Ollama model to use for descriptions (default: llama3.2)
            ollama_endpoint: Ollama endpoint URL
        """
        self.spacy_model = spacy_model
        self.ollama_model = ollama_model
        self.ollama_endpoint = ollama_endpoint
        self.nlp = None
        self.linker = None
        self.lm = None
        self.batch_predictor = None
        self.semantic_types = []
        
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
                max_tokens=500
            )
            dspy.settings.configure(lm=self.lm)
            self.batch_predictor = dspy.Predict(BatchEntityProcessingSignature)
            logger.info(f"✓ DSPy initialized with Ollama model: {self.ollama_model}")
        except Exception as e:
            logger.error(f"Failed to initialize DSPy with Ollama: {e}")
            raise RuntimeError(f"Could not initialize DSPy with Ollama model {ollama_model}")
    
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
    
    def batch_process_entities(self, paragraph_content: str, detected_entities: List[Dict]) -> Dict[str, Any]:
        """
        Process all entities in a paragraph using a single LLM call for descriptions and relationships.
        
        Args:
            paragraph_content: The full paragraph content
            detected_entities: List of entities detected by SciSpacy with their metadata
            
        Returns:
            Dictionary containing processed entities and relationships
        """
        try:
            # Prepare entity list for LLM
            entity_input_list = []
            for ent_data in detected_entities:
                entity_input_list.append({
                    "entity_name": ent_data["entity_name"],
                    "spacy_label": ent_data["spacy_label"]
                })
            
            # Convert to JSON string for LLM input
            import json
            entity_list_json = json.dumps(entity_input_list)
            
            # Make single LLM call for all entities and relationships
            logger.info(f"Sending to LLM - paragraph_content: {paragraph_content[:100]}...")
            logger.info(f"Sending to LLM - entity_list_json: {entity_list_json}")
            
            # Debug: Let's see what DSPy is actually sending to the LLM
            try:
                # Get the signature to see the prompt structure
                signature = self.batch_predictor.signature
                logger.info(f"DSPy signature: {signature}")
                logger.info(f"Input fields: {signature.input_fields}")
                logger.info(f"Output fields: {signature.output_fields}")
            except Exception as e:
                logger.warning(f"Could not inspect signature: {e}")
            
            result = self.batch_predictor(
                paragraph_content=paragraph_content,
                entity_list=entity_list_json
            )
            
            logger.info(f"LLM raw response: {result.processed_entities}")
            logger.info(f"Response type: {type(result.processed_entities)}")
            logger.info(f"Response attributes: {dir(result)}")
            
            # Parse the JSON response
            try:
                # Try to parse the processed_entities field first
                response_json = json.loads(result.processed_entities)
                logger.info("Successfully parsed processed_entities field")
                
                # Extract the processed_entities from the wrapped response
                if "processed_entities" in response_json:
                    processed_data = response_json["processed_entities"]
                    logger.info("Successfully extracted processed_entities from wrapper")
                else:
                    # If no wrapper, use the response directly
                    processed_data = response_json
                    logger.info("No wrapper found, using response directly")
                    
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"Failed to parse processed_entities field: {e}")
                logger.warning(f"Raw response: {result.processed_entities}")
                
                # Try to parse the raw response as JSON
                try:
                    # If the response is already a dict object, use it directly
                    if isinstance(result.processed_entities, dict):
                        processed_data = result.processed_entities
                        logger.info("Used response as dict directly")
                    else:
                        # Try to parse as raw JSON string
                        processed_data = json.loads(str(result.processed_entities))
                        logger.info("Successfully parsed raw JSON response")
                except (json.JSONDecodeError, TypeError) as e2:
                    logger.warning(f"Failed to parse raw JSON response: {e2}")
                    
                    # Try to convert Python dict string to JSON
                    try:
                        import ast
                        # Convert Python dict string to actual dict
                        python_dict = ast.literal_eval(str(result.processed_entities))
                        if isinstance(python_dict, dict):
                            processed_data = python_dict
                            logger.info("Successfully parsed Python dict string")
                        else:
                            processed_data = {"entities": [], "relationships": []}
                    except (ValueError, SyntaxError) as e3:
                        logger.warning(f"Failed to parse Python dict: {e3}")
                        # Fallback to empty structure
                        processed_data = {"entities": [], "relationships": []}
            
            # Create mapping from entity names to descriptions
            entity_descriptions = {}
            for entity_info in processed_data.get("entities", []):
                entity_descriptions[entity_info["entity_name"]] = entity_info.get("entity_description", "")
            
            # Return processed entities with descriptions and relationships
            return {
                "entity_descriptions": entity_descriptions,
                "relationships": processed_data.get("relationships", [])
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
            
            # Combine SciSpacy data with LLM results
            final_entities = []
            entity_descriptions = batch_results.get("entity_descriptions", {})
            
            for entity in unique_detected_entities:
                entity_name = entity['entity_name']
                
                # Get description from batch processing or fallback
                entity_description = entity_descriptions.get(entity_name, f"A {entity['spacy_label'].lower().replace('_', ' ')} entity")
                
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
            
            for i, paragraph in enumerate(paragraphs_to_process):
                logger.info(f"Processing paragraph {i+1}/{len(paragraphs_to_process)}")
                
                # Extract entities and relationships
                extraction_results = self.extract_entities_from_paragraph(paragraph['content'])
                entities = extraction_results.get('entities', [])
                relationships = extraction_results.get('relationships', [])
                
                paragraph_result = {
                    'paragraph_index': i + 1,
                    'chunk_number': paragraph['chunk_number'],
                    'content': paragraph['content'],
                    'entities': entities,
                    'entity_count': len(entities),
                    'relationships': relationships,
                    'relationship_count': len(relationships)
                }
                
                results['paragraphs'].append(paragraph_result)
                
                # Log progress
                if entities:
                    logger.info(f"  Found {len(entities)} entities and {len(relationships)} relationships")
                    for entity in entities[:3]:  # Show first 3 entities
                        logger.info(f"    - {entity['entity_name']} ({entity['entity_type']}) [{entity['spacy_label']}]")
                else:
                    logger.info("  No entities found")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process chunked JSON: {e}")
            raise

def entity_processor(json_path: str, txt_path: str, max_paragraphs: int = 20, 
                    spacy_model: str = "en_core_sci_md", ollama_model: str = "llama3.2",
                    ollama_endpoint: str = "http://localhost:11434") -> Dict[str, Any]:
    """
    Main function to process entities from chunked JSON using SciSpacy for detection and DSPy/Ollama for descriptions.
    
    Args:
        json_path: Path to the chunked JSON file
        txt_path: Path to the semantic types text file (pipe-delimited)
        max_paragraphs: Maximum number of paragraphs to process (default: 20)
        spacy_model: SciSpacy model to use (default: en_core_sci_md)
        ollama_model: Ollama model to use for descriptions (default: llama3.2)
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
    ollama_model = sys.argv[5] if len(sys.argv) > 5 else "llama3.2"
    
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