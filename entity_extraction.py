import spacy
from scispacy.linking import EntityLinker

import json

# https://allenai.github.io/scispacy/ 
# 1. Load the model
nlp = spacy.load("en_core_sci_sm")
document_path = "semantically_chunked_paragraphs_.test-docs12911-022-02052-9(1)_marker.json"
document_output_path = "semantically_chunked_paragraphs_.test-docs12911-022-02052-9(1)_marker_with_entities_and_context.json"

nlp.add_pipe("scispacy_linker", config={"linker_name": "umls", "resolve_abbreviations": True})




def add_entities_to_chunks(chunks_data, nlp=None):
    """
    Add entity information to each paragraph in the chunks data.
    
    Args:
        chunks_data: The chunks data dictionary
        nlp: Optional spaCy model (will create one if not provided)
    
    Returns:
        The processed chunks data with entities added
    """
    # Set up entity detection if not provided
    if nlp is None:
        nlp = spacy.load("en_core_sci_sm")
        if "scispacy_linker" not in nlp.pipe_names:
            nlp.add_pipe("scispacy_linker", config={"linker_name": "umls", "resolve_abbreviations": True})
    # Create a copy of the data
    processed_data = json.loads(json.dumps(chunks_data))
    
    # Process each chunk
    for chunk in processed_data.get("chunks", []):
        print(f"Processing chunk {chunk.get('chunk_number', 'unknown')}")
        
        # Process each paragraph in the chunk
        for paragraph in chunk.get("paragraphs", []):
            content = paragraph.get("content", "")
            
            # Only process text paragraphs (skip headers)
            if paragraph.get("type") == "text" and content.strip():
                # Extract entities from the paragraph content
                doc = nlp(content)
                entities = []
                
                for ent in doc.ents:
                    entity_info = {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "description": spacy.explain(ent.label_)
                    }
                    
                    # Try to get UMLS information
                    if hasattr(ent._, 'umls_ents') and ent._.umls_ents:
                        umls_ents = ent._.umls_ents
                        if umls_ents:
                            cui, score = umls_ents[0]
                            entity_info["cui"] = cui
                            entity_info["score"] = float(score)
                            try:
                                entry = nlp.get_pipe("scispacy_linker").kb.cui_to_entity[cui]
                                entity_info["canonical_name"] = entry.canonical_name
                            except KeyError:
                                entity_info["canonical_name"] = None
                    elif hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                        kb_ents = ent._.kb_ents
                        if kb_ents:
                            cui, score = kb_ents[0]
                            entity_info["cui"] = cui
                            entity_info["score"] = float(score)
                            entity_info["canonical_name"] = None
                    
                    entities.append(entity_info)
                
                # Add entities to the paragraph
                paragraph["entities"] = entities
                
                # Print some statistics
                if entities:
                    print(f"  Found {len(entities)} entities")
                    for entity in entities[:2]:  # Show first 2 entities
                        print(f"    - {entity['text']} ({entity['label']})")
            else:
                # For non-text paragraphs, add empty entities list
                paragraph["entities"] = []
    
    print("Entity extraction completed!")
    return processed_data


# Load your chunks data
with open(document_path, "r") as f:
    chunks_data = json.load(f)

# Add entities
processed_data = add_entities_to_chunks(chunks_data)

# Save the results
with open(document_output_path, "w") as f:
    json.dump(processed_data, f, indent=2)

print("Results saved to", document_output_path) 

import json
import requests
import dspy
from typing import Dict, List, Any, Optional
import logging
from urllib.parse import quote
import time
from tqdm import tqdm
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure DSPy with Ollama
try:
    # Create LLM configuration for Ollama with optimized settings
    llm = dspy.LM(
        model="ollama/llama3.2",
        api_base="http://localhost:11434",
        max_tokens=50,  # Reduced from 150 to 50 for faster generation
        top_p=0.7,      # Add top_p for faster sampling
    )
    
    # Set the LLM as the default model for DSPy
    dspy.settings.configure(lm=llm)
except Exception as e:
    logger.error(f"Error configuring DSPy with Ollama: {e}")
    raise

import json
import spacy
from scispacy.linking import EntityLinker
from typing import Dict, List, Any, Optional, Tuple
import logging
import dspy
from dspy.signatures import Signature
import time
from collections import defaultdict
import hashlib

class EntityContextSignature(Signature):
    """Signature for generating contextual information about entities."""
    
    entity_text = dspy.InputField(desc="The entity text to provide context for")
    entity_type = dspy.InputField(desc="The type/category of the entity")
    paragraph_content = dspy.InputField(desc="The full paragraph content where the entity appears")
    
    context = dspy.OutputField(desc="A concise explanation of what this entity refers to in the context of the paragraph, including its role and significance")

class EntityContextGenerator:
    """Efficient entity context generator using DSPy and Ollama."""
    
    def __init__(self, model_name: str = "llama3.2", batch_size: int = 5, cache_results: bool = True):
        """
        Initialize the context generator.
        
        Args:
            model_name: Ollama model to use (default: llama3.2)
            batch_size: Number of entities to process in each batch
            cache_results: Whether to cache results for efficiency
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_results = cache_results
        self.cache = {}
        
        # Initialize DSPy with Ollama using the correct format
        try:
            # Use the correct DSPy format for Ollama: "ollama_chat/model_name"
            ollama_model_string = f"ollama_chat/{model_name}"
            self.lm = dspy.LM(model=ollama_model_string)
            self.context_predictor = dspy.Predict(EntityContextSignature)
            self.context_predictor.lm = self.lm
        except Exception as e:
            logging.warning(f"Failed to initialize Ollama with model {model_name}: {e}")
            logging.info("Falling back to default DSPy configuration")
            # Fallback to a default model that should work
            try:
                self.lm = dspy.LM(model="gpt-3.5-turbo")  # Fallback to OpenAI if available
                self.context_predictor = dspy.Predict(EntityContextSignature)
                self.context_predictor.lm = self.lm
            except Exception as fallback_error:
                logging.error(f"Failed to initialize any LM: {fallback_error}")
                raise RuntimeError("Could not initialize any language model for context generation")
    
    def _generate_cache_key(self, entity_text: str, entity_type: str, paragraph_content: str) -> str:
        """Generate a cache key for the entity context."""
        content_hash = hashlib.md5(f"{entity_text}|{entity_type}|{paragraph_content}".encode()).hexdigest()
        return f"{entity_text}_{entity_type}_{content_hash[:8]}"
    
    def _generate_context_single(self, entity_text: str, entity_type: str, paragraph_content: str) -> str:
        """Generate context for a single entity."""
        try:
            result = self.context_predictor(
                entity_text=entity_text,
                entity_type=entity_type,
                paragraph_content=paragraph_content
            )
            return result.context
        except Exception as e:
            logging.warning(f"Failed to generate context for entity '{entity_text}': {e}")
            return f"Entity '{entity_text}' of type '{entity_type}' appears in the text."
    
    def _generate_context_batch(self, entities_batch: List[Tuple[str, str, str]]) -> List[str]:
        """Generate context for a batch of entities efficiently."""
        contexts = []
        
        for entity_text, entity_type, paragraph_content in entities_batch:
            cache_key = self._generate_cache_key(entity_text, entity_type, paragraph_content)
            
            if self.cache_results and cache_key in self.cache:
                contexts.append(self.cache[cache_key])
                continue
            
            context = self._generate_context_single(entity_text, entity_type, paragraph_content)
            
            if self.cache_results:
                self.cache[cache_key] = context
            
            contexts.append(context)
        
        return contexts
    
    def generate_contexts_for_entities(self, entities: List[Dict], paragraph_content: str) -> List[Dict]:
        """
        Generate context for a list of entities in a paragraph.
        
        Args:
            entities: List of entity dictionaries
            paragraph_content: The paragraph content
            
        Returns:
            List of entities with context added
        """
        if not entities:
            return entities
        
        # Prepare batch data
        batch_data = []
        for entity in entities:
            entity_text = entity.get("text", "")
            entity_type = entity.get("label", "ENTITY")
            batch_data.append((entity_text, entity_type, paragraph_content))
        
        # Process in batches
        contexts = []
        for i in range(0, len(batch_data), self.batch_size):
            batch = batch_data[i:i + self.batch_size]
            batch_contexts = self._generate_context_batch(batch)
            contexts.extend(batch_contexts)
        
        # Add context to entities
        for entity, context in zip(entities, contexts):
            entity["context"] = context
        
        return entities

def extract_entities_from_chunked_json_with_context(
    input_file_path: str, 
    output_file_path: str, 
    nlp: Optional[spacy.language.Language] = None,
    ollama_model: str = "llama3.2",
    batch_size: int = 5,
    cache_contexts: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Process a semantically chunked JSON file and extract entities with context from each paragraph.
    
    Args:
        input_file_path: Path to the input JSON file with chunked data
        output_file_path: Path where the processed JSON with entities and context will be saved
        nlp: Optional spaCy model (will create one if not provided)
        ollama_model: Ollama model to use for context generation (default: llama3.2)
        batch_size: Number of entities to process in each batch for context generation
        cache_contexts: Whether to cache context results for efficiency
        verbose: Whether to print progress information
    
    Returns:
        The processed data dictionary with entities and context added to each paragraph
    """
    
    # Set up logging
    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    # Initialize spaCy model if not provided
    if nlp is None:
        if verbose:
            logger.info("Loading spaCy model...")
        nlp = spacy.load("en_core_sci_sm")
        
        # Add entity linker if not already present
        if "scispacy_linker" not in nlp.pipe_names:
            if verbose:
                logger.info("Adding scispacy_linker to pipeline...")
            nlp.add_pipe("scispacy_linker", config={"linker_name": "umls", "resolve_abbreviations": True})
    
    # Initialize context generator
    if verbose:
        logger.info(f"Initializing context generator with Ollama model: {ollama_model}")
    
    context_generator = EntityContextGenerator(
        model_name=ollama_model,
        batch_size=batch_size,
        cache_results=cache_contexts
    )
    
    # Load the input JSON file
    if verbose:
        logger.info(f"Loading data from {input_file_path}")
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in input file: {e}")
    
    # Create a deep copy of the data to avoid modifying the original
    processed_data = json.loads(json.dumps(chunks_data))
    
    # Get total chunks for progress tracking
    total_chunks = len(processed_data.get("chunks", []))
    total_entities_found = 0
    total_contexts_generated = 0
    
    if verbose:
        logger.info(f"Processing {total_chunks} chunks...")
    
    # Process each chunk
    for chunk_idx, chunk in enumerate(processed_data.get("chunks", []), 1):
        chunk_number = chunk.get("chunk_number", chunk_idx)
        
        if verbose:
            logger.info(f"Processing chunk {chunk_number}/{total_chunks}: {chunk.get('title', 'Untitled')}")
        
        # Process each paragraph in the chunk
        for paragraph in chunk.get("paragraphs", []):
            content = paragraph.get("content", "")
            paragraph_type = paragraph.get("type", "text")
            
            # Only process text paragraphs (skip headers, lists, etc.)
            if paragraph_type == "text" and content.strip():
                # Extract entities from the paragraph content
                doc = nlp(content)
                entities = []
                
                for ent in doc.ents:
                    entity_info = {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "description": spacy.explain(ent.label_) if spacy.explain(ent.label_) else "Unknown entity type"
                    }
                    
                    # Try to get UMLS information
                    if hasattr(ent._, 'umls_ents') and ent._.umls_ents:
                        umls_ents = ent._.umls_ents
                        if umls_ents:
                            cui, score = umls_ents[0]
                            entity_info["cui"] = cui
                            entity_info["score"] = float(score)
                            try:
                                entry = nlp.get_pipe("scispacy_linker").kb.cui_to_entity[cui]
                                entity_info["canonical_name"] = entry.canonical_name
                            except KeyError:
                                entity_info["canonical_name"] = None
                    elif hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                        # Fallback to kb_ents if umls_ents is not available
                        kb_ents = ent._.kb_ents
                        if kb_ents:
                            cui, score = kb_ents[0]
                            entity_info["cui"] = cui
                            entity_info["score"] = float(score)
                            entity_info["canonical_name"] = None
                    else:
                        # No entity linking available
                        entity_info["cui"] = None
                        entity_info["score"] = None
                        entity_info["canonical_name"] = None
                    
                    entities.append(entity_info)
                
                # Generate context for entities
                if entities:
                    if verbose:
                        logger.info(f"  Generating context for {len(entities)} entities using {ollama_model}...")
                    
                    start_time = time.time()
                    entities_with_context = context_generator.generate_contexts_for_entities(entities, content)
                    end_time = time.time()
                    
                    total_contexts_generated += len(entities)
                    
                    if verbose:
                        logger.info(f"  Context generation completed in {end_time - start_time:.2f}s")
                        # Show example contexts
                        for entity in entities_with_context[:2]:
                            context_preview = entity['context'][:100] + "..." if len(entity['context']) > 100 else entity['context']
                            logger.info(f"    - {entity['text']}: {context_preview}")
                
                # Add entities to the paragraph
                paragraph["entities"] = entities_with_context if entities else []
                total_entities_found += len(entities)
                
            else:
                # For non-text paragraphs, add empty entities list
                paragraph["entities"] = []
    
    # Save the processed data
    if verbose:
        logger.info(f"Saving results to {output_file_path}")
    
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"Failed to save output file: {e}")
    
    if verbose:
        logger.info(f"Entity extraction and context generation completed!")
        logger.info(f"Total entities found: {total_entities_found}")
        logger.info(f"Total contexts generated: {total_contexts_generated}")
        logger.info(f"Cache hits: {len(context_generator.cache)}")
        logger.info(f"Results saved to: {output_file_path}")
    
    return processed_data


def extract_entities_from_chunked_json_with_context_simple(
    input_file_path: str, 
    output_file_path: str
) -> Dict[str, Any]:
    """
    Simplified version of the entity extraction with context function using llama3.2.
    
    Args:
        input_file_path: Path to the input JSON file with chunked data
        output_file_path: Path where the processed JSON with entities and context will be saved
    
    Returns:
        The processed data dictionary with entities and context added to each paragraph
    """
    return extract_entities_from_chunked_json_with_context(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        ollama_model="llama3.2",
        batch_size=5,
        cache_contexts=True,
        verbose=True
    )


try:
    result = extract_entities_from_chunked_json_with_context(
        input_file_path=document_path,
        output_file_path=document_output_path,
        ollama_model="llama3.2",
        batch_size=3,  # Smaller batch size for testing
        cache_contexts=True,
        verbose=True
    )
    print("Entity extraction with context completed successfully!")
    
except Exception as e:
    print(f"Error during entity extraction with context: {e}")


# Loading UMLS 
import pandas as pd

mrdef = pd.read_csv("./2025AA-full/data-full/2025AA/META/MRDEF.RRF", delimiter='|', header=None, dtype=str) # definitions
mrsty = pd.read_csv("./2025AA-full/data-full/2025AA/META/MRSTY.RRF", delimiter='|', header=None, dtype=str) # semantic types
mrrel = pd.read_csv("./2025AA-full/data-full/2025AA/META/MRREL.RRF", delimiter='|', header=None, dtype=str) # relationships

import json
import pandas as pd
import os

def entity_definition_and_semantic_type(json_file_path, mrdef_path, mrsty_path):
    """
    Process a JSON file containing semantically chunked paragraphs with entities and context.
    For each entity with a CUI, find its definition and semantic type from UMLS data.
    Update the entity's label to the semantic type and description to the definition.
    Modifies the input JSON file directly.
    
    Args:
        json_file_path (str): Path to the JSON file with entities (will be modified in-place)
        mrdef_path (str): Path to the MRDEF.RRF file containing definitions
        mrsty_path (str): Path to the MRSTY.RRF file containing semantic types
    
    Returns:
        None: The input JSON file is modified directly
    """
    
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load UMLS data files
    # MRDEF.RRF columns: CUI, AUI, ATUI, SATUI, SAB, DEF, SUPPRESS, CVF
    # We need CUI (column 0) and DEF (column 5)
    mrdef = pd.read_csv(mrdef_path, sep='|', header=None, dtype=str)
    
    # MRSTY.RRF columns: CUI, TUI, STN, STY, ATUI, CVF  
    # We need CUI (column 0) and STY (column 3)
    mrsty = pd.read_csv(mrsty_path, sep='|', header=None, dtype=str)
    
    # Process each chunk and its paragraphs
    for chunk in data['chunks']:
        for paragraph in chunk['paragraphs']:
            if 'entities' in paragraph and paragraph['entities']:
                for entity in paragraph['entities']:
                    if 'cui' in entity and entity['cui']:
                        cui = entity['cui']
                        
                        # Find definition from MRDEF
                        try:
                            definition_row = mrdef.loc[mrdef[0] == cui, 5]
                            if not definition_row.empty:
                                definition = definition_row.iloc[0]
                                entity['description'] = definition
                        except Exception as e:
                            print(f"Error finding definition for CUI {cui}: {e}")
                        
                        # Find semantic type from MRSTY
                        try:
                            semantic_type_row = mrsty.loc[mrsty[0] == cui, 3]
                            if not semantic_type_row.empty:
                                semantic_type = semantic_type_row.iloc[0]
                                entity['label'] = semantic_type
                        except Exception as e:
                            print(f"Error finding semantic type for CUI {cui}: {e}")
    
    # Write the modified data back to the original file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_updated_json(data, output_file_path):
    """
    Save the updated JSON data to a new file.
    
    Args:
        data (dict): The updated JSON data
        output_file_path (str): Path where to save the updated JSON file
    """
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


json_file = document_output_path    
mrdef_file = "2025AA-full/data-full/2025AA/META/MRDEF.RRF"
mrsty_file = "2025AA-full/data-full/2025AA/META/MRSTY.RRF"

entity_definition_and_semantic_type(json_file, mrdef_file, mrsty_file)
print(f"Input file {json_file} has been updated with entity definitions and semantic types") 
