import json
import dspy
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# ------------------- Verbose signature -------------------
class MedicalSummarySignature(dspy.Signature):
    """DSPy signature for generating structured medical summaries from text chunks with entities and relationships."""
    
    chunk_content = dspy.InputField(desc="The full text content of the medical chunk/paragraph")
    entities_data = dspy.InputField(desc="JSON string containing the entities and relationships extracted from the chunk")
    
    medical_summary = dspy.OutputField(desc=""" You are a medical summarizer. Generate a structured summary from the provided medical source (report, paper, or book), strictly adhering to the following categories. The summary should list key information under each category in a concise format. Summaries should be based off of the entities and relationships extracted from the chunk. No additional explanations or detailed descriptions are necessary unless directly related to the categories:

Each category should be addressed only if relevant to the content of the medical source. Ensure the summary is clear and direct, suitable for quick reference.

Respond with the medical summary as a JSON string in this exact format, not python code or anything else. Items within angle brackets are instructions for you to fill in for that category and should not be included in the final output:
{
  "ANATOMICAL_STRUCTURE": <Mention any anatomical structures specifically discussed>,
  "BODY_FUNCTION": <List any body functions highlighted>,
  "BODY_MEASUREMENT": <Include normal measurements like blood pressure or temperature>,
  "BM_RESULT": <Results of these measurements>,
  "BM_UNIT": <Units for each measurement>,
  "BM_VALUE": <Values of these measurements>,
  "LABORATORY_DATA": <Outline any laboratory tests mentioned>,
  "LAB_RESULT": <Outcomes of these tests (e.g., 'increased', 'decreased')>,
  "LAB_VALUE": <Specific values from the tests>,
  "LAB_UNIT": <Units of measurement for these values>,
  "MEDICINE": <Name medications discussed>,
  "MED_DOSE": <Dose prescribed (e.g., '500 mg')>,
  "MED_DURATION": <Duration of use (e.g., '7 days')>,
  "MED_FORM": <Form of medication (e.g., 'tablet', 'injection')>,
  "MED_FREQUENCY": <How often the medication is taken (e.g., 'twice daily')>,
  "MED_ROUTE": <Route of administration (e.g., 'oral', 'IV')>,
  "MED_STATUS": <Status of the medication (e.g., 'ongoing', 'discontinued')>,
  "MED_STRENGTH": <Strength or concentration (e.g., '5%')>,
  "MED_UNIT": <Unit for medication (e.g., 'mg', 'ml')>,
  "MED_TOTALDOSE": <Total dose over the full course (e.g., '3500 mg')>,
  "PROBLEM": <Identify any medical conditions or findings>,
  "PROCEDURE": <Describe any procedures>,
  "PROCEDURE_RESULT": <Outcomes of these procedures>,
  "PROC_METHOD": <Methods used for procedures (e.g., 'laparoscopic')>,
  "SEVERITY": <Severity of the conditions mentioned>,
  "MEDICAL_DEVICE": <List any medical devices used>,
  "SUBSTANCE_ABUSE": <Note any substance abuse mentioned>
}
Remember to follow the instructions in the angle brackets and remove any angle brackets from the final JSON string response.

Example:
{
    "ANATOMICAL_STRUCTURE": "heart",
    "MEDICINE": "aspirin",
    "PROBLEM": "infection",
    "PROCEDURE": "surgery"
}                                    
""")

# ------------------- Slim fallback signature -------------------
class MedicalSummarySlimSignature(dspy.Signature):
    """A concise fallback prompt used if the verbose prompt returns no data."""

    chunk_content = dspy.InputField(desc="Medical paragraph text")
    entities_flat = dspy.InputField(desc="Comma-separated entity names present in the paragraph")

    medical_summary = dspy.OutputField(desc="Return ONLY JSON object with any of these keys that appear: ANATOMICAL_STRUCTURE, MEDICINE, PROBLEM, PROCEDURE. Use empty strings for missing keys. Example: {\"MEDICINE\": \"aspirin\", \"PROBLEM\": \"infection\"}")


class EntitySummarizer:
    """Generate structured medical summaries from processed chunks using DSPy and LLM."""
    
    def __init__(self, ollama_model: str = "llama3.1:8b", ollama_endpoint: str = "http://localhost:11434"):
        """
        Initialize the Entity Summarizer.
        
        Args:
            ollama_model: Ollama model to use for summarization (default: llama3.1:8b)
            ollama_endpoint: Ollama endpoint URL
        """
        self.ollama_model = ollama_model
        self.ollama_endpoint = ollama_endpoint
        
        # Initialize DSPy with Ollama for summarization
        try:
            ollama_model_string = f"ollama/{self.ollama_model}"
            self.lm = dspy.LM(
                model=ollama_model_string, 
                api_base=self.ollama_endpoint,
                temperature=0.5,  # Lower temperature for more consistent structured output
                max_tokens=500   # Increased for larger entity sets and relationships
            )
            dspy.settings.configure(lm=self.lm)
            self.summary_predictor = dspy.Predict(MedicalSummarySignature)
            logger.info(f"✓ EntitySummarizer initialized with Ollama model: {self.ollama_model}")
        except Exception as e:
            logger.error(f"Failed to initialize EntitySummarizer with Ollama: {e}")
            raise RuntimeError(f"Could not initialize EntitySummarizer with Ollama model {ollama_model}")
    
    def _parse_summary_response(self, raw_response: str) -> Dict[str, str]:
        """
        Parse the LLM response to extract the structured medical summary.
        
        Args:
            raw_response: Raw response from the LLM
            
        Returns:
            Dictionary containing the structured medical categories
        """
        try:
            # Clean the response
            response_text = str(raw_response).strip()
            
            # Remove markdown code blocks if present
            import re
            response_text = re.sub(r"^```(?:json)?|```$", "", response_text, flags=re.IGNORECASE).strip()
            
            # First try: Direct JSON parsing (when LLM returns JSON directly)
            try:
                summary_data = json.loads(response_text)
                if isinstance(summary_data, dict):
                    # Filter out empty values for cleaner output
                    filtered_data = {k: v for k, v in summary_data.items() if v and v.strip()}
                    if filtered_data:  # Only return if we have actual data
                        return filtered_data
                    else:
                        logger.debug("LLM returned empty medical summary, attempting text parsing")
            except json.JSONDecodeError:
                logger.debug("Response is not valid JSON, attempting text parsing")
            
            # Second try: Look for medical_summary field in DSPy format
            if "medical_summary:" in response_text.lower():
                # Extract content after "medical_summary:"
                parts = response_text.split(":", 1)
                if len(parts) > 1:
                    json_part = parts[1].strip()
                    try:
                        summary_data = json.loads(json_part)
                        if isinstance(summary_data, dict):
                            filtered_data = {k: v for k, v in summary_data.items() if v and v.strip()}
                            if filtered_data:
                                return filtered_data
                    except json.JSONDecodeError:
                        pass
            
            # Third try: Extract key-value pairs from text format
            fallback_summary = {}
            medical_categories = [
                "ANATOMICAL_STRUCTURE", "BODY_FUNCTION", "BODY_MEASUREMENT", "BM_RESULT", 
                "BM_UNIT", "BM_VALUE", "LABORATORY_DATA", "LAB_RESULT", "LAB_VALUE", 
                "LAB_UNIT", "MEDICINE", "MED_DOSE", "MED_DURATION", "MED_FORM", 
                "MED_FREQUENCY", "MED_ROUTE", "MED_STATUS", "MED_STRENGTH", "MED_UNIT", 
                "MED_TOTALDOSE", "PROBLEM", "PROCEDURE", "PROCEDURE_RESULT", "PROC_METHOD", 
                "SEVERITY", "MEDICAL_DEVICE", "SUBSTANCE_ABUSE"
            ]
            
            # Try to extract information from text format
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().upper().replace('"', '').replace("'", "")
                        value = parts[1].strip().replace('"', '').replace("'", "").rstrip(',')
                        if key in medical_categories and value and value.lower() not in ['', 'null', 'none']:
                            fallback_summary[key] = value
            
            # Only return fallback if we found some data
            if fallback_summary:
                return fallback_summary
            
            logger.warning("No medical information could be extracted from the response")
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to parse summary response: {e}")
            return {}
    
    def _heuristic_summary_from_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate a simple summary using entity types when the LLM fails."""
        summary: Dict[str, List[str]] = {}
        for ent in entities:
            etype = ent.get('entity_type', '').lower()
            name = ent['entity_name']
            if 'drug' in etype or 'substance' in etype or 'chemical' in etype:
                summary.setdefault('MEDICINE', []).append(name)
            elif 'procedure' in etype or 'surgery' in etype:
                summary.setdefault('PROCEDURE', []).append(name)
            elif 'disease' in etype or 'problem' in etype or 'disorder' in etype:
                summary.setdefault('PROBLEM', []).append(name)
            elif 'anatomical' in etype or 'body' in etype:
                summary.setdefault('ANATOMICAL_STRUCTURE', []).append(name)
        # Convert lists to comma-separated strings
        return {k: ', '.join(v) for k, v in summary.items()}

    def summarize_chunk(self, chunk_content: str, entities: List[Dict[str, Any]], 
                       relationships: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate a structured medical summary for a chunk with its entities and relationships.
        
        Args:
            chunk_content: The text content of the chunk
            entities: List of entities extracted from the chunk
            relationships: List of relationships between entities
            
        Returns:
            Dictionary containing structured medical summary categories
        """
        try:
            # Skip if no content or entities
            if not chunk_content.strip() or not entities:
                logger.debug("Skipping medical summary - no content or entities")
                return {}
            
            # Prepare entities and relationships data for the LLM (keep only essential fields)
            filtered_entities = []
            for entity in entities:
                # Keep only the essential fields for medical summarization
                filtered_entity = {
                    'entity_name': entity.get('entity_name', ''),
                    'entity_type': entity.get('entity_type', ''),
                    'entity_description': entity.get('entity_description', '')
                }
                filtered_entities.append(filtered_entity)
            
            # Filter relationships to keep only essential fields
            filtered_relationships = []
            for rel in relationships:
                filtered_rel = {
                    'source_entity': rel.get('source_entity', ''),
                    'target_entity': rel.get('target_entity', ''),
                    'relationship_description': rel.get('relationship_description', '')
                }
                filtered_relationships.append(filtered_rel)
            
            entities_data = {
                "entities": filtered_entities,
                "relationships": filtered_relationships
            }
            entities_json = json.dumps(entities_data, indent=2)
            
            logger.debug(f"Generating medical summary for chunk (length: {len(chunk_content)} chars, "
                        f"{len(entities)} entities, {len(relationships)} relationships)")
            
            # Generate summary using DSPy
            try:
                result = self.summary_predictor(
                    chunk_content=chunk_content,
                    entities_data=entities_json
                )
                
                # Extract the medical summary JSON from DSPy result
                raw_response = ""
                if hasattr(result, 'medical_summary'):
                    raw_response = result.medical_summary
                    logger.debug(f"DSPy medical_summary field: {raw_response[:200]}...")
                else:
                    # Fallback: try to get raw completion text
                    raw_response = str(result)
                    logger.debug(f"DSPy fallback response: {raw_response[:200]}...")
                
                # Parse the response
                summary = self._parse_summary_response(raw_response)
                
                if not summary:
                    logger.warning("LLM medical summary empty – generating heuristic fallback from entities")
                    summary = self._heuristic_summary_from_entities(filtered_entities)
                else:
                    logger.debug(f"Generated medical summary with {len(summary)} relevant categories")
                return summary
                    
            except Exception as dspy_error:
                logger.warning(f"DSPy prediction failed: {dspy_error}")
                
                # Try to extract any useful information from the error
                error_str = str(dspy_error)
                if "LM Response:" in error_str:
                    # Extract the LM response from the error message
                    response_start = error_str.find("LM Response:") + len("LM Response:")
                    response_end = error_str.find("\n\n", response_start)
                    if response_end == -1:
                        response_end = len(error_str)
                    
                    raw_response = error_str[response_start:response_end].strip()
                    logger.debug(f"Extracted response from error: {raw_response}")
                    
                    # Try to parse this extracted response
                    summary = self._parse_summary_response(raw_response)
                    if summary:
                        logger.info("Successfully recovered medical summary from error message")
                        return summary
                
                logger.warning("Could not recover medical summary from DSPy error")
                return {}
            
        except Exception as e:
            logger.warning(f"Failed to generate medical summary for chunk: {e}")
            return {}
    
    def summarize_paragraph(self, paragraph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add medical summary to a paragraph result structure.
        
        Args:
            paragraph_data: Dictionary containing paragraph content, entities, and relationships
            
        Returns:
            Updated paragraph data with medical summary added
        """
        try:
            # Extract data from paragraph structure
            content = paragraph_data.get('content', '')
            entities = paragraph_data.get('entities', [])
            relationships = paragraph_data.get('relationships', [])
            
            # Generate summary
            medical_summary = self.summarize_chunk(content, entities, relationships)
            
            # Add summary to paragraph data (after relationships as requested)
            updated_paragraph = paragraph_data.copy()
            updated_paragraph['medical_summary'] = medical_summary
            
            return updated_paragraph
            
        except Exception as e:
            logger.warning(f"Failed to add medical summary to paragraph: {e}")
            # Return original data if summarization fails
            return paragraph_data


def summarize_entities_for_results(results: Dict[str, Any], 
                                 ollama_model: str = "llama3.1:8b",
                                 ollama_endpoint: str = "http://localhost:11434") -> Dict[str, Any]:
    """
    Add medical summaries to all paragraphs in entity processing results.
    
    Args:
        results: Entity processing results dictionary
        ollama_model: Ollama model to use for summarization
        ollama_endpoint: Ollama endpoint URL
        
    Returns:
        Updated results with medical summaries added to each paragraph
    """
    try:
        # Initialize summarizer
        summarizer = EntitySummarizer(ollama_model=ollama_model, ollama_endpoint=ollama_endpoint)
        
        # Process each paragraph
        updated_results = results.copy()
        updated_paragraphs = []
        
        paragraphs = results.get('paragraphs', [])
        total_paragraphs = len(paragraphs)
        
        logger.info(f"Adding medical summaries to {total_paragraphs} paragraphs...")
        
        for idx, paragraph in enumerate(paragraphs):
            logger.info(f"Generating medical summary for paragraph {idx + 1}/{total_paragraphs}")
            
            # Add medical summary to paragraph
            updated_paragraph = summarizer.summarize_paragraph(paragraph)
            updated_paragraphs.append(updated_paragraph)
        
        # Update results
        updated_results['paragraphs'] = updated_paragraphs
        
        # Update metadata
        if 'metadata' in updated_results:
            updated_results['metadata']['medical_summaries_added'] = True
            updated_results['metadata']['summarization_model'] = ollama_model
        
        logger.info(f"✓ Medical summaries added to {total_paragraphs} paragraphs")
        return updated_results
        
    except Exception as e:
        logger.error(f"Failed to add medical summaries to results: {e}")
        return results  # Return original results if summarization fails


# Example usage and integration
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python summarize_entities.py <entity_processing_results.json> [ollama_model]")
        sys.exit(1)
    
    results_path = sys.argv[1]
    ollama_model = sys.argv[2] if len(sys.argv) > 2 else "llama3.1:8b"
    
    try:
        # Load existing entity processing results
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Add medical summaries
        updated_results = summarize_entities_for_results(results, ollama_model=ollama_model)
        
        # Save updated results
        output_path = results_path.replace('.json', '_with_summaries.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results with medical summaries saved to {output_path}")
        
        # Print summary statistics
        total_summaries = sum(1 for p in updated_results.get('paragraphs', []) 
                             if p.get('medical_summary'))
        print(f"Added medical summaries to {total_summaries} paragraphs")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)