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

# Example usage
if __name__ == "__main__":
    # Define file paths
    json_file = "semantically_chunked_paragraphs_UNMC_with_entities_and_context.json"
    mrdef_file = "2025AA-full/data-full/2025AA/META/MRDEF.RRF"
    mrsty_file = "2025AA-full/data-full/2025AA/META/MRSTY.RRF"
    
    # Check if files exist
    if not os.path.exists(json_file):
        print(f"Error: JSON file {json_file} not found")
    elif not os.path.exists(mrdef_file):
        print(f"Error: MRDEF file {mrdef_file} not found")
    elif not os.path.exists(mrsty_file):
        print(f"Error: MRSTY file {mrsty_file} not found")
    else:
        # Process the data and modify the input file directly
        print("Processing entities and updating input file...")
        entity_definition_and_semantic_type(json_file, mrdef_file, mrsty_file)
        print(f"Input file {json_file} has been updated with entity definitions and semantic types") 