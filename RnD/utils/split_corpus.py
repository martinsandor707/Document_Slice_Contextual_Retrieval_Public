import json
import os
from collections import defaultdict

def split_json_by_document(input_file_path, output_directory):
    """
    Reads a JSON file containing chunks from multiple documents and 
    splits it into separate JSON files for each unique document.
    """
    
    # 1. Check if input file exists
    if not os.path.exists(input_file_path):
        print(f"Error: The file '{input_file_path}' was not found.")
        return

    # 2. Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    try:
        # 3. Load the data
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("Error: The JSON root is not a list/array.")
            return

        # 4. Group chunks by document name
        docs_map = defaultdict(list)
        
        for chunk in data:
            doc_name = chunk.get('document')
            if doc_name:
                chunk_without_context = {'text': chunk["original_text"], "document":chunk["document"], "id":chunk["id"]}
                docs_map[doc_name].append(chunk_without_context)
            else:
                print("Warning: Found a chunk without a 'document' key. Skipping.")

        print(f"Found {len(docs_map)} unique documents in the corpus.")
        
        # 5. Write separate files
        for doc_name, chunks in docs_map.items():
            # Create a safe filename (handle paths or extensions in the doc name)
            safe_name = os.path.basename(doc_name)
            
            # Ensure the output file ends in .json
            if not safe_name.lower().endswith('.json'):
                output_filename = f"{safe_name}.json"
            else:
                output_filename = safe_name

            output_path = os.path.join(output_directory, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(chunks, out_f, indent=4, ensure_ascii=False)
            
            print(f"Saved {len(chunks)} chunks to: {output_path}")

        print("\nSuccess! Splitting complete.")

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Please check if the input file is valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- CONFIGURATION ---
# Change these strings to match your actual file structure
INPUT_JSON_FILE = 'preprocessed_chunks/tobacco_sliding.json'   # The name of your big file
OUTPUT_DIR = 'split_documents/smoking'    # The folder where new files will appear

if __name__ == "__main__":
    split_json_by_document(INPUT_JSON_FILE, OUTPUT_DIR)