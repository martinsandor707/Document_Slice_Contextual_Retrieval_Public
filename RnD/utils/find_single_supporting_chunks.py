import json
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python find_single_supporting_chunks.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Filter objects where supporting_chunks has exactly 1 element
        single_chunk_objects = [
            obj for obj in data 
            if isinstance(obj.get('supporting_chunks'), list) and len(obj['supporting_chunks']) == 1
        ]
        
        # Print the "document" value of each filtered object
        for obj in single_chunk_objects:
            if 'document' in obj:
                print(obj['document'])
    
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{json_file}' is not valid JSON.")
        sys.exit(1)

if __name__ == "__main__":
    main()
