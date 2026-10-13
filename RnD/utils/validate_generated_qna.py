import json
import sys
import os

def main():
    FILES = {
        "chunks": "split_documents/smoking/smoke-free_implementation_report_en.pdf.json",
        "generated": "generated_qna.json",
        "old": "q_and_a/Gemini/smoking.json"
    }

    # --- Step 1: Load and Create Set of Valid IDs ---
    print(f"Loading IDs from {FILES['chunks']}...")
    try:
        with open(FILES['chunks'], 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            valid_ids = {item['id'] for item in chunks_data}
    except FileNotFoundError:
        print(f"Error: Could not find '{FILES['chunks']}'.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: The key {e} was missing in one of the objects in '{FILES['chunks']}'.")
        sys.exit(1)

    # --- Step 2: Load Generated QnA ---
    print(f"Loading {FILES['generated']}...")
    try:
        with open(FILES['generated'], 'r', encoding='utf-8') as f:
            generated_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{FILES['generated']}'.")
        sys.exit(1)

    # --- Step 3: Validate IDs ---
    print("Validating supporting_chunks...")
    for index, qna_item in enumerate(generated_data):
        supporting_chunks = qna_item.get("supporting_chunks", [])

        if len(supporting_chunks) == 1:
            print(f"\n[CRITICAL ERROR] Validation Failed.")
            print(f"'supporting_chunks' contains only 1 element at index {index}")
            sys.exit(1)

        if not qna_item.get("question"):
            print(f"\n[CRITICAL ERROR] Validation Failed.")
            print(f"Missing 'question' field in 'generated_qna.json' at index {index}")
            sys.exit(1)

        if not qna_item.get("gold_answer"):
            print(f"\n[CRITICAL ERROR] Validation Failed.")
            print(f"Missing 'gold_answer' field in 'generated_qna.json' at index {index}")
            sys.exit(1)

        if not qna_item.get("document"):
            print(f"\n[CRITICAL ERROR] Validation Failed.")
            print(f"Missing 'document' field in 'generated_qna.json' at index {index}")
            sys.exit(1)
        
        for chunk_id in supporting_chunks:
            if chunk_id not in valid_ids:
                print(f"\n[CRITICAL ERROR] Validation Failed.")
                print(f"Invalid ID found: '{chunk_id}'")
                print(f"Location: 'generated_qna.json' at index {index}")
                sys.exit(1)

    print("Validation successful.")

    # --- Step 4: Load Old QnA and Deduplicate ---
    old_data = []
    existing_hashes = set()

    if os.path.exists(FILES['old']):
        try:
            with open(FILES['old'], 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    old_data = json.loads(content)
                    
                    # Create a set of "signatures" for existing items
                    # We sort keys to ensure {"a":1, "b":2} is treated same as {"b":2, "a":1}
                    for item in old_data:
                        item_hash = json.dumps(item, sort_keys=True)
                        existing_hashes.add(item_hash)
                        
        except json.JSONDecodeError:
            print(f"Warning: '{FILES['old']}' contained invalid JSON. Overwriting with new data.")
            old_data = []

    # --- Step 5: Append Unique Items Only ---
    items_to_add = []
    skipped_count = 0

    for item in generated_data:
        # Generate signature for the new item
        item_hash = json.dumps(item, sort_keys=True)
        
        if item_hash not in existing_hashes:
            items_to_add.append(item)
            # Add to set so we don't add duplicates within the new batch itself
            existing_hashes.add(item_hash)
        else:
            skipped_count += 1

    if not items_to_add:
        print(f"No new unique items to add. ({skipped_count} duplicates found)")
        return

    combined_data = old_data + items_to_add

    # --- Step 6: Save ---
    try:
        with open(FILES['old'], 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=4)
        print(f"Success: Appended {len(items_to_add)} new items to '{FILES['old']}'.")
        print(f"Skipped {skipped_count} duplicates.")
    except IOError as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()