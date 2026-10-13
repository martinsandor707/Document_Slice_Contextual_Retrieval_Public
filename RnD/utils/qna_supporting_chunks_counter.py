import json
from collections import Counter
import sys

def count_frequencies_from_file(file_path):
    try:
        # Open the file in read mode
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate that the JSON root is a list
        if not isinstance(data, list):
            print("Error: The JSON root must be an array (list).")
            return

        # Extract lengths of 'supporting_chunks'
        # .get() handles cases where the key is missing by defaulting to an empty list
        lengths = [len(item.get("supporting_chunks", [])) for item in data]

        # Calculate frequencies
        frequency_map = Counter(lengths)

        # Print Output
        print(f"{'Elements':<10} | {'Frequency':<10}")
        print("-" * 25)

        for length, count in sorted(frequency_map.items()):
            print(f"{length:<10} | {count}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not valid JSON.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    file_name = "q_and_a/Gemini/scientific_multi_chunk_control.json" 
    count_frequencies_from_file(file_name)