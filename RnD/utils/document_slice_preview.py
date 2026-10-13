from tqdm import tqdm
import ollama

MAX_TOKENS = 2_000
OLLAMA_MODEL_NAME = "any_model_here"

chunks = [...]  # Assume this is already calculated.
chunks_str = [...] # Same as `chunks` but only the text content of each chunk

for chunk in tqdm(chunks, desc="Adding context for chunks...", leave=False):    
    doc_slice = ""
    chunk_index = chunks.index(chunk)

    context_length = 16_000 # Reduce window to save memory
    context_length = context_length - 2 * MAX_TOKENS # We need to reserve space for the chunk itself (twice, the context also contains the chunk)
    total_context_chunk_number = context_length // (MAX_TOKENS*2) # Number of chunks taken from before and after the current chunk to create document slice. 

    start_index_original = chunk_index - total_context_chunk_number
    start_index_truncated = max(0, start_index_original) # Avoid index out of bounds

    end_index_original = chunk_index + total_context_chunk_number
    end_index_truncated = min(len(chunks)-1, end_index_original) # Avoid index out of bounds

    if start_index_original < 0: # We are at the start of the document, so we need to add more chunks at the end
        end_index_truncated = min(len(chunks)-1, end_index_truncated + abs(start_index_original))
    if end_index_original > len(chunks)-1: # We are at the end of the document, so we need to add more chunks at the start
        start_index_truncated = max(0, start_index_truncated - abs(end_index_original - end_index_truncated))

    for i in range(start_index_truncated, end_index_truncated + 1):
        doc_slice += " " + chunks_str[i]

    doc_slice = "FULL DOCUMENT:\n" + doc_slice
    ollama_prompt = f"CHUNK:\n{chunks_str[chunk_index]}"
    history =  [{'role': 'user', 'content': doc_slice}, {'role': 'user', 'content': ollama_prompt}]

    response = ollama.chat(
        model=OLLAMA_MODEL_NAME,
        messages=history,
    )
    context = response['message']['content']


    text_to_embed = context + chunks_str[chunk_index]