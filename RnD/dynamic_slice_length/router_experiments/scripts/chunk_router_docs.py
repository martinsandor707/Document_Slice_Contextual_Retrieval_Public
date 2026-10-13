"""Chunk RnD/input_router/*.pdf exactly like the benchmark corpus (doc_slice_chunking.ipynb + utils/split_corpus.py):
DocumentConverter() defaults, HybridChunker(nomic tokenizer, max_tokens=2000, merge_peers=True), the same string
cleaning, id = sha256(cleaned text); one JSON list [{text, document, id}] per PDF -> RnD/split_documents_router/."""
import hashlib, json, os, re, sys, time, unicodedata
import torch
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

RND = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))   # .../RnD
INPUT_DIR, OUT_DIR = f"{RND}/input_router", f"{RND}/split_documents_router"
EMBEDDING_MODEL_NAME, MAX_TOKENS = "nomic-ai/nomic-embed-text-v1.5", 2000


def clean_docling_chunk_strings(chunks):
    cleaned_chunks = []
    for chunk in chunks:
        chunk = unicodedata.normalize("NFKD", chunk).replace(" ", " ")
        chunk = chunk.translate(str.maketrans({"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"'}))
        chunk = re.sub(r"http\S+", "", chunk)
        chunk = re.sub(r"[ \t]+", " ", chunk)
        chunk = re.sub(r"\n\s*\n", "\n\n", chunk)
        chunk = chunk.strip()
        cleaned_chunks.append(chunk)
    return cleaned_chunks


os.makedirs(OUT_DIR, exist_ok=True)
converter = DocumentConverter()
tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME), max_tokens=MAX_TOKENS)
chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
for source in sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")):
    t0 = time.time()
    doc = converter.convert(f"{INPUT_DIR}/{source}").document
    chunks = list(chunker.chunk(dl_doc=doc))
    chunks_str = clean_docling_chunk_strings([c.text for c in chunks])
    records = [{"text": t, "document": source, "id": hashlib.sha256(t.encode()).hexdigest()} for t in chunks_str]
    with open(f"{OUT_DIR}/{source}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
    print(f"{source:62s} {len(records):3d} chunks, {sum(len(t) for t in chunks_str):7d} chars, {time.time()-t0:5.0f}s", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
print("done")
