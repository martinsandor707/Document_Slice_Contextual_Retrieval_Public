"""Measures how much GPU memory the other GPU users of the March 2026 notebooks hold next to the Ollama runner: the
Docling converter (layout + table models, RapidOCR on torch) after converting one corpus PDF, and the nomic embedder
exactly as cell 6 of RnD/anthropic_traditional_chunking.ipynb loads it (LanceDB registry "huggingface",
TransformersEmbeddingFunction, fp32 on CUDA) after embedding all 382 chunks. Reports nvidia-smi total used memory
after each step. With --hold docling|both the process keeps the models resident and prints "ready", so
ollama_offload_under_vram_pressure.py can use it as a realistic co-resident load instead of a synthetic ballast.
Usage: HF_HUB_OFFLINE=1 .venv/bin/python RnD/verification/coresident_gpu_footprint.py [--hold docling|both] [pdf]"""
import glob, os, subprocess, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from _common import RND, SPLIT, load_docs
hold = sys.argv[sys.argv.index("--hold") + 1] if "--hold" in sys.argv else None
args = [a for a in sys.argv[1:] if not a.startswith("--") and a != hold]
def nvsmi(): return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
pdf = args[0] if args else sorted(p for p in glob.glob(str(RND / "**" / "*.pdf"), recursive=True) if "citds_article" not in p)[0]
import torch
base = nvsmi(); print(f"GPU used before anything: {base} MiB", flush=True)
from docling.document_converter import DocumentConverter
t0 = time.time(); conv = DocumentConverter(); doc = conv.convert(pdf).document
after_docling = nvsmi(); print(f"Docling: converted {os.path.basename(pdf)[:60]} in {time.time()-t0:.0f} s -> GPU used {after_docling} MiB (+{after_docling-base} MiB); torch reserved {torch.cuda.memory_reserved()/2**20:.0f} MiB", flush=True)
if hold == "docling":
    print("ready", flush=True)
    while True: time.sleep(1)
from lancedb.embeddings import get_registry
texts = [c["text"] for ch in load_docs(SPLIT).values() for c in ch]
t0 = time.time(); hf = get_registry().get("huggingface").create(name="nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cuda")
after_load = nvsmi(); print(f"nomic embedder loaded via LanceDB registry (fp32) -> GPU used {after_load} MiB (+{after_load-after_docling} MiB)", flush=True)
emb = hf.compute_source_embeddings(texts)
after_embed = nvsmi(); print(f"embedded {len(texts)} chunks in {time.time()-t0:.0f} s, dim {len(emb[0])} -> GPU used {after_embed} MiB (+{after_embed-after_load} MiB); torch reserved {torch.cuda.memory_reserved()/2**20:.0f} MiB, max allocated {torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
print(f"co-resident footprint, Docling + embedder: {after_embed-base} MiB = {(after_embed-base)/1024:.2f} GiB (Docling alone {(after_docling-base)/1024:.2f} GiB)", flush=True)
if hold == "both":
    print("ready", flush=True)
    while True: time.sleep(1)
