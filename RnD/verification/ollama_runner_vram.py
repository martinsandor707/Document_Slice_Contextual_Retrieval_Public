"""Measures the GPU memory footprint of the gemma3:4b-it-qat Ollama runner, i.e. the 'peak VRAM' row of Figure 6 and
the memory statement of Sect. 2.5: the fixed-radius ablation ran every k (0..4) through the same model with
num_ctx 16 384 (Models/chunker_full_doc.Modelfile), and Ollama reserves the KV cache for the configured window when the
model is loaded, so the resident footprint is a property of the window, not of k. For each window the script
  1. makes sure no model is loaded, reads the idle GPU memory (nvidia-smi),
  2. sends the summariser system prompt plus the LONGEST k=3 prompt of the corpus (window + chunk, as the notebook
     builds it) with temperature 0, sampling nvidia-smi every 0.25 s in the background and recording the peak,
  3. reads /api/ps for the runner's reported size and size_vram (the CPU/GPU split when the window does not fit),
  4. unloads the model.
Windows: 16 384 (all slice runs incl. k = 4, whose longest prompts are truncated to the window) and 30 000 (baseline).
Numbers are for the Ollama version installed today, not for the April 2026 runs. GPU: do not run anything else.
Run from the repository root:  .venv/bin/python RnD/verification/ollama_runner_vram.py"""
import json, re, subprocess, threading, time, urllib.request
import ollama
from _common import RND, SPLIT, load_docs
import sys; sys.path.insert(0, str(RND / "verification"))
from prompt_volume_and_attention_proxy import slice_chars

MODEL = "gemma3:4b-it-qat"
mf = open(RND / "Models" / "chunker_full_doc.Modelfile", encoding="utf-8").read()
SYSTEM = re.search(r'^SYSTEM\s+"""(.*?)"""', mf, flags=re.S | re.M).group(1)

def nvsmi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])

def ps():
    with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=5) as r:
        return json.load(r)

# longest k=3 prompt of the corpus (window + chunk), built exactly like the generation notebooks
docs = load_docs(SPLIT)
best = None
for name, chunks in docs.items():
    for i, c in enumerate(chunks):
        n = slice_chars(chunks, i, 3) + len(c["text"])
        if best is None or n > best[0]:
            best = (n, name, i)
n_chars, name, i = best
chunks = docs[name]; k = 3
lo, hi = i - k, i + k
lo_t, hi_t = max(0, lo), min(len(chunks) - 1, hi)
if lo < 0: hi_t = min(len(chunks) - 1, hi_t + abs(lo))
if hi > len(chunks) - 1: lo_t = max(0, lo_t - abs(hi - hi_t))
doc_slice = "FULL DOCUMENT:\n" + "".join(" " + chunks[j]["text"] for j in range(lo_t, hi_t + 1))
messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": doc_slice}, {"role": "user", "content": f"CHUNK:\n{chunks[i]['text']}"}]
print(f"probe prompt: longest k=3 prompt of the corpus = {n_chars} characters (~{n_chars/4.169:.0f} nomic tokens), {name} chunk {i}")

results = {}
for num_ctx in (16384, 30000):
    subprocess.run(["ollama", "stop", MODEL], check=False, capture_output=True)
    time.sleep(3)
    idle = nvsmi()
    peak = [idle]; stop = False
    def sampler():
        while not stop:
            peak.append(nvsmi()); time.sleep(0.25)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    t0 = time.time()
    resp = ollama.chat(model=MODEL, messages=messages, options={"temperature": 0.0, "num_ctx": num_ctx})
    elapsed = time.time() - t0
    loaded = nvsmi()
    info = [m for m in ps().get("models", []) if m["name"].startswith("gemma3")]
    stop = True; th.join()
    m = info[0] if info else {}
    results[num_ctx] = dict(idle_mib=idle, peak_mib=max(peak), after_mib=loaded, ps_size_gib=m.get("size", 0) / 2**30,
                            ps_size_vram_gib=m.get("size_vram", 0) / 2**30, prompt_eval_count=resp.get("prompt_eval_count"),
                            seconds=elapsed, load_s=resp.get("load_duration", 0) / 1e9, prompt_eval_s=resp.get("prompt_eval_duration", 0) / 1e9)
    r = results[num_ctx]
    print(f"num_ctx {num_ctx}: idle {idle} MiB -> peak {r['peak_mib']} MiB during the call (runner footprint {(r['peak_mib']-idle)/1024:.2f} GiB), "
          f"after {loaded} MiB; /api/ps size {r['ps_size_gib']:.2f} GiB, of which on GPU {r['ps_size_vram_gib']:.2f} GiB "
          f"({100*r['ps_size_vram_gib']/r['ps_size_gib'] if r['ps_size_gib'] else 0:.0f} %); prompt_eval_count {r['prompt_eval_count']} tokens "
          f"(nomic proxy ~{n_chars/4.169:.0f}), load {r['load_s']:.1f} s, prompt eval {r['prompt_eval_s']:.1f} s, total {elapsed:.1f} s")
subprocess.run(["ollama", "stop", MODEL], check=False, capture_output=True)
print("ollama version:", subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip())
json.dump(results, open(RND / "verification" / "ollama_runner_vram.json", "w"), indent=1)
print("saved RnD/verification/ollama_runner_vram.json")
