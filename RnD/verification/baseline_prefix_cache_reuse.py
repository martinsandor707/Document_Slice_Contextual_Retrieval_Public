"""Checks whether consecutive chunks of one document in the full-document baseline could reuse Ollama's prompt
(prefix) cache. Cell 3 of RnD/anthropic_traditional_chunking.ipynb executes `entire_doc = "FULL DOCUMENT:\\n" + entire_doc`
inside the per-chunk loop, so chunk j of a document is sent with j+1 copies of the header, which changes the token
sequence right after the system prompt. Sends chunk 0 then chunk 1 of the longest document (a) with the notebook's
accumulating header and (b) with a constant header, and prints Ollama's prompt_eval_count / prompt_eval_duration for
each call: a cache hit shows up as a prompt_eval_count far below the full prompt length on the second call.
(prompt_eval_count is used here only as a cache-hit diagnostic, not as a token statistic for the manuscript.)
Usage: .venv/bin/python RnD/verification/baseline_prefix_cache_reuse.py <host:port>"""
import re, sys, unicodedata, ollama
from _common import RND, SPLIT, load_docs
HOST = sys.argv[1]; client = ollama.Client(host=f"http://{HOST}"); MODEL = "gemma3:4b-it-qat"
SYSTEM = re.search(r'^SYSTEM\s+"""(.*?)"""', open(RND / "Models" / "chunker_anthropic.Modelfile", encoding="utf-8").read(), flags=re.S | re.M).group(1)
def clean(chunks):
    out = []
    for c in chunks:
        c = unicodedata.normalize("NFKD", c).replace(" ", " ").translate(str.maketrans({"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"'}))
        c = re.sub(r"http\S+", "", c); c = re.sub(r"[ \t]+", " ", c); out.append(re.sub(r"\n\s*\n", "\n\n", c).strip())
    return out
docs = load_docs(SPLIT); name = max(docs, key=lambda n: sum(len(c["text"]) for c in docs[n])); chunks_str = clean([c["text"] for c in docs[name]])
doc = " ".join(chunks_str)
def call(entire_doc, j):
    r = client.chat(model=MODEL, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": entire_doc}, {"role": "user", "content": f"CHUNK:\n{chunks_str[j]}"}], options={"temperature": 0.0, "num_ctx": 30000})
    return r["prompt_eval_count"], r["prompt_eval_duration"] / 1e9, r["total_duration"] / 1e9
print(f"{HOST} version {client.list and ollama.Client(host=f'http://{HOST}')._client.get('/api/version').json()['version']}; document {name}, {len(chunks_str)} chunks, {len(doc)} chars")
for label, headers in (("notebook (accumulating header)", lambda j: "FULL DOCUMENT:\n" * (j + 1) + doc), ("constant header", lambda j: "FULL DOCUMENT:\n" + doc)):
    client.generate(model=MODEL, prompt="", keep_alive=0); import time; time.sleep(3)
    rows = [call(headers(j), j) for j in (0, 1, 2)]
    print(f"  {label:32s} " + " | ".join(f"chunk {j}: evaluated {n} tok in {d:.1f} s (call {t:.1f} s)" for j, (n, d, t) in zip((0, 1, 2), rows)))
client.generate(model=MODEL, prompt="", keep_alive=0)
