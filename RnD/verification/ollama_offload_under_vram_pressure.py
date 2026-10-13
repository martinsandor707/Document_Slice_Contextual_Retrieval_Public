"""How many Gemma 3 layers an Ollama release keeps on the 8 GB RTX 4060, and how much CPU it burns, as a function of
the prompt, the context window and the GPU memory already taken by other processes.
Prompt modes: "k3" = the longest k=3 Document Slice prompt of the corpus at num_ctx 30000 and 16384 (as in
ollama_runner_vram.py); "full" = the full-document baseline exactly as cell 3 of RnD/anthropic_traditional_chunking.ipynb
builds it (system prompt of RnD/Models/chunker_anthropic.Modelfile, cleaned chunks joined by spaces behind
"FULL DOCUMENT:", longest document s41586-019-1138-y with its longest chunk) at the baseline's num_ctx 30000.
Co-resident load: a number = GiB held by gpu_ballast.py; "docling" = coresident_gpu_footprint.py --hold docling (a live
Docling converter after one PDF conversion); "both" = Docling plus the LanceDB nomic embedder.
Per window it records: offloaded layers (server log), `ollama ps` output, nvidia-smi peak, /api/ps size and size_vram,
host RSS of the server processes, prompt_eval timings, CPU time of the server processes for the cold call (includes model
load) and for a warm call with a nonce-prefixed prompt (full prompt re-evaluation, as every baseline chunk caused),
expressed as cores busy and as a percentage of the 24 hardware threads (what a system monitor shows).
Usage: .venv/bin/python RnD/verification/ollama_offload_under_vram_pressure.py <host:port> <label> <log path|journal>
       <ballast GiB|docling|both> [--prompt k3|full] [--ollama-bin PATH]
Writes RnD/verification/ollama_offload_pressure_<label>.json (results + relevant server log lines)."""
import argparse, json, os, re, subprocess, sys, threading, time, unicodedata, urllib.request
import ollama
from _common import RND, SPLIT, load_docs
sys.path.insert(0, str(RND / "verification"))
from prompt_volume_and_attention_proxy import slice_chars

ap = argparse.ArgumentParser(); ap.add_argument("host"); ap.add_argument("label"); ap.add_argument("logsrc"); ap.add_argument("ballast")
ap.add_argument("--prompt", default="k3", choices=["k3", "full"]); ap.add_argument("--ollama-bin", default="ollama"); A = ap.parse_args()
HOST, LABEL, LOGSRC = A.host, A.label, A.logsrc
MODEL = "gemma3:4b-it-qat"; THREADS = os.cpu_count()
client = ollama.Client(host=f"http://{HOST}")
def system_prompt(modelfile):
    mf = open(RND / "Models" / modelfile, encoding="utf-8").read()
    return re.search(r'^SYSTEM\s+"""(.*?)"""', mf, flags=re.S | re.M).group(1)
def clean(chunks):                      # clean_docling_chunk_strings of the notebooks, verbatim
    out = []
    for chunk in chunks:
        chunk = unicodedata.normalize("NFKD", chunk).replace(" ", " ")
        chunk = chunk.translate(str.maketrans({"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"'}))
        chunk = re.sub(r"http\S+", "", chunk); chunk = re.sub(r"[ \t]+", " ", chunk); chunk = re.sub(r"\n\s*\n", "\n\n", chunk).strip()
        out.append(chunk)
    return out
def nvsmi():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
def api(path):
    with urllib.request.urlopen(f"http://{HOST}{path}", timeout=10) as r: return json.load(r)
def ollama_ps():
    env = dict(os.environ, OLLAMA_HOST=f"http://{HOST}")
    return subprocess.run([A.ollama_bin, "ps"], capture_output=True, text=True, env=env).stdout.strip()
def server_pids():
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit(): continue
        try: cmd = open(f"/proc/{d}/cmdline", "rb").read().decode(errors="ignore")
        except Exception: continue
        if ("ollama" in cmd or "llama-server" in cmd) and "python" not in cmd and "ballast" not in cmd: out.append(int(d))
    return out
def cpu_seconds(pids):
    tot = 0.0; hz = os.sysconf("SC_CLK_TCK")
    for p in pids:
        try: f = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split(); tot += (int(f[11]) + int(f[12])) / hz
        except Exception: pass
    return tot
def rss_mib(pids):
    tot = 0
    for p in pids:
        try: tot += int(open(f"/proc/{p}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // 2**20
        except Exception: pass
    return tot
def log_since(t0, pos):
    if LOGSRC == "journal":
        return subprocess.run(["journalctl", "-u", "ollama", f"--since=@{int(t0)}", "-o", "cat", "--no-pager"], capture_output=True, text=True).stdout
    return open(LOGSRC, encoding="utf-8", errors="ignore").read()[pos:]
def timed_call(messages, num_ctx):
    pids0 = server_pids(); c0 = cpu_seconds(pids0); t0 = time.time()
    resp = client.chat(model=MODEL, messages=messages, options={"temperature": 0.0, "num_ctx": num_ctx})
    el = time.time() - t0; pids = server_pids(); c1 = cpu_seconds(pids)
    return resp, el, c1 - c0, pids

docs = load_docs(SPLIT)
if A.prompt == "k3":
    SYSTEM = system_prompt("chunker_full_doc.Modelfile"); windows = (30000, 16384)
    n_chars, name, i = max(((slice_chars(ch, i, 3) + len(c["text"]), name, i) for name, ch in docs.items() for i, c in enumerate(ch)))
    chunks = docs[name]; k = 3; lo, hi = i - k, i + k; lo_t, hi_t = max(0, lo), min(len(chunks) - 1, hi)
    if lo < 0: hi_t = min(len(chunks) - 1, hi_t + abs(lo))
    if hi > len(chunks) - 1: lo_t = max(0, lo_t - abs(hi - hi_t))
    body = "FULL DOCUMENT:\n" + "".join(" " + chunks[j]["text"] for j in range(lo_t, hi_t + 1)); chunk_text = chunks[i]["text"]
    probe = f"longest k=3 slice prompt: {name} chunk {i}, {n_chars} chars"
else:
    SYSTEM = system_prompt("chunker_anthropic.Modelfile"); windows = (30000,)
    name = max(docs, key=lambda n: sum(len(c["text"]) for c in docs[n])); chunks_str = clean([c["text"] for c in docs[name]])
    i = max(range(len(chunks_str)), key=lambda j: len(chunks_str[j])); body = "FULL DOCUMENT:\n" + " ".join(chunks_str); chunk_text = chunks_str[i]
    probe = f"full-document baseline prompt: {name} ({len(chunks_str)} chunks, {len(' '.join(chunks_str))} chars) with its longest chunk {i} ({len(chunk_text)} chars); prompt total {len(SYSTEM)+len(body)+len(chunk_text)+7} chars"
def msgs(prefix=""):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prefix + body}, {"role": "user", "content": f"CHUNK:\n{chunk_text}"}]

try: client.generate(model=MODEL, prompt="", keep_alive=0)      # start from an empty GPU
except Exception: pass
for _ in range(20):
    if nvsmi() < 200: break
    time.sleep(1)
idle0 = nvsmi(); holder = None
if A.ballast not in ("0", "0.0"):
    cmd = [sys.executable, str(RND / "verification" / ("coresident_gpu_footprint.py" if A.ballast in ("docling", "both") else "gpu_ballast.py"))]
    cmd += ["--hold", A.ballast] if A.ballast in ("docling", "both") else [A.ballast]
    holder = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=open(RND / "verification" / ".holder_stderr.log", "w"), text=True, env=dict(os.environ, HF_HUB_OFFLINE="1"))
    print(f"   co-resident holder started: {' '.join(cmd[1:])}")
    for line in holder.stdout:
        if line.strip() == "ready": break
        if "GPU used" in line or "footprint" in line: print("   holder:", line.strip())
    else: sys.exit("co-resident holder exited before becoming ready; see RnD/verification/.holder_stderr.log")
    time.sleep(1)
held_mib = nvsmi() - idle0
print(f"server {HOST} version {api('/api/version')['version']}; {probe}; co-resident load '{A.ballast}' holds {held_mib} MiB; GPU used before load {nvsmi()} MiB")
results = {"version_label": LABEL, "server_version": api('/api/version')['version'], "prompt_mode": A.prompt, "probe": probe, "coresident_load": A.ballast, "coresident_mib_measured": held_mib, "cpu_threads": THREADS}
try:
    for num_ctx in windows:
        try: client.generate(model=MODEL, prompt="", keep_alive=0)
        except Exception: pass
        time.sleep(4); base = nvsmi(); peak = [base]; stop = False
        def sampler():
            while not stop: peak.append(nvsmi()); time.sleep(0.25)
        th = threading.Thread(target=sampler, daemon=True); th.start()
        t_start = time.time(); pos = 0 if LOGSRC == "journal" else os.path.getsize(LOGSRC)
        cold, el_c, cpu_c, pids = timed_call(msgs(), num_ctx)
        ps_cli = ollama_ps()
        warm, el_w, cpu_w, pids = timed_call(msgs(f"nonce-{time.time_ns()} "), num_ctx)
        stop = True; th.join(); rss = rss_mib(pids)
        m = next((x for x in api("/api/ps").get("models", []) if x["name"].startswith("gemma3")), {})
        text = log_since(t_start, pos)
        keep = [l for l in text.splitlines() if re.search(r"offload|layers|KV buffer|kv_cache: size|memory success|available gpu|load request|gpu memory|flash|CPU_Mapped|compute buffer", l, re.I) and "memory_seq_rm" not in l]
        off = re.findall(r"offloaded (\d+)/(\d+) layers", text)
        r = dict(num_ctx=num_ctx, gpu_before_load_mib=base, peak_mib=max(peak), runner_footprint_gib=(max(peak) - base) / 1024, offloaded_layers=off[-1][0] + "/" + off[-1][1] if off else None,
                 ollama_ps=ps_cli, ps_size_gib=m.get("size", 0) / 2**30, ps_size_vram_gib=m.get("size_vram", 0) / 2**30, server_rss_mib=rss,
                 prompt_eval_count=cold.get("prompt_eval_count"), cold_s=el_c, load_s=cold.get("load_duration", 0) / 1e9, cold_prompt_eval_s=cold.get("prompt_eval_duration", 0) / 1e9,
                 cold_cpu_s=cpu_c, cold_cores_busy=cpu_c / el_c, warm_s=el_w, warm_prompt_eval_s=warm.get("prompt_eval_duration", 0) / 1e9, warm_eval_count=warm.get("eval_count"),
                 warm_cpu_s=cpu_w, warm_cores_busy=cpu_w / el_w, warm_cpu_pct_of_machine=100 * cpu_w / el_w / THREADS, log=keep[:60])
        results[str(num_ctx)] = r
        print(f"num_ctx {num_ctx}: offloaded {r['offloaded_layers']} layers; GPU before load {base} MiB, peak {r['peak_mib']} MiB of 8188 (runner {r['runner_footprint_gib']:.2f} GiB); /api/ps {r['ps_size_gib']:.2f} GiB, in VRAM {r['ps_size_vram_gib']:.2f} GiB; server RSS {rss} MiB\n"
              f"   ollama ps:\n      " + ps_cli.replace("\n", "\n      ") + "\n"
              f"   cold call {el_c:.1f} s (load {r['load_s']:.1f} s, prompt eval {r['cold_prompt_eval_s']:.1f} s for {r['prompt_eval_count']} tok), CPU {cpu_c:.0f} s = {r['cold_cores_busy']:.1f} cores\n"
              f"   warm call {el_w:.1f} s (prompt eval {r['warm_prompt_eval_s']:.1f} s, {r['warm_eval_count']} gen tok), CPU {cpu_w:.0f} s = {r['warm_cores_busy']:.1f} cores = {r['warm_cpu_pct_of_machine']:.0f} % of {THREADS} threads")
        for l in keep:
            if re.search(r"offloaded|KV buffer|CPU_Mapped|compute buffer", l): print("   |", l.strip()[:160])
    try: client.generate(model=MODEL, prompt="", keep_alive=0)
    except Exception: pass
finally:
    if holder: holder.terminate(); holder.wait()
json.dump(results, open(RND / "verification" / f"ollama_offload_pressure_{LABEL}.json", "w"), indent=1)
print("saved", f"RnD/verification/ollama_offload_pressure_{LABEL}.json; GPU now {nvsmi()} MiB")
