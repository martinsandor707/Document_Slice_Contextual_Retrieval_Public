"""Run GPU jobs strictly one after another (Ollama and torch/LanceDB never overlap).  Usage: python -u chain.py <step> [...]

steps:  wait            - block until GPU memory in use < 1500 MiB (up to 30 min), so Ollama loads fully on the GPU
        real:<tag>      - run_pipeline.py <tag>  (generation notebook -> benchmark cells)
        sweep:<groups>  - sweep.py <group> ...  (offline simulator; comma-separated groups)
"""
import os
import subprocess
import sys
import time

S = os.path.dirname(os.path.abspath(__file__))
PY = "/home/martin/projects/TDK/Document_Slice_Contextual_Retrieval/.venv/bin/python"
RND = "/home/martin/projects/TDK/Document_Slice_Contextual_Retrieval/RnD"


def vram_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def wait(max_used=1500, timeout=1800):
    t0 = time.time()
    while vram_used() > max_used and time.time() - t0 < timeout:
        print(f"[chain] waiting for GPU memory: {vram_used()} MiB in use", flush=True)
        time.sleep(30)
    print(f"[chain] GPU memory in use at start: {vram_used()} MiB", flush=True)


def sh(args):
    print(f"[chain] {time.strftime('%H:%M:%S')} start: {' '.join(args)}", flush=True)
    t0 = time.time()
    r = subprocess.run(args, cwd=RND)
    print(f"[chain] {time.strftime('%H:%M:%S')} done (exit {r.returncode}) in {(time.time()-t0)/60:.1f} min", flush=True)
    return r.returncode


if __name__ == "__main__":
    for step in sys.argv[1:]:
        if step == "wait":
            wait()
        elif step.startswith("real:"):
            subprocess.run(["ollama", "stop", "gemma3:4b-it-qat"], check=False)
            sh([PY, "-u", os.path.join(S, "run_pipeline.py"), step.split(":", 1)[1]])
            subprocess.run(["ollama", "stop", "gemma3:4b-it-qat"], check=False)
        elif step.startswith("sweep:"):
            subprocess.run(["ollama", "stop", "gemma3:4b-it-qat"], check=False)
            sh([PY, "-u", os.path.join(S, "sweep.py")] + step.split(":", 1)[1].split(","))
        elif step.startswith("routers:"):
            # router passes on Ollama (one forward pass per chunk, no summariser); torch steps never run at the same time
            wait()
            sh([PY, "-u", os.path.join(S, "router_variants.py")] + step.split(":", 1)[1].split(","))
            subprocess.run(["ollama", "stop", "gemma3:4b-it-qat"], check=False)
        elif step.startswith("evalreal:"):
            # per-question evaluation of the LanceDB table a real run just built (torch only) -> res_real_<tag>.json
            subprocess.run(["ollama", "stop", "gemma3:4b-it-qat"], check=False)
            tag = step.split(":", 1)[1]
            code = (f"import sys; sys.path.insert(0, {S!r}); import evalkit as ek; "
                    f"res = ek.evaluate(ek.open_rnd_table('ablation_doc_slice_radius_dynamic')); ek.save(res, {os.path.join(S, 'res_real_') + '%s.json' % tag!r}); "
                    f"print(ek.fmt(res, 'real ' + {tag!r}))")
            sh([PY, "-c", code])
    print("[chain] ALL DONE", flush=True)
