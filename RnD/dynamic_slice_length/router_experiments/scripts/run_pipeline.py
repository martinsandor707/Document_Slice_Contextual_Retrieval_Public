"""Run the REAL pipeline for the router configuration currently in RnD/utils/dynamic_slice_prediction.py:

  1. delete preprocessed_chunks/ablation_doc_slice_radius_dynamic.json (the only JSON we may delete)
  2. execute RnD/ablation_dynamic_doc_slice_generation.ipynb in memory (all cells; overwrites the dynamic LanceDB table)
  3. execute RnD/benchmarking_retrievals.ipynb in memory up to and including the cell after '# Averaging metrics'
  4. store the printed metrics + the generation summary under real_runs/<tag>.json (scratchpad)

Neither notebook file is modified.  Usage: python run_pipeline.py <tag>
"""
import json
import os
import re
import shutil
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = "/home/martin/projects/TDK/Document_Slice_Contextual_Retrieval"
RND = os.path.join(ROOT, "RnD")
S = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(S, "real_runs")
DYN_JSON = os.path.join(RND, "preprocessed_chunks/ablation_doc_slice_radius_dynamic.json")


def outputs_text(cell):
    return "".join("".join(o.get("text", "")) if o["output_type"] == "stream" else "".join(o.get("data", {}).get("text/plain", ""))
                   for o in cell.get("outputs", []))


def main(tag):
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    if os.path.exists(DYN_JSON):
        os.remove(DYN_JSON)
        print("deleted", DYN_JSON, flush=True)

    # ---- 1) generation notebook -------------------------------------------------------------------------------------
    gen = nbformat.read(os.path.join(RND, "ablation_dynamic_doc_slice_generation.ipynb"), as_version=4)
    NotebookClient(gen, timeout=3600, kernel_name="python3", resources={"metadata": {"path": RND}}).execute()
    gen_out = outputs_text(gen.cells[1])
    summary = [l.strip() for l in re.split(r"[\r\n]", gen_out) if l.strip().startswith(("Processed", "Suggested", "Summariser", "codecarbon"))]
    print("\n".join(summary), flush=True)
    t_gen = time.time() - t0
    shutil.copy(DYN_JSON, os.path.join(OUT_DIR, f"{tag}_chunks.json"))   # keep this run's summaries for later analysis

    # ---- 2) benchmark notebook, cells up to and including the one after '# Averaging metrics' --------------------------
    bench = nbformat.read(os.path.join(RND, "benchmarking_retrievals.ipynb"), as_version=4)
    stop = next(i for i, c in enumerate(bench.cells) if c.cell_type == "markdown" and "Averaging metrics" in c.source) + 1
    bench.cells = bench.cells[: stop + 1]
    # tqdm.notebook needs ipywidgets, which this kernel does not have -> console tqdm in the in-memory copy only
    bench.cells[0].source = bench.cells[0].source.replace("from tqdm.notebook import tqdm", "from tqdm import tqdm")
    t1 = time.time()
    NotebookClient(bench, timeout=3600, kernel_name="python3", resources={"metadata": {"path": RND}}).execute()
    metrics_text = outputs_text(bench.cells[-1])
    print(metrics_text, flush=True)
    t_bench = time.time() - t1

    m = {}
    for block in re.findall(r"METRICS AT N=(\d+)\s*\|.*?\n-+\n(.*?)(?=\n-+\n|\Z)", metrics_text, flags=re.S):
        n, body = block
        vals = {}
        for line in body.strip().splitlines():
            name, own, base = [x.strip() for x in line.split("|")]
            vals[name] = {"own": float(own), "baseline": float(base)}
        m[int(n)] = vals
    rec = {"tag": tag, "generation_summary": summary, "metrics": m, "gen_minutes": round(t_gen / 60, 1), "bench_minutes": round(t_bench / 60, 1),
           "module_sha": os.popen(f"sha256sum {RND}/utils/dynamic_slice_prediction.py").read().split()[0]}
    json.dump(rec, open(os.path.join(OUT_DIR, f"{tag}.json"), "w"), indent=1)
    print(f"\n[{tag}] R@20 own {m[20]['RECALL']['own']} vs baseline {m[20]['RECALL']['baseline']} | MRR {m[20]['MRR']['own']} | MAR {m[20]['MAR']['own']} "
          f"| gen {rec['gen_minutes']} min, bench {rec['bench_minutes']} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
