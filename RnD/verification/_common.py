"""Shared helpers for the verification scripts (read-only access to the frozen artefacts under RnD/)."""
from pathlib import Path
import csv, glob, json

RND = Path(__file__).resolve().parents[1]          # .../RnD
SPLIT = RND / "split_documents"
SPLIT_POLICY = RND / "split_documents" / "smoking"
SPLIT_ROUTER = RND / "split_documents_router"
RESULTS = RND / "dynamic_slice_length" / "router_experiments" / "results"
REAL_RUNS = RND / "dynamic_slice_length" / "router_experiments" / "real_runs"
EMISSIONS = RND / "emissions_data" / "emissions.csv"
QNA_SCI = RND / "q_and_a" / "Gemini" / "scientific_multi_chunk_control.json"
QNA_POLICY = RND / "q_and_a" / "Gemini" / "smoking.json"
DYNAMIC_JSON = RND / "preprocessed_chunks" / "ablation_doc_slice_radius_dynamic.json"
GOLD = RND / "dynamic_slice_length" / "slice_radius_gold.json"
EXEMPLARS = RND / "dynamic_slice_length" / "router_exemplars_heldout.json"   # the ten router exemplar passages; the six held-out papers themselves are not in the public copy


def load_docs(split_dir):
    """{file name: [chunk records]} in sorted file order (the order the notebooks use)."""
    return {Path(p).name: json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(str(split_dir / "*.json")))}


def emissions_rows():
    """Rows of emissions.csv keyed by timestamp. Values are read BY POSITION for the columns up to
    energy_consumed (identical in codecarbon 3.2.3 and 3.3.1); the later columns are re-aligned for the 3.3.1
    rows, which insert cloud_provider/cloud_region after region and therefore sit two columns to the right of
    the 3.2.3 header of the file."""
    rows = list(csv.reader(open(EMISSIONS, encoding="utf-8")))
    hdr = rows[0]
    out = {}
    for r in rows[1:]:
        d = dict(zip(hdr, r))
        version = r[hdr.index("codecarbon_version") + (2 if r[hdr.index("region") + 1] == "" and len(r) > len(hdr) - 1 and r[hdr.index("cloud_provider")] not in ("", "N") else 0)]
        # simpler and robust: detect the 3.3.1 layout by the position of the version string
        v_idx = next(i for i, x in enumerate(r) if x in ("3.2.3", "3.3.1"))
        shift = v_idx - hdr.index("codecarbon_version")
        for name in ("codecarbon_version", "cpu_count", "cpu_model", "gpu_count", "gpu_model", "ram_total_size",
                     "tracking_mode", "cpu_utilization_percent", "gpu_utilization_percent",
                     "ram_utilization_percent", "ram_used_gb", "on_cloud", "pue"):
            d[name] = r[hdr.index(name) + shift]
        for name in ("duration", "emissions", "energy_consumed", "cpu_energy", "gpu_energy", "ram_energy",
                     "cpu_power", "gpu_power", "ram_power", "cpu_utilization_percent", "gpu_utilization_percent",
                     "ram_used_gb"):
            d[name] = float(d[name])
        out[d["timestamp"]] = d
    return out


def nomic_tokenizer():
    """The tokenizer the HybridChunker and cell 2 of doc_slice_chunking.ipynb used (loaded from the local HF cache)."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", local_files_only=True)
