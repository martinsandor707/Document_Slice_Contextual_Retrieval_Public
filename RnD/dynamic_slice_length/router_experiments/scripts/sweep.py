"""Policy sweep on the offline simulator.  Usage: python sweep.py <sweep-name> [...] | python sweep.py --report

Policies are dicts id -> k (0..3) built from the CURRENT router assignment (dynamic JSON), from saved router
variant runs (routes_<variant>.json) or from deterministic chunk features.  Results are appended to
sweep_results.jsonl and per-question data saved to res_<name>.json.
"""
import glob
import json
import os
import sys
import time
from collections import Counter

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
import evalkit as ek

qna = ek.qna()
# The ORIGINAL router assignment (the user's dynamic run) is frozen in routing_original.json; the live dynamic JSON is
# overwritten by every real run and must not be used as the base.
_routing = json.load(open(os.path.join(S, "routing_original.json")))
k_dyn = {cid: v["k"] for cid, v in _routing.items()}
tier = {cid: v["tier"] for cid, v in _routing.items()}
text_of = {cid: r["original_text"] for cid, r in ek.fixed_cached(2).items()}
ids = list(k_dyn)
LOG = os.path.join(S, "sweep_results.jsonl")

# chunk position (index, n) per id from the split documents
pos = {}
for p in sorted(glob.glob(f"{ek.RND}/split_documents/*.json")):
    d = json.load(open(p))
    for i, c in enumerate(d):
        pos.setdefault(c["id"], (i, len(d)))


# ---- summaries produced by REAL runs of this pipeline (module-default router with tables->1, and constant k=2) ----
def _real(tag):
    p = os.path.join(S, "real_runs", f"{tag}_chunks.json")
    return {r["id"]: r for r in json.load(open(p))} if os.path.exists(p) else {}


REAL = {tag: _real(tag) for tag in ["tables1", "fixed2", "binary_t2"]}
SOURCES = [("real_tables1", REAL["tables1"]), ("real_fixed2", REAL["fixed2"])]
SOURCES_BIN = [("real_binary_t2", REAL["binary_t2"]), ("real_fixed2", REAL["fixed2"])]   # base = the 2-class router's own run
k_final = {cid: int(r["doc_slice_radius"]) for cid, r in REAL["tables1"].items()}     # the final config's own routing
tier_final = {cid: r["slice_tier"] for cid, r in REAL["tables1"].items()}


def remap_final(mapping):
    """mapping: k -> new k for non-table chunks, 'T' -> new k for gated tables (which otherwise keep their k)."""
    out = {}
    for cid, k in k_final.items():
        if tier_final[cid] == "table":
            out[cid] = mapping.get("T", k)
        else:
            out[cid] = mapping.get(k, k)
    return out


def route_policy_final(variant, mapping, table_k=1):
    """letters of a saved router run -> k, gated tables -> table_k (base = the same 380 ids as the final config)."""
    routes = json.load(open(f"{S}/routes_{variant}.json"))
    return {cid: (table_k if r["letter"] == "T" else mapping[r["letter"]]) for cid, r in routes.items() if cid in k_final}


def run(name, policy, k0_mode="skip", note="", sources=None):
    assert max(policy.values()) <= 3, "k above 3 is not allowed"
    t0 = time.time()
    res = ek.simulate_sources(name, policy, sources, k0_mode) if sources is not None else ek.simulate(name, policy, k0_mode)
    ek.save(res, f"{S}/res_{name}.json")
    a = res["avg"]
    row = {"name": name, "note": note, "k_dist": res["k_dist"], "k0_mode": k0_mode, "sources_used": res.get("sources_used"),
           "R5": a[5]["recall"], "R10": a[10]["recall"], "R15": a[15]["recall"], "R20": a[20]["recall"], "MRR20": a[20]["mrr"], "MAR20": a[20]["mar"],
           "secs": round(time.time() - t0)}
    with open(LOG, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{name:34s} R@20 {a[20]['recall']:.4f}  MRR {a[20]['mrr']:.5f}  MAR {a[20]['mar']:.5f}  R@5 {a[5]['recall']:.4f} R@10 {a[10]['recall']:.4f} | k {res['k_dist']} | src {res.get('sources_used')} | {row['secs']}s", flush=True)
    return res


def remap(mapping, base=None):
    """mapping: routed k -> new k (tables keep their own entry under key 'T')."""
    base = base or k_dyn
    out = {}
    for cid, k in base.items():
        key = "T" if tier[cid] == "table" else k
        out[cid] = mapping.get(key, mapping.get(k, k))
    return out


def route_policy(variant, mapping, table_k=3):
    """mapping: letter -> k for a saved router run; gated tables -> table_k."""
    routes = json.load(open(f"{S}/routes_{variant}.json"))
    return {cid: (table_k if r["letter"] == "T" else mapping[r["letter"]]) for cid, r in routes.items() if cid in k_dyn}


def length_policy(short=700, long_=2500, k_short=3, k_mid=2, k_long=1, table_k=3):
    out = {}
    for cid in ids:
        if tier[cid] == "table":
            out[cid] = table_k
        else:
            L = len(text_of[cid])
            out[cid] = k_short if L < short else (k_mid if L <= long_ else k_long)
    return out


def position_policy(edge=1, k_edge=1, k_mid=2, table_k=3):
    out = {}
    for cid in ids:
        i, n = pos[cid]
        out[cid] = table_k if tier[cid] == "table" else (k_edge if (i < edge or i >= n - edge) else k_mid)
    return out


# policies are built lazily (lambda) so that a missing routes file only breaks the sweep that needs it
SWEEPS = {
    "proxy": [
        ("sim_dyn_skip", lambda: remap({}), "skip", "current routing, contexts from fixed-k JSONs, k=0 skipped (proxy of the real dynamic run)"),
        ("sim_all2", lambda: {i: 2 for i in ids}, "skip", "fixed k=2 rebuilt from JSON (should match table ablation_doc_slice_radius_2)"),
    ],
    "k0": [
        ("sim_dyn_k0summary", lambda: remap({}), "summary", "current routing, k=0 chunks get the fixed k=0 self-summary instead of nothing"),
        ("sim_A_to_1", lambda: remap({0: 1}), "skip", "A -> k=1"),
        ("sim_A_to_2", lambda: remap({0: 2}), "skip", "A -> k=2"),
    ],
    "classes": [
        ("sim_B_to_2", lambda: remap({0: 2, 1: 2}), "skip", "A,B -> 2 (only C=2, D=3, T=3 differ from all-2)"),
        ("sim_B_to_3", lambda: remap({0: 2, 1: 3}), "skip", "A -> 2, B -> 3"),
        ("sim_C_to_3", lambda: remap({0: 2, 2: 3}), "skip", "A -> 2, C -> 3"),
        ("sim_C_to_1", lambda: remap({0: 2, 2: 1}), "skip", "A -> 2, C -> 1"),
        ("sim_D_to_2", lambda: remap({0: 2, 3: 2}), "skip", "A -> 2, D(slm) -> 2"),
        ("sim_T_to_2", lambda: remap({0: 2, "T": 2}), "skip", "A -> 2, tables -> 2"),
        ("sim_all3", lambda: {i: 3 for i in ids}, "skip", "fixed k=3 rebuilt from JSON"),
        ("sim_all1", lambda: {i: 1 for i in ids}, "skip", "fixed k=1 rebuilt from JSON"),
    ],
    "tables": [
        ("sim_T_to_1", lambda: remap({0: 2, "T": 1}), "skip", "A -> 2, tables -> 1"),
        ("sim_dyn_T2", lambda: remap({"T": 2}), "skip", "current routing but tables -> 2"),
        ("sim_dyn_T1", lambda: remap({"T": 1}), "skip", "current routing but tables -> 1"),
    ],
    "anchor": [
        ("sim_anchor8_A1B2C3", lambda: route_policy("anchor8", {"A": 1, "B": 2, "C": 3}), "skip", "anchor few-shot: A->1 B->2 C->3, T->3"),
        ("sim_anchor8_A0B2C3", lambda: route_policy("anchor8", {"A": 0, "B": 2, "C": 3}), "skip", "anchor few-shot: A skipped, B->2 C->3, T->3"),
        ("sim_anchor8_A1B3C3", lambda: route_policy("anchor8", {"A": 1, "B": 3, "C": 3}), "skip", "anchor few-shot: A->1 B->3 C->3, T->3"),
        ("sim_anchor8_A2B3C3", lambda: route_policy("anchor8", {"A": 2, "B": 3, "C": 3}), "skip", "anchor few-shot: A->2 B->3 C->3, T->3"),
        ("sim_anchor8_A1B2C3_T2", lambda: route_policy("anchor8", {"A": 1, "B": 2, "C": 3}, table_k=2), "skip", "anchor few-shot: A->1 B->2 C->3, T->2"),
        ("sim_anchor0_A1B2C3", lambda: route_policy("anchor0", {"A": 1, "B": 2, "C": 3}), "skip", "anchor zero-shot: A->1 B->2 C->3, T->3"),
    ],
    # prioritised subset (about 35 min) - the most informative policies first
    "prio": [
        ("sim_A_to_2", lambda: remap({0: 2}), "skip", "A -> k=2 (never skip)"),
        ("sim_dyn_T2", lambda: remap({"T": 2}), "skip", "current routing but tables -> 2"),
        ("sim_dyn_T1", lambda: remap({"T": 1}), "skip", "current routing but tables -> 1"),
        ("sim_C_to_3", lambda: remap({0: 2, 2: 3}), "skip", "A -> 2, C -> 3"),
        ("sim_B_to_3", lambda: remap({0: 2, 1: 3}), "skip", "A -> 2, B -> 3"),
        ("sim_C_to_1", lambda: remap({0: 2, 2: 1}), "skip", "A -> 2, C -> 1"),
        ("sim_dyn_k0summary", lambda: remap({}), "summary", "current routing, k=0 chunks get the fixed k=0 self-summary instead of nothing"),
        ("sim_anchor8_A1B2C3", lambda: route_policy("anchor8", {"A": 1, "B": 2, "C": 3}), "skip", "anchor few-shot: A->1 B->2 C->3, T->3"),
        ("sim_anchor0_A1B2C3", lambda: route_policy("anchor0", {"A": 1, "B": 2, "C": 3}), "skip", "anchor zero-shot: A->1 B->2 C->3, T->3"),
        ("sim_len_s3_m2_l2", lambda: length_policy(k_long=2), "skip", "length only: <700->3, else 2, T->3"),
        ("sim_pos_edge1", lambda: position_policy(), "skip", "position only: first/last chunk ->1, else 2, T->3"),
    ],
    # round 2 (after tables->1 came out best): combinations around it, the untested class moves, the heuristics
    "round2": [
        ("sim_A_to_1", lambda: remap({0: 1}), "skip", "A -> k=1 (never skip, cheap)"),
        ("sim_T_to_1", lambda: remap({0: 2, "T": 1}), "skip", "A -> 2, tables -> 1"),
        ("sim_dyn_T1_D2", lambda: remap({"T": 1, 3: 2}), "skip", "tables -> 1, fragments D -> 2"),
        ("sim_dyn_T1_A1", lambda: remap({"T": 1, 0: 1}), "skip", "tables -> 1, A -> 1 (never skip)"),
        ("sim_dyn_T1_B2", lambda: remap({"T": 1, 1: 2}), "skip", "tables -> 1, B -> 2"),
        ("sim_dyn_T2_D2", lambda: remap({"T": 2, 3: 2}), "skip", "tables -> 2, fragments D -> 2"),
        ("sim_B_to_2", lambda: remap({0: 2, 1: 2}), "skip", "A,B -> 2 (only C=2, D=3, T=3 differ from all-2)"),
        ("sim_D_to_2", lambda: remap({0: 2, 3: 2}), "skip", "A -> 2, D(slm) -> 2"),
        ("sim_T_to_2", lambda: remap({0: 2, "T": 2}), "skip", "A -> 2, tables -> 2"),
        ("sim_anchor0_A1B2C3", lambda: route_policy("anchor0", {"A": 1, "B": 2, "C": 3}), "skip", "anchor zero-shot: A->1 B->2 C->3, T->3"),
        ("sim_len_s3_m2_l2", lambda: length_policy(k_long=2), "skip", "length only: <700->3, else 2, T->3"),
        ("sim_len_s3_m2_l1", lambda: length_policy(), "skip", "length only: <700 chars->3, 700-2500->2, >2500->1, T->3"),
        ("sim_len_s3_m3_l2", lambda: length_policy(k_mid=3, k_long=2), "skip", "length only: <=2500->3, >2500->2, T->3"),
        ("sim_pos_edge1", lambda: position_policy(), "skip", "position only: first/last chunk ->1, else 2, T->3"),
        ("sim_pos_edge1_skip", lambda: position_policy(k_edge=0), "skip", "position only: first/last chunk skipped, else 2, T->3"),
    ],
    # ---- round 3: start from the FINAL config's own real summaries (tables->1 run + constant-k=2 run) --------------
    # mapping-only modifications (no new prompt)
    "final_map": [
        ("fin_base", lambda: remap_final({}), "skip", "final config rebuilt from its own real summaries (must reproduce 0.9420)"),
        ("fin_D2", lambda: remap_final({3: 2}), "skip", "fragments D -> 2 (no k=3 anywhere)"),
        ("fin_D1", lambda: remap_final({3: 1}), "skip", "fragments D -> 1"),
        ("fin_A2", lambda: remap_final({0: 2}), "skip", "A -> 2 (never skip; k=2 summaries from the constant-k=2 run)"),
        ("fin_A1", lambda: remap_final({0: 1}), "skip", "A -> 1 (paper's k=1 summaries)"),
        ("fin_A3", lambda: remap_final({0: 3}), "skip", "A -> 3 (paper's k=3 summaries)"),
        ("fin_A0sum", lambda: remap_final({}), "summary", "A gets the paper's k=0 self-summary instead of nothing"),
        ("fin_B2", lambda: remap_final({1: 2}), "skip", "B -> 2 (k=2 summaries from the constant-k=2 run)"),
        ("fin_B2_D2", lambda: remap_final({1: 2, 3: 2}), "skip", "A skip, everything else 2, tables 1  (= what a 2-class router would do)"),
        ("fin_T2", lambda: remap_final({"T": 2}), "skip", "tables -> 2 (k=2 summaries from the constant-k=2 run)"),
    ],
    # new prompts (letters from routes_<variant>.json produced by router_variants.py), tables -> 1
    "final_prompts": [
        ("fin_genre3_A0B1C2", lambda: route_policy_final("genre3", {"A": 0, "B": 1, "C": 2}), "skip", "3-class genre prompt (no fragment class): A skip, B->1, C->2, T->1"),
        ("fin_genre3_A0B2C2", lambda: route_policy_final("genre3", {"A": 0, "B": 2, "C": 2}), "skip", "3-class genre prompt: A skip, else 2, T->1"),
        ("fin_binary_S0N2", lambda: route_policy_final("binary", {"S": 0, "N": 2}), "skip", "2-class prompt (self-describing vs needs context): S skip, N->2, T->1"),
        ("fin_binary_S0N1", lambda: route_policy_final("binary", {"S": 0, "N": 1}), "skip", "2-class prompt: S skip, N->1, T->1"),
        ("fin_split5_cur", lambda: route_policy_final("split5", {"A": 0, "B": 0, "C": 1, "D": 2, "E": 2}), "skip", "5-class prompt (abstract/conclusion | boilerplate | exposition | body | fragment): both A-types skipped"),
        ("fin_split5_abs1", lambda: route_policy_final("split5", {"A": 1, "B": 0, "C": 1, "D": 2, "E": 2}), "skip", "5-class: abstracts/conclusions -> 1, boilerplate skipped"),
        ("fin_split5_abs2", lambda: route_policy_final("split5", {"A": 2, "B": 0, "C": 1, "D": 2, "E": 2}), "skip", "5-class: abstracts/conclusions -> 2, boilerplate skipped"),
    ],
    # ---- exemplar re-sourcing check: 2-class router with exemplars from the held-out papers, on the binary_t2 run's own
    #      summaries (chunks that change class take k=2 summaries from the constant-k=2 run, or are skipped)
    "heldout": [
        ("bin_own_base", lambda: route_policy_final("binary", {"S": 0, "N": 2}, table_k=2), "skip", "binary_t2 routing rebuilt from its own real summaries (must reproduce 0.9487)"),
        ("bin_heldout", lambda: route_policy_final("binary_heldout", {"S": 0, "N": 2}, table_k=2), "skip", "2-class router with HELD-OUT exemplars, tables->2, on binary_t2's summaries"),
    ],
    "determ": [
        ("sim_len_s3_m2_l1", lambda: length_policy(), "skip", "length only: <700 chars->3, 700-2500->2, >2500->1, T->3"),
        ("sim_len_s3_m2_l2", lambda: length_policy(k_long=2), "skip", "length only: <700->3, else 2, T->3"),
        ("sim_len_s3_m3_l2", lambda: length_policy(k_mid=3, k_long=2), "skip", "length only: <=2500->3, >2500->2, T->3"),
        ("sim_pos_edge1", lambda: position_policy(), "skip", "position only: first/last chunk ->1, else 2, T->3"),
        ("sim_pos_edge1_skip", lambda: position_policy(k_edge=0), "skip", "position only: first/last chunk skipped, else 2, T->3"),
    ],
}


def report():
    rows = [json.loads(l) for l in open(LOG)]
    rows.sort(key=lambda r: -r["R20"])
    print(f"{'policy':34s} {'R@20':>7s} {'MRR@20':>8s} {'MAR@20':>8s} {'R@5':>7s} {'R@10':>7s}  k distribution / note")
    for r in rows:
        print(f"{r['name']:34s} {r['R20']:7.4f} {r['MRR20']:8.5f} {r['MAR20']:8.5f} {r['R5']:7.4f} {r['R10']:7.4f}  {r['k_dist']} | {r['note'][:70]}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        for sw in sys.argv[1:]:
            for name, make_policy, k0m, note in SWEEPS[sw]:
                if os.path.exists(f"{S}/res_{name}.json"):
                    print("skip (done):", name, flush=True)
                    continue
                src = SOURCES_BIN if sw == "heldout" else (SOURCES if sw.startswith("final") else None)
                run(name, make_policy(), k0m, note, sources=src)
