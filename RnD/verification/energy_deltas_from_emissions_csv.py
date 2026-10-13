"""Verifies every energy, emission, duration and power figure of the Springer manuscript that comes from
RnD/emissions_data/emissions.csv:  Table 5 (tab:power), Table 6 (energy column), Table 7 (tab:primary: tracked
generation time, energy, GPU energy, emissions and their deltas), Table 8 (tab:policy), Figs 7 and 9, the
abstract and Sect. 4.2 / 5.1 percentages, the grid carbon intensity (emissions/energy per row).
Rows are identified by their codecarbon end timestamp; the 3.3.1 rows are re-aligned by position (see _common).
Run from the repository root:  .venv/bin/python RnD/verification/energy_deltas_from_emissions_csv.py"""
from _common import emissions_rows

rows = emissions_rows()
RUNS = {
    "full-document baseline (scientific)": "2026-03-05T23:59:43",
    "static k=3 (scientific, March generation)": "2026-03-05T22:53:27",
    "routed pipeline, primary (notebook run)": "2026-09-25T20:06:47",
    "routed pipeline, replicate (binary_t2_heldout)": "2026-09-25T16:06:28",
    "constant k=2 through the routed notebook (fixed2)": "2026-09-24T00:17:40",
    "original 4-class router run": "2026-09-23T22:32:37",
    "policy corpus, full-document baseline": "2026-07-19T16:44:18",
    "policy corpus, static k=3": "2026-07-19T15:01:49",
    "routed pipeline, end to end from PDFs (parse+route+summarise)": "2026-10-05T19:21:45",
}
print(f"{'run':52s} {'end time':19s} {'s':>8s} {'Wh':>7s} {'cpu':>6s} {'gpu':>6s} {'ram':>6s} {'gCO2':>6s} {'CI':>7s} {'cpuW':>6s} {'gpuW':>6s} {'gpu%':>5s} {'ramGB':>6s} ver")
for name, ts in RUNS.items():
    r = rows[ts]
    print(f"{name:52s} {ts:19s} {r['duration']:8.1f} {r['energy_consumed']*1e3:7.2f} {r['cpu_energy']*1e3:6.2f} {r['gpu_energy']*1e3:6.2f} {r['ram_energy']*1e3:6.2f} {r['emissions']*1e3:6.2f} {r['emissions']/r['energy_consumed']*1e3:7.2f} {r['cpu_power']:6.1f} {r['gpu_power']:6.1f} {r['gpu_utilization_percent']:5.1f} {r['ram_used_gb']:6.1f} {r['codecarbon_version']}")

def delta(a, b, key):
    return 1 - rows[b][key] / rows[a][key]
pairs = [("baseline -> static k=3", RUNS["full-document baseline (scientific)"], RUNS["static k=3 (scientific, March generation)"]),
         ("baseline -> routed (primary)", RUNS["full-document baseline (scientific)"], RUNS["routed pipeline, primary (notebook run)"]),
         ("static k=3 -> routed (primary)", RUNS["static k=3 (scientific, March generation)"], RUNS["routed pipeline, primary (notebook run)"]),
         ("baseline -> routed (replicate)", RUNS["full-document baseline (scientific)"], RUNS["routed pipeline, replicate (binary_t2_heldout)"]),
         ("policy baseline -> policy k=3", RUNS["policy corpus, full-document baseline"], RUNS["policy corpus, static k=3"])]
print()
for name, a, b in pairs:
    print(f"{name:32s} time {delta(a,b,'duration'):+.1%} (x{rows[a]['duration']/rows[b]['duration']:.2f})  energy {delta(a,b,'energy_consumed'):+.1%} (x{rows[a]['energy_consumed']/rows[b]['energy_consumed']:.2f})  CO2 {delta(a,b,'emissions'):+.1%}  GPU energy {delta(a,b,'gpu_energy'):+.1%}  CPU energy {delta(a,b,'cpu_energy'):+.1%}")
b = rows[RUNS["full-document baseline (scientific)"]]; d = rows[RUNS["routed pipeline, primary (notebook run)"]]
print(f"\nabstract / Sect. 4.2 / conclusion: {b['duration']/60:.1f} -> {d['duration']/60:.1f} min; {b['energy_consumed']*1e3:.1f} -> {d['energy_consumed']*1e3:.1f} Wh; {b['emissions']*1e3:.2f} -> {d['emissions']*1e3:.2f} g; routed/baseline energy = {d['energy_consumed']/b['energy_consumed']:.3f}, time = {d['duration']/b['duration']:.3f}")
print("end-to-end (notebook comments, not tracked): 71m34s = 4294 s, 29m09s = 1749 s, ratio", round(4294/1749, 3))
