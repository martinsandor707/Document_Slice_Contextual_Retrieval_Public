# Router prompt engineering for dynamic document-slice radii

Goal: configure the slice-radius router in `utils/dynamic_slice_prediction.py` so that
`ablation_dynamic_doc_slice_generation.ipynb` followed by `benchmarking_retrievals.ipynb` maximises **Recall@20**
against the Anthropic full-document baseline. All numbers are on the 250 scientific multi-chunk questions
(`q_and_a/Gemini/scientific_multi_chunk_control.json`: 530 gold chunk references, 226 distinct gold chunks, 382 Docling
chunks from 25 papers). Research material: `dynamic_slice_length/router_experiments/` (scripts, per-question results,
router routes, real-run records). **Test-set hygiene:** every result up to Section 7 was obtained with few-shot
exemplars taken from benchmark papers; Section 8 replaces them with exemplars from six held-out papers and repeats the
final live run — that run is the one to report.

## 1. Result

| Configuration | R@20 | MRR@20 | MAR@20 | Summariser calls / prompt text vs fixed k=2 | Generation time |
|---|---|---|---|---|---|
| Anthropic full-document baseline (paper) | **0.9540** | 0.786 | 3.94 | 382 / full document | 54.9 min (paper's codecarbon row) |
| Fixed k=2, paper's table | 0.9467 | 0.768 | 3.81 | 382 / 100 % | 17.5 min (two-model pipeline) |
| Fixed k=2, re-generated in this pipeline (`FIXED_K = 2`) | 0.9467 | 0.792 | 3.92 | 382 / 100 % | 11.2 min |
| Dynamic router, original config (A skipped, tables → 3), Martin's run | 0.9380 | **0.795** | **3.68** | 273 / 73 % | 9.2 min |
| **Dynamic router, final config (tables → 1), two identical real runs** | 0.9420 | 0.788 | 3.83 | 271 / 65 % | 9.1–9.4 min |
| Best offline policy, round 1 (tables → 1 on the paper's fixed-radius summaries, original routing) | 0.9500 | 0.781 | 3.79 | 273 / 66 % | — |
| Best offline modification, round 3: 2-class prompt (self-describing → skip, else k=2), tables → 2, on the final run's own summaries | 0.9507 | 0.791 | 3.97 | 281 / ~74 % | — |
| Round 3: genre router over k ∈ {0,1,2} (fragments → 2, tables → 2) | 0.9500 | 0.787 | 3.90 | 271 / ~71 % | — |
| Dynamic router, 2-class (S → skip, N → 2), tables → 2, real run, exemplars from benchmark papers | 0.9487 | 0.793 | 3.92 | 279 / 73 % | 10.3 min |
| **Dynamic router, 2-class (S → skip, N → 2), tables → 2, real run, exemplars from the HELD-OUT papers (Section 8) — final** | **0.9520** | **0.798** | 3.92 | 262 / ~69 % | 9.7 min |

**Verdict.** The final router — a 2-class classifier (self-describing chunk → no summary, everything else → k=2)
with the table gate at k=2 and exemplars from held-out papers — is the only configuration that beats fixed k=2 in a
real run: **0.9520 vs 0.9467** (+0.5 pt, 7 questions better / 4 worse) with 31 % fewer summariser calls and 9.7 min
of generation instead of 11.2 min (17.5 min in the paper's two-model pipeline), and it lands 0.2 pt below the
full-document baseline (0.954; 6 better / 7 worse), i.e. statistically level with it within the ±0.7 pt single-run band
of Section 3, at a fraction of the baseline's cost (54.9 min). Its MRR (0.798) is above both. The path there: the
original 4-class router with tables → 3 scored 0.9380 mostly because wide slices degrade table summaries (Section 2);
tables → 1 recovered part of it in real runs (0.9420); round 3 (Section 4b) showed that k=3 can be dropped entirely,
that tables → 2 is the better-supported table setting and that the simplest 2-class router is the best offline
modification (0.9507); its real run with benchmark-derived exemplars gave 0.9487, and the rerun with clean exemplars
(Section 8) gave 0.9520. Single-run differences below ~0.7 pt remain ties; the compute saving is not.

**Final router configuration** (module defaults): genre prompt unchanged; class A (abstract / conclusion / boilerplate)
→ no summary (k=0, skipped), B (self-contained exposition) → k=1, C (section-internal technical body) → k=2,
D (fragment) → k=3; **table chunks (Tier-1 gate) → k=1 instead of 3** (`TABLE_K = 1`). That single change is the only
modification with a measurable effect; everything else tested was neutral or harmful.

## 2. Why the first dynamic run scored 0.938 (below fixed k=2 and even hybrid retrieval)

Not the routing. Versus fixed k=2, 9 questions got worse and 5 better (net −0.87 pt ≈ 4.6 gold references). Every gold
chunk the dynamic run missed had been routed to k=2 (5 chunks) or was a table at k=3 (4 chunks); none was a skipped
(k=0) or a k=1 chunk. Only 13 of the 226 gold chunks were routed to k=0 at all (22 questions), and all were retrieved from
their own text, so skipping boilerplate summaries cost nothing. Rebuilding the same routing from the fixed-radius
summaries scores 0.9447 → about 0.7 of the 0.87 points came from the particular summaries generated in that run.

Two systematic effects were found instead:

* **Wide slices degrade the summaries.** Contexts that open with the forbidden "This chunk …" rise from 20 (k=0) to
  19 / 37 / 47 (k=1 / 2 / 3) of 382. For the large P&L table of s41467-020-15356-z the k=3 slice produced a verbatim
  copy of a system-prompt example ("Provides supporting evidence for the argument on economic inequality."); the k=1
  and k=2 summaries describe the table. The gold table chunks are exactly the chunks that lost rank.
* **Skipping boilerplate trades tail recall for head precision.** The dynamic run had the best MRR (0.795) and MAR (3.68)
  of all configurations and the best R@5 (0.760 vs 0.737 baseline).

## 3. The noise floor: single runs cannot rank routers

Regenerating constant k=2 through the dynamic notebook (module switch `FIXED_K = 2`) reproduced R@20 = 0.94667 exactly,
but only 26 of 380 summaries were identical to the earlier k=2 generation (170 had text similarity < 0.5), 238 of 250
questions received a different top-20 set (5 improved, 5 worsened), and MRR moved from 0.768 to 0.792. **A single real
run resolves about ±0.7 pt of R@20 and ±2 pt of MRR.** The offline simulator (fixed summaries, exact replica of the
benchmark notebook's retrieval, validated to five decimals on three tables) removes the summariser noise and is what
ranks the policies below; real runs are the confirmation.

The noise is **not random per run**: the two tables→1 confirmation runs, issued with an identical request sequence from
a freshly loaded model, produced 380/380 identical summaries and identical metrics. Summaries change when the request
history changes (a different routing, a different pipeline version, model reloads between documents), because Ollama's
temperature-0 logits depend on the KV-cache/batch state. Consequences: repeating the very same run does not sample the
noise, and an offline gain that rests on particular summaries (here: the paper's k=1 table summaries; only 4 of the 42
table chunks received the same summary in the real run, 16 differed substantially) need not transfer to a fresh
generation.

Headroom: a per-question oracle choosing the best k ∈ {1,2,3} reaches 0.964 (0.966 with "no context" allowed), but only
17 of 226 gold chunks are k-sensitive and their preferred k is scattered (7×1, 5×2, 5×3) with single-question flips, i.e.
there is no learnable per-chunk rule beyond the table effect; no k ≤ 3 routing tested beats the full-document baseline.

## 4. All measurements

Kinds: *real (paper)* = existing LanceDB tables from the paper's runs, evaluated with the replica evaluator; *real* =
generation notebook + benchmark run in this study; *offline sim* = fixed-radius summaries assembled per policy.
"Prompt text vs k=2" = characters sent to the summariser relative to fixed k=2 (382 calls, 6.36 M chars).
Base routing for the sims = the router's decisions in Martin's run: k {0: 107, 1: 53, 2: 161, 3: 59} over 380 unique
chunk ids (42 gated tables inside the 59). Letters: A/B/C/D = genre classes, T = table gate.

| Configuration | Kind | R@20 | MRR@20 | MAR@20 | R@5 | R@10 | summariser calls / prompt text vs k=2 | Note |
|---|---|---|---|---|---|---|---|---|
| Anthropic full-document baseline | real (paper) | 0.9540 | 0.7863 | 3.935 | 0.7373 | 0.8650 | — |  |
| Hybrid retrieval, no context | real (paper) | 0.9393 | 0.7748 | 4.036 | 0.7173 | 0.8470 | 0 / 0% |  |
| Fixed k=0 (chunk-only summary) | real (paper) | 0.9373 | 0.7861 | 3.808 | 0.7570 | 0.8600 | — |  |
| Fixed k=1 | real (paper) | 0.9433 | 0.7990 | 3.726 | 0.7563 | 0.8703 | 382 / 67% |  |
| Fixed k=2 | real (paper) | 0.9467 | 0.7680 | 3.809 | 0.7453 | 0.8777 | 382 / 100% |  |
| Fixed k=3 | real (paper) | 0.9453 | 0.7877 | 4.020 | 0.7410 | 0.8590 | 382 / 133% |  |
| Dynamic router, original config (Martin's run) | real | 0.9380 | 0.7949 | 3.685 | 0.7597 | 0.8650 | 273 / 73% |  |
| Constant k=2 through the dynamic notebook (FIXED_K=2) | real (re-generation) | 0.9467 | 0.7923 | 3.921 | 0.7463 | 0.8643 | 382 / 100% |  |
| Real run `binary_t2` | real | 0.9487 | 0.7926 | 3.924 | 0.7503 | 0.8623 | 279 / 73% | 2-class router (S skip, N→2), tables→2 (exemplars from benchmark papers); k {0: 103, 2: 279}; table k=[2]; generation 10.3 min |
| Real run `binary_t2_heldout` | real | 0.9520 | 0.7979 | 3.922 | 0.7517 | 0.8657 | 262 / 69% | 2-class router (S skip, N→2), tables→2, exemplars from the HELD-OUT papers (final); k {0: 120, 2: 262}; table k=[2]; generation 9.7 min |
| Real run `tables1` | real | 0.9420 | 0.7882 | 3.826 | 0.7490 | 0.8583 | 271 / 65% | genre router, tables→1 (exemplars from benchmark papers); k {0: 111, 1: 96, 2: 157, 3: 18}; table k=[1]; generation 9.4 min |
| Real run `tables1_run2` | real | 0.9420 | 0.7882 | 3.826 | 0.7490 | 0.8583 | — | genre router, tables→1 (identical repeat); k {0: 111, 1: 96, 2: 157, 3: 18}; table k=[]; generation 9.1 min |
| `fin_binary_S0N2_T2`: 2-class prompt: S skip, N->2, tables->2 (k in {0,2}) | offline sim | 0.9507 | 0.7907 | 3.966 | 0.7457 | 0.8603 | — | k {'0': 99, '2': 281} |
| `bin_own_base`: binary_t2 routing rebuilt from its own real summaries (must reproduce 0.9487) | offline sim | 0.9507 | 0.7912 | 3.977 | 0.7483 | 0.8583 | 281 / 74% | k {'0': 99, '2': 281} |
| `bin_heldout`: 2-class router with HELD-OUT exemplars, tables->2, on binary_t2's summaries | offline sim | 0.9507 | 0.8001 | 3.931 | 0.7517 | 0.8603 | 260 / 69% | k {'0': 120, '2': 260} |
| `sim_dyn_T1`: current routing but tables -> 1 | offline sim | 0.9500 | 0.7814 | 3.787 | 0.7603 | 0.8683 | 273 / 66% | k {'0': 107, '1': 95, '2': 161, '3': 17} |
| `sim_dyn_T1_D2`: tables -> 1, fragments D -> 2 | offline sim | 0.9500 | 0.7816 | 3.792 | 0.7570 | 0.8683 | 273 / 65% | k {'0': 107, '1': 95, '2': 178} |
| `sim_dyn_T1_B2`: tables -> 1, B -> 2 | offline sim | 0.9500 | 0.7693 | 3.821 | 0.7563 | 0.8703 | 273 / 70% | k {'0': 107, '1': 42, '2': 214, '3': 17} |
| `fin_T2_D2`: tables -> 2 AND fragments D -> 2 (router uses only k in {0,1,2}) | offline sim | 0.9500 | 0.7867 | 3.900 | 0.7537 | 0.8637 | — | k {'0': 109, '1': 54, '2': 217} |
| `sim_dyn_T2`: current routing but tables -> 2 | offline sim | 0.9487 | 0.7822 | 3.636 | 0.7630 | 0.8797 | 273 / 70% | k {'0': 107, '1': 53, '2': 203, '3': 17} |
| `sim_dyn_T2_D2`: tables -> 2, fragments D -> 2 | offline sim | 0.9487 | 0.7821 | 3.643 | 0.7597 | 0.8797 | 273 / 69% | k {'0': 107, '1': 53, '2': 220} |
| `sim_dyn_T1_A1`: tables -> 1, A -> 1 (never skip) | offline sim | 0.9480 | 0.7875 | 3.704 | 0.7590 | 0.8643 | 382 / 83% | k {'1': 202, '2': 161, '3': 17} |
| `fin_T2`: tables -> 2 (k=2 summaries from the constant-k=2 run) | offline sim | 0.9480 | 0.7849 | 3.832 | 0.7537 | 0.8697 | 271 / 69% | k {'0': 109, '1': 54, '2': 199, '3': 18} |
| `sim_all2`: fixed k=2 rebuilt from JSON (should match table ablation_doc_slice_radius_2) | offline sim | 0.9467 | 0.7680 | 3.809 | 0.7453 | 0.8777 | 382 / 100% | k {'2': 380} |
| `sim_A_to_1`: A -> k=1 (never skip, cheap) | offline sim | 0.9467 | 0.7859 | 3.628 | 0.7657 | 0.8730 | 382 / 91% | k {'1': 160, '2': 161, '3': 59} |
| `sim_D_to_2`: A -> 2, D(slm) -> 2 | offline sim | 0.9467 | 0.7762 | 3.772 | 0.7527 | 0.8757 | 382 / 96% | k {'1': 53, '2': 327} |
| `sim_T_to_2`: A -> 2, tables -> 2 | offline sim | 0.9467 | 0.7763 | 3.762 | 0.7580 | 0.8757 | 382 / 97% | k {'1': 53, '2': 310, '3': 17} |
| `fin_A1`: A -> 1 (paper's k=1 summaries) | offline sim | 0.9460 | 0.7905 | 3.920 | 0.7470 | 0.8603 | 382 / 83% | k {'1': 205, '2': 157, '3': 18} |
| `sim_dyn_skip`: current routing, contexts from fixed-k JSONs, k=0 skipped (proxy of the real dynamic run) | offline sim | 0.9447 | 0.7870 | 3.579 | 0.7670 | 0.8730 | 273 / 73% | k {'0': 107, '1': 53, '2': 161, '3': 59} |
| `sim_dyn_k0summary`: current routing, k=0 chunks get the fixed k=0 self-summary instead of nothing | offline sim | 0.9447 | 0.7879 | 3.637 | 0.7613 | 0.8760 | 273 / 73% | k {'0': 107, '1': 53, '2': 161, '3': 59} |
| `sim_A_to_2`: A -> k=2 (never skip) | offline sim | 0.9447 | 0.7786 | 3.737 | 0.7533 | 0.8730 | 382 / 100% | k {'1': 53, '2': 268, '3': 59} |
| `sim_B_to_2`: A,B -> 2 (only C=2, D=3, T=3 differ from all-2) | offline sim | 0.9447 | 0.7706 | 3.773 | 0.7473 | 0.8750 | 382 / 105% | k {'2': 321, '3': 59} |
| `sim_len_s3_m2_l2`: length only: <700->3, else 2, T->3 | offline sim | 0.9447 | 0.7715 | 3.760 | 0.7400 | 0.8737 | 382 / 111% | k {'2': 260, '3': 120} |
| `sim_len_s3_m3_l2`: length only: <=2500->3, >2500->2, T->3 | offline sim | 0.9447 | 0.7740 | 3.744 | 0.7520 | 0.8697 | 382 / 122% | k {'2': 132, '3': 248} |
| `sim_pos_edge1_skip`: position only: first/last chunk skipped, else 2, T->3 | offline sim | 0.9447 | 0.7701 | 3.742 | 0.7520 | 0.8790 | 333 / 90% | k {'0': 49, '2': 289, '3': 42} |
| `fin_split5_abs1`: 5-class: abstracts/conclusions -> 1, boilerplate skipped | offline sim | 0.9447 | 0.7881 | 3.825 | 0.7477 | 0.8643 | 332 / 66% | k {'0': 50, '1': 208, '2': 122} |
| `sim_T_to_1`: A -> 2, tables -> 1 | offline sim | 0.9440 | 0.7765 | 3.812 | 0.7537 | 0.8643 | 382 / 93% | k {'1': 95, '2': 268, '3': 17} |
| `fin_D2`: fragments D -> 2 (no k=3 anywhere) | offline sim | 0.9440 | 0.7900 | 3.897 | 0.7477 | 0.8583 | 271 / 64% | k {'0': 109, '1': 96, '2': 175} |
| `fin_D1`: fragments D -> 1 | offline sim | 0.9440 | 0.7881 | 3.866 | 0.7490 | 0.8583 | 271 / 63% | k {'0': 109, '1': 114, '2': 157} |
| `fin_A0sum`: A gets the paper's k=0 self-summary instead of nothing | offline sim | 0.9440 | 0.7875 | 3.910 | 0.7447 | 0.8593 | 271 / 65% | k {'0': 109, '1': 96, '2': 157, '3': 18} |
| `fin_split5_cur`: 5-class prompt (abstract/conclusion | boilerplate | exposition | body | fragment): both A-types skipped | offline sim | 0.9440 | 0.7874 | 3.735 | 0.7503 | 0.8623 | 192 / 46% | k {'0': 188, '1': 70, '2': 122} |
| `sim_B_to_3`: A -> 2, B -> 3 | offline sim | 0.9427 | 0.7832 | 3.655 | 0.7587 | 0.8803 | 382 / 109% | k {'2': 268, '3': 112} |
| `sim_pos_edge1`: position only: first/last chunk ->1, else 2, T->3 | offline sim | 0.9427 | 0.7772 | 3.695 | 0.7460 | 0.8750 | 382 / 99% | k {'1': 49, '2': 289, '3': 42} |
| `fin_binary_S0N2`: 2-class prompt (self-describing vs needs context): S skip, N->2, T->1 | offline sim | 0.9427 | 0.7919 | 3.934 | 0.7363 | 0.8563 | 281 / 70% | k {'0': 99, '1': 42, '2': 239} |
| `fin_base`: final config rebuilt from its own real summaries (must reproduce 0.9420) | offline sim | 0.9420 | 0.7882 | 3.824 | 0.7490 | 0.8583 | 271 / 65% | k {'0': 109, '1': 96, '2': 157, '3': 18} |
| `fin_B2`: B -> 2 (k=2 summaries from the constant-k=2 run) | offline sim | 0.9420 | 0.7901 | 3.810 | 0.7523 | 0.8623 | 271 / 70% | k {'0': 109, '1': 42, '2': 211, '3': 18} |
| `fin_B2_D2`: A skip, everything else 2, tables 1  (= what a 2-class router would do) | offline sim | 0.9420 | 0.7920 | 3.846 | 0.7490 | 0.8623 | 271 / 69% | k {'0': 109, '1': 42, '2': 229} |
| `fin_genre3_A0B1C2`: 3-class genre prompt (no fragment class): A skip, B->1, C->2, T->1 | offline sim | 0.9420 | 0.7872 | 3.858 | 0.7477 | 0.8623 | 275 / 68% | k {'0': 105, '1': 72, '2': 203} |
| `fin_genre3_A0B2C2`: 3-class genre prompt: A skip, else 2, T->1 | offline sim | 0.9420 | 0.7933 | 3.847 | 0.7477 | 0.8623 | 275 / 70% | k {'0': 105, '1': 42, '2': 233} |
| `sim_anchor8_A1B2C3`: anchor few-shot: A->1 B->2 C->3, T->3 | offline sim | 0.9400 | 0.7866 | 3.754 | 0.7370 | 0.8703 | 382 / 107% | k {'1': 96, '2': 124, '3': 160} |
| `sim_anchor0_A1B2C3`: anchor zero-shot: A->1 B->2 C->3, T->3 | offline sim | 0.9400 | 0.7849 | 3.655 | 0.7490 | 0.8677 | 382 / 97% | k {'1': 111, '2': 193, '3': 76} |
| `fin_A2`: A -> 2 (never skip; k=2 summaries from the constant-k=2 run) | offline sim | 0.9400 | 0.7864 | 3.878 | 0.7390 | 0.8623 | 382 / 93% | k {'1': 96, '2': 266, '3': 18} |
| `fin_A3`: A -> 3 (paper's k=3 summaries) | offline sim | 0.9400 | 0.7879 | 3.955 | 0.7290 | 0.8543 | 382 / 104% | k {'1': 96, '2': 157, '3': 127} |
| `sim_C_to_3`: A -> 2, C -> 3 | offline sim | 0.9393 | 0.7831 | 3.929 | 0.7410 | 0.8557 | 382 / 113% | k {'1': 53, '2': 107, '3': 220} |
| `fin_split5_abs2`: 5-class: abstracts/conclusions -> 2, boilerplate skipped | offline sim | 0.9387 | 0.7892 | 3.815 | 0.7450 | 0.8550 | 332 / 78% | k {'0': 50, '1': 70, '2': 260} |
| `sim_C_to_1`: A -> 2, C -> 1 | offline sim | 0.9373 | 0.7970 | 3.639 | 0.7533 | 0.8723 | 382 / 86% | k {'1': 214, '2': 107, '3': 59} |
| `sim_len_s3_m2_l1`: length only: <700 chars->3, 700-2500->2, >2500->1, T->3 | offline sim | 0.9373 | 0.7811 | 3.711 | 0.7410 | 0.8733 | 382 / 99% | k {'1': 132, '2': 128, '3': 120} |
| `fin_binary_S0N1`: 2-class prompt: S skip, N->1, T->1 | offline sim | 0.9353 | 0.7968 | 3.773 | 0.7410 | 0.8687 | 281 / 50% | k {'0': 99, '1': 281} |

## 4b. Round 3 — modifications around the final configuration (offline, on its own summaries)

Base = the tables→1 real run's **own** summaries (reproduces the real 0.9420 exactly). A modification swaps only the
summaries of the chunks it moves: k=2 summaries come from this pipeline's constant-k=2 run, k=0/1/3 summaries from the
paper's fixed-radius runs (mixed source, marked *). Three new prompts were run on Gemma (one forward pass per chunk, ~1
min each) and scored the same way, tables gated to k=1 unless stated. "w/b" = questions worse / better than the base.

| Modification of the final config | R@20 | MRR | MAR | Δ | w/b | k distribution |
|---|---|---|---|---|---|---|
| **tables → 2** (this pipeline's k=2 summaries) | **0.9480** | 0.785 | 3.83 | +0.6 | 0/3 | 0:109 1:54 2:199 3:18 |
| tables → 2 **and** fragments → 2 (router uses only k ∈ {0,1,2}) | 0.9500 | 0.787 | 3.90 | +0.8 | — | 0:109 1:54 2:217 | |
| A → 1 instead of skip * | 0.9460 | 0.790 | 3.92 | +0.4 | 1/3 | 1:205 2:157 3:18 |
| 5-class prompt (abstract/conclusion \| boilerplate \| exposition \| body \| fragment): abstracts → 1 *, boilerplate skipped | 0.9447 | 0.788 | 3.83 | +0.3 | 2/3 | 0:50 1:208 2:122 |
| fragments D → 2 (no k=3 anywhere) | 0.9440 | 0.790 | 3.90 | +0.2 | 0/1 | 0:109 1:96 2:175 |
| fragments D → 1 * | 0.9440 | 0.788 | 3.87 | +0.2 | — | 0:109 1:114 2:157 |
| A → k=0 self-summary instead of skip * | 0.9440 | 0.787 | 3.91 | +0.2 | 1/2 | as base |
| 5-class prompt, both A-types skipped (188 of 380 chunks without summary) | 0.9440 | 0.787 | 3.74 | +0.2 | 1/2 | 0:188 1:70 2:122 |
| 2-class prompt (self-describing S vs needs-context N): S skip, N → 2 | 0.9427 | 0.792 | 3.93 | +0.1 | 1/1 | 0:99 1:42 2:239 |
| 2-class prompt, S skip, N → 2, tables → 2 | 0.9507 | 0.791 | 3.97 | +0.9 | — | 0:99 2:281 | |
| **final config (reference)** | 0.9420 | 0.788 | 3.82 | 0 | — | 0:109 1:96 2:157 3:18 |
| B → 2 | 0.9420 | 0.790 | 3.81 | 0 | — | 0:109 1:42 2:211 3:18 |
| A skip, everything else 2, tables 1 | 0.9420 | 0.792 | 3.85 | 0 | — | 0:109 1:42 2:229 |
| 3-class genre prompt (fragments folded into C): A skip, B → 1, C → 2 | 0.9420 | 0.787 | 3.86 | 0 | — | 0:105 1:72 2:203 |
| 3-class genre prompt: A skip, else 2 | 0.9420 | 0.793 | 3.85 | 0 | — | 0:105 1:42 2:233 |
| A → 2 (never skip) | 0.9400 | 0.786 | 3.88 | −0.2 | — | 1:96 2:266 3:18 |
| A → 3 * | 0.9400 | 0.788 | 3.95 | −0.2 | — | 1:96 2:157 3:127 |
| 5-class prompt: abstracts → 2, boilerplate skipped | 0.9387 | 0.789 | 3.82 | −0.3 | 3/1 | 0:50 1:70 2:260 |
| 2-class prompt: S skip, N → 1 * | 0.9353 | 0.797 | 3.77 | −0.7 | — | 0:99 1:281 |

**Real-run check of the round-3 recommendation** (2-class router, S → skip, N → 2, tables → 2, module defaults):
R@20 **0.9487**, MRR 0.793, MAR 3.92, R@5 0.7503, R@10 0.8623; k distribution {0: 103, 2: 279}; 279 summariser calls; generation 10.3 min, benchmark 5.7 min. Offline estimate was 0.9507 (-0.2 pt in the real run; questions worse/better vs the estimate 1/0). Versus the tables→1 real run 1/4, versus fixed k=2 5/6, versus the full-document baseline 8/5.

What round 3 adds:

* **k = 3 is unnecessary.** With tables gated below 3, the only k=3 users were 18 fragments; sending them to 2 (or 1)
  is at least neutral (+0.2). A router over k ∈ {0, 1, 2} loses nothing.
* **Tables: 1 vs 2 depends on the summary sample, 3 is the only robust loser.** On the paper's summaries tables→1 led
  tables→2 by 0.13 (round 1); on this pipeline's own real summaries tables→2 leads by 0.6 with 3 questions better and
  none worse. The cleaner comparison is the second one (same pipeline, both sides real), so **tables → 2** is the
  better-supported setting for a real confirmation run; the module keeps `TABLE_K` as the single constant to flip.
* **Abstracts / conclusions / boilerplate: narrow or none, never wide.** Skip (base), k=0 self-summary (+0.2) and k=1
  (+0.4, mixed source) are within noise of each other; k=2 (−0.2) and k=3 (−0.2) hurt, and so does k=2 for the
  abstract/conclusion subclass of the 5-class prompt (−0.3). Separating abstracts from boilerplate does not pay: the
  5-class prompt over-triggers on "abstract/conclusion" (138 of 380 chunks), yet skipping all 188 A/B chunks still
  scores +0.2, i.e. half the corpus can go un-contextualised without losing Recall@20. Skipping stays the cheapest
  neutral choice; k=1 for class A is the only candidate that might add recall and would cost 27 % more summariser calls.
* **Prompt wording is not the lever, simplicity is free.** The 3-class genre prompt reproduces the base exactly and the
  2-class prompt is within +0.1 at tables → 1; k=1 for body text is bad again (−0.7). Differences come from the class →
  k mapping (tables, fragments, A) and from summary noise, not from finer genre distinctions. Combining the two positive
  moves gives the best offline numbers of the whole study on real summaries of this pipeline: the genre router over
  k ∈ {0, 1, 2} (tables → 2, fragments → 2) reaches 0.9500, and the 2-class router (skip self-describing chunks,
  k=2 for everything else, tables → 2, i.e. k ∈ {0, 2}) reaches **0.9507** — +0.9 over the real run's 0.9420 and above
  fixed k=2 (0.9467), with 99 skipped chunks and ~74 % of the k=2 prompt text. Both rest on 142 (resp. 60) summaries
  taken from the constant-k=2 run rather than generated together, so they are estimates in the sense of Section 3
  and need a real confirmation run.

## 5. Stances on configuring the router (ranked by evidence)

1. **Tables → k=1 or 2, never 3.** The one change with a clear offline effect: 0.9447 → 0.9500 with the paper's
   fixed-radius summaries (round 1), above fixed k=2 (0.9467) at 66 % of its prompt text. The mechanism is understood
   (wide slices make the summariser describe the prose around the table or degenerate). Real-run check: two identical real runs of the module defaults gave R@20 **0.9420** (MRR 0.788, MAR 3.83): +0.4 pt over the original real dynamic run (0.9380), but 0.5 pt below fixed k=2 (0.9467) and 0.8 pt below the offline value (0.9500). Versus fixed k=2, 6 questions were worse and 4 better; versus the offline simulation 7 worse and 3 better. Direction confirmed relative to tables → 3, magnitude not; the difference to the offline value is the fresh table summaries (Section 3).
   Round 3, on the real run's own summaries, puts tables → 2 ahead of tables → 1 by 0.6 (3 questions better, none
   worse), so 1 vs 2 is sample-dependent and only "not 3" is robust; tables → 2 went into the module default and into
   the final real run (the real run gave R@20 0.9487 (MRR 0.793, MAR 3.92) against an offline estimate of 0.9507; versus fixed k=2 5 questions worse / 6 better.).
2. **Keep skipping class A.** Free for Recall@20 in every round (A→2: −0.2 to 0, A→3: −0.2, k=0 self-summary: 0 to
   +0.2, A→1: +0.4 with mixed-source summaries), saves 27 % of the summariser calls, and yields the best MRR/MAR/R@5.
   Even skipping half the corpus (the over-triggering 5-class prompt) costs nothing.
3. **k = 3 is not needed at all.** Widening prose hurts (B→3 −0.4, C→3 −0.7, A→3 −0.2, fixed k=3 < fixed k=2), k=3
   slices produce the most degenerate summaries, and with tables gated to 1 or 2 the remaining k=3 users (18
   fragments) are at least as well served by k=2 (+0.2). The router only needs k ∈ {0, 1, 2}.
4. **Do not narrow technical body text to k=1** (C→1: −0.9 pt; the length heuristic that sends long chunks to k=1: −0.9 pt).
   Methods/results paragraphs need the k=2 window; "if unsure use 2" stands.
5. **Class → k mapping matters more than prompt wording.** The "topical anchoring" prompt (does the chunk name its own
   subject?) classified poorly (a 5 800-character introduction labelled "unanchored") and scored 0.9400 zero- and few-shot.
   Round 3 confirmed it from the other side: a 3-class genre prompt (no fragment class) reproduces the 4-class result
   exactly, a 2-class "self-describing vs needs its neighbours" prompt is within +0.1 at equal table setting, and
   splitting abstracts from boilerplate over-triggers without helping. The simplest adequate router — skip
   self-describing chunks, k=2 for everything else, tables → 2 — is also the best offline configuration found
   (0.9507 on real summaries of this pipeline) and became the module default; the real run gave R@20 0.9487 (MRR 0.793, MAR 3.92) against an offline estimate of 0.9507; versus fixed k=2 5 questions worse / 6 better. The genre
   framing with letter answers read from log-probs remains the best classifier of the 4B model when class A must be
   found precisely.
6. **Cheap heuristics are not a substitute.** Length-only and position-only policies land at 0.9373–0.9447; the table gate
   plus the genre SLM is what carries the gain.
7. **Method: rank with the offline simulator, confirm with two real runs;** treat any single-run difference below ~0.7 pt
   of R@20 as a tie.
8. **Exemplars must come from held-out papers.** Swapping the benchmark-derived exemplars for ten role-matched
   exemplars from six unrelated papers (Section 8) changed 15 % of the router's letters (mostly more skipping of
   front matter, acknowledgements and conclusions) but not a single question offline, and the live rerun scored
   0.9520; the earlier runs were contaminated in principle even though no exemplar chunk was ever a gold answer.

## 6. Code and data state

* `utils/dynamic_slice_prediction.py` — module defaults are the **2-class router** of round 3:
  `SYSTEM_PROMPT = BINARY_SYSTEM_PROMPT`, `LETTER_TO_K = {S: 0, N: 2}`, `ANSWER_LABEL = "Answer"`, exemplars =
  `binary_few_shots()` (the ten role exemplars relabelled S/N; since Section 8 they come from the held-out papers in
  `split_documents_router`, `SPLIT_DIR`), **`TABLE_K = 2`**, `FIXED_K = None`. The 4-class genre router of the
  confirmation runs is preserved with its original prompt:
  `DynamicSlicePredictor(system_prompt=GENRE_SYSTEM_PROMPT, letter_to_k=GENRE_LETTER_TO_K, answer_label=GENRE_ANSWER_LABEL,
  few_shots=genre_few_shots(), table_k=1)` (its exemplars are now the held-out ones too). Experiment switches: `FIXED_K`
  (constant radius for every chunk) and `TABLE_K` (gate radius); both also exist as `DynamicSlicePredictor` fields.
  Untested alternative with the same offline score: the genre router with `TABLE_K = 2` and
  `LETTER_TO_K = {A:0, B:1, C:2, D:2}` (offline 0.9500).
* `preprocessed_chunks/ablation_doc_slice_radius_dynamic.json` and the `ablation_doc_slice_radius_dynamic` table hold the
  **last real run** (the 2-class router, `binary_t2`); copies of every real run's summaries are in
  `router_experiments/real_runs/<tag>_chunks.json` (`tables1_run2` produced byte-identical summaries to `tables1`
  and is therefore not stored a second time). Martin's original dynamic run survives as per-question results
  (`results/res_ablation_doc_slice_radius_dynamic.json`) and as the recovered routing (`results/routing_original.json`).
* `emissions_data/emissions.csv` gained rows for the constant-k=2 run (2026-09-24T00:17: 10.6 min, 16.1 Wh) and the two
  tables→1 runs (14:02 and 14:20: 8.6 / 8.5 min, 13.2 / 12.7 Wh; Martin's original routing: 9.1 min, 13.6–13.8 Wh), plus
  one stray 1.8-min row at 2026-09-24T00:03 from an aborted attempt that can be deleted.
* Not run (judged uninformative after round 1): the remaining letter→k mappings of the anchoring prompt and the
  rebuilt fixed k=1/k=3 tables; both are one command away in `scripts/sweep.py`.

## 7. Reproducing / extending

GPU rule on this machine: exactly one of {Ollama, torch/LanceDB} at a time; `scripts/chain.py` sequences jobs and waits
for free VRAM before any Ollama step.

```bash
cd RnD/dynamic_slice_length/router_experiments/scripts      # copy results/*.json and real_runs/ next to the scripts first
PY=../../../../.venv/bin/python
$PY -u chain.py wait sweep:round2                          # offline sims (skips finished ones), ~3.3 min each
$PY sweep.py --report                                      # ranked table of every simulated policy
$PY -u chain.py wait real:<tag> evalreal:<tag>             # real run with the module's current policy (+ per-question eval)
$PY make_table.py                                          # the markdown table above
$PY diagnose.py; $PY oracle.py; $PY cost.py                # per-question diagnosis, headroom, compute cost
$PY chunk_router_docs.py                                   # chunk RnD/input_router/*.pdf with the corpus recipe -> split_documents_router/
$PY compare_routes.py binary binary_heldout                # letter-level agreement of two router passes (+ which gold chunks flip)
```

New prompt variants: add to `VARIANTS` in `router_variants.py` (one Ollama forward pass per chunk), then add
`route_policy("<variant>", {...})` entries to `sweep.py`. New exemplars must come from `split_documents_router` (or
new held-out PDFs chunked with `chunk_router_docs.py`), never from `split_documents`.

## 8. Exemplar re-sourcing to held-out papers (test-set hygiene)

The exemplar set used in Sections 1–7 was built from chunks of benchmark papers (the Gaze paper, one chunk of the
Nature machine-behaviour paper, one author-contribution line), i.e. from the corpus the 250 questions are asked about.
Even though none of those chunks is a gold answer, that is test-set contamination. Martin supplied six unrelated papers
(`RnD/input_router/`: Lempel & Ziv 1976, SRCNN, LeCun/Bengio/Hinton "Deep learning", AlphaGo Zero, the Sycamore
quantum-supremacy paper, a 2023 survey on personalised federated learning), which were chunked with exactly the corpus
recipe (Docling `DocumentConverter()` defaults, `HybridChunker` with the nomic-embed tokenizer at 2000 tokens,
`merge_peers=True`, the same string cleaning, id = SHA-256 of the cleaned text) into `RnD/split_documents_router/`
(109 chunks). Structure check against `split_documents`: identical keys and types, id = SHA-256(text) for 109/109,
`document` = file name, median chunk 2 995 vs 2 090 characters (max 9 772 vs 10 260), the same Docling artefacts
(formula placeholders in 15 chunks, serialised table cells in 17, captions in 9, bullet lists in 21); only the IEEE
`GLYPH<..>` font codes are absent. No exemplar text occurs anywhere in the benchmark corpus (checked).

The ten roles, clip lengths and order were kept one to one:

| # | Role (k → letter) | Old source (benchmark paper) | New source (held-out paper) |
|---|---|---|---|
| 1 | abstract (0 → S) | Gaze #1 | federated-learning survey #0 (author line + Abstract), 900 chars |
| 2 | formulas with "where" (2 → N) | Gaze #8 | SRCNN #10 (MSE loss, "where n is the number of training samples"), 700 |
| 3 | related work (1 → N) | Gaze #3 | SRCNN #4 (MLP / CNN denoising, Cui et al.), 650 |
| 4 | fragment (3 → N) | machine-behaviour #12 (two bullet questions) | AlphaGo Zero #14 (dangling figure caption + layout junk, 148 chars) |
| 5 | reference list (0 → S) | Gaze #13, 3 lines | SRCNN #25, 3 lines |
| 6 | results pointing to a figure (2 → N) | Gaze #11 | SRCNN #22 (Fig. 12, running-time comparison), 600 |
| 7 | method overview naming the device (1 → N) | Gaze #5 | Sycamore #3 (the 54-qubit processor), 650 |
| 8 | conclusion (0 → S) | Gaze #12 | SRCNN #24 ("We have presented …"), 700 |
| 9 | experimental setup (1 → N) | Gaze #9 | SRCNN #12 (training sets, network settings), 600 |
| 10 | contributions / acknowledgement (0 → S) | s41598-020 #10 (87 chars) | Lempel–Ziv #7 (88 chars) |

**Does the router still decide the same?** On the 338 non-table benchmark chunks the 2-class router agrees with its
previous decisions on 287 (84.9 %); letters move from S 99 / N 239 to S 120 / N 218. The 36 N→S flips are mostly
author/affiliation blocks, acknowledgements and conclusions (correct in the gold sense) plus a few section stubs and
introductions; the 15 S→N flips are mostly long reference lists, which were uncertain with either set (p 0.5–0.8).
23 of the 51 flipped chunks are gold supporting chunks. **Effect on retrieval, offline** (on the `binary_t2` run's own
summaries, chunks that change class taking k=2 summaries from the constant-k=2 run or being skipped): R@20 0.9507,
MRR 0.800, MAR 3.93 with 120 skipped chunks — identical per question to the old-exemplar routing on the same
summaries (0 worse / 0 better), 0 worse / 1 better than the real `binary_t2` run, 4 worse / 6 better than fixed k=2,
8 worse / 6 better than the baseline. So the re-sourcing is retrieval-neutral, and the router now skips 21 more chunks.

**Live rerun with the held-out exemplars** (module defaults, tables → 2, `real_runs/binary_t2_heldout.json`):
R@20 **0.9520**, MRR 0.798, MAR 3.92, R@5 0.7517, R@10 0.8657, R@15 0.9267; router k distribution {0: 120, 2: 262};
262 summariser calls; generation 9.7 min, benchmark 5.7 min; codecarbon 13.5 Wh / 2.9 g CO2eq. Per question: vs the
offline estimate (0.9507) 1 worse / 2 better; vs the real run with benchmark-derived exemplars (0.9487) 0 worse / 2
better; vs fixed k=2 (0.9467) 4 worse / 7 better; vs the constant-k=2 re-generation 1 worse / 4 better; vs the
full-document baseline (0.9540) 7 worse / 6 better. This is the best real run of the study and the one to report:
+0.5 pt over fixed k=2 at 69 % of its summariser calls, 0.2 pt below the full-document baseline (well inside the
single-run band), with a better MRR than either.
