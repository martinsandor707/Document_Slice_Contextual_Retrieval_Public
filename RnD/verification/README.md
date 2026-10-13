# Verification scripts for the Springer Nature manuscript

Each script recomputes, from the frozen artefacts under `RnD/`, the numbers that the corresponding part of
`RnD/nature_article/sn_green_rag_article.tex` reports, and prints them next to the derivation. All scripts are
read-only. Run them from the repository root with the project interpreter, e.g.

    .venv/bin/python RnD/verification/chunk_token_statistics.py

> **Public copy of the data (since 2026-10-14).** The records of five source papers are not in the public repository for licensing reasons (see the table in the top-level README): the chunk files, generated contexts and database rows now cover 20 papers / 292 chunks. The article's numbers were computed on all 25 papers / 382 chunks, so these scripts reproduce them only after the five papers' records are restored into their original locations (available from the authors on request, or regenerated from the PDFs with `RnD/end_to_end_dynamic_doc_slice_generation.ipynb`). The router exemplar pool (six held-out papers, `RnD/split_documents_router/`) is not in the public copy either; the ten exemplar passages are shipped in `RnD/dynamic_slice_length/router_exemplars_heldout.json`, and `chunk_character_statistics.py` and `chunk_token_statistics.py` print the pool's archived statistics in its place.

| Script | Verifies |
|---|---|
| `chunk_character_statistics.py` | Table 1 rows 1–4 and 7–11, Table 4 (chunks, characters, questions, gold references, exemplar pool), Sect. 3.1 artefact counts |
| `chunk_token_statistics.py` | Table 1 rows 5–6 (token quartiles, bands, cap), Table 4 token means/medians for both corpora, the token median used in Sect. 3.5 |
| `prompt_length_statistics.py` | Sect. 3.2 (summariser system prompt 789 chars / 164 tokens, per-call bound), Sect. 3.3 (router system and user prompt lengths, ~2 000 tokens), Sect. 3.5 (typical k=2 prompt), Table 3; per-call character and nomic-token lengths of every summariser and router message for the baseline, fixed k = 0..4 and the routed runs |
| `prompt_volume_and_attention_proxy.py` | Sect. 3.5 prompt volumes and sum(P_i^2) ratios, Table 6 / Table 7 "prompt volume" columns, Fig. 4 data, the typical k=2 prompt vs the 12 190-token bound |
| `energy_deltas_from_emissions_csv.py` | Table 5, Table 6 energy column, Table 7 time/energy/GPU/emission rows and deltas, Table 8, Figs 7 and 9, abstract and Sect. 4.2 / 5.1 percentages, grid carbon intensity |
| `retrieval_significance_tests.py` | Table 7 `|Q_20^+|`, the provisional BCa bootstrap / McNemar footnote, per-question better/worse counts, no-hit counts (Sect. 3.6, 4.4) |
| `router_routing_statistics.py` | Sect. 4.3 / Table 7 routing statistics (k distribution, k-bar, calls, summary length, replicate), Sect. 3.3 gold-set agreement, the negative short-chunk/skip overlap check |
| `generated_summary_statistics.py` | Sect. 3.2 and 4.3 statements about the generated summaries: min / Q1 / median / mean / Q3 / max of summary length in characters and nomic tokens for every stored generation (baseline, fixed k = 0..4, routed runs, policy corpus), forbidden-phrase and degenerate-copy counts, per-radius breakdown |
| `policy_corpus_retrieval_metrics.py` | Table 8 (policy corpus): Recall / MRR / MAR at K = 5..20 on the 80 policy questions for the full-document baseline, the rebuilt k=3 slice table (`tobacco_sliding_text_table`, embeds summary + chunk; built by `RnD/build_tobacco_sliding_text_table.py`) and the original slice table (embedded the raw chunk only), with per-question tests against the baseline; needs the GPU (nomic embedder + ColBERT) |
| `ollama_runner_vram.py` | Figure 6 'resident GPU memory' row and the memory statements of Sect. 2.5 / Table 2: nvidia-smi peak and /api/ps split of the gemma3:4b-it-qat runner at num_ctx 16 384 and 30 000 while it processes the longest k=3 prompt; writes `ollama_runner_vram.json`; needs the GPU and a running Ollama server |
| `ollama_offload_under_vram_pressure.py` | Sect. 2.5 / Table 2 statements about GPU-memory pressure and host offloading: layers kept on the GPU (`ollama ps`, server log), nvidia-smi peak, host RSS and CPU time of the Ollama processes for the longest k=3 prompt (num_ctx 30 000 and 16 384) or the full-document baseline prompt on the longest document (num_ctx 30 000), with a chosen amount of GPU memory already held by other processes (`gpu_ballast.py`, or a live Docling / Docling + embedder process via `coresident_gpu_footprint.py --hold`); run on 2026-10-05 against the installed Ollama 0.35.1 and against the extracted package of Ollama 0.17.5 (the release installed when the March 2026 baseline ran); results in `ollama_offload_pressure_*.json` |
| `coresident_gpu_footprint.py` | GPU memory held by the other GPU users of the March notebooks: the Docling converter after one PDF (1.7–1.8 GiB) and the LanceDB nomic embedder after embedding the 382 chunks (+4.2–4.5 GiB, 5.9–6.1 GiB together); `--hold` keeps them resident for the script above |
| `gpu_ballast.py` | Helper: holds a given number of GiB on the GPU for the script above |
| `baseline_prefix_cache_reuse.py` | Whether the full-document baseline could reuse Ollama's prompt cache between consecutive chunks of a document (cell 3 of `anthropic_traditional_chunking.ipynb` prepends one more `FULL DOCUMENT:` header per chunk); measured prompt_eval counts on 0.17.5 and 0.35.1 |
| `latex_manuscript_consistency_check.py` | Structural checks of the .tex file: braces, environments, labels/references, citation keys, placeholders, word budgets, notes coverage |

`_common.py` holds the shared paths and the position-based reader for `emissions.csv` (the rows written by
codecarbon 3.3.1 are shifted by two columns after `region` relative to the 3.2.3 header).

Retrieval metrics of Tables 6–7 are not recomputed here (the policy-corpus metrics of Table 8 are, by `policy_corpus_retrieval_metrics.py`): they are the printed outputs of
`RnD/benchmarking_retrievals.ipynb` and `RnD/ablation_doc_slice_retrieval.ipynb` and require the GPU stack; the
per-question files used by `retrieval_significance_tests.py` are their stored per-question results.

## GPU-memory pressure findings (2026-10-05, RTX 4060 Laptop 8 GiB, gemma3:4b-it-qat)

Measured with `ollama_offload_under_vram_pressure.py`; one row per JSON file. "Co-resident" is GPU memory held by
other processes before the model loads. CPU = CPU time of the Ollama processes during a warm call divided by its
wall-clock time, as cores and as a share of the 24 hardware threads (the share a system monitor shows).

| Ollama | prompt | num_ctx | co-resident | layers on GPU | `ollama ps` | nvidia-smi peak | prompt eval | CPU |
|---|---|---|---|---|---|---|---|---|
| 0.17.5 | k=3, 11.8k tok | 30 000 | none | 35/35 | 6.4 GB, 100 % GPU | 4 995 MiB | 3.2 s | 2.2 cores, 9 % |
| 0.17.5 | k=3 | 16 384 | none | 35/35 | 100 % GPU | 4 711 MiB | 3.2 s | 2.0 cores, 9 % |
| 0.17.5 | k=3 | 30 000 / 16 384 | 1.65 GiB ballast | 35/35 / 35/35 | 100 % GPU | 6 644 / 6 360 MiB | 3.2 s | 2.2 / 1.8 cores |
| 0.17.5 | k=3 | 30 000 / 16 384 | 2.7 GiB ballast | 34/35 / 35/35 | | 5 576 / 7 382 MiB | 3.3 / 3.2 s | 2.8 / 2.2 cores |
| 0.17.5 | k=3 | 30 000 / 16 384 | 3.7 GiB ballast | 34/35 / 34/35 | | 6 600 / 6 318 MiB | 3.2 s | 2.9 cores, 12 % |
| 0.35.1 | k=3 | 30 000 / 16 384 | none | 35/35 | 3.6 GB, 100 % GPU | 4 947 / 4 663 MiB | 3.2 s | 4.6 / 4.7 cores, 19–20 % |
| 0.35.1 | k=3 | 30 000 / 16 384 | 2.7 GiB ballast | 24/35 / 29/35 | | 6 966 / 6 978 MiB | 4.6 / 3.9 s | 4.2 / 5.4 cores |
| 0.35.1 | k=3 | 30 000 / 16 384 | 3.7 GiB ballast | 11/35 / 13/35 | | 6 936 / 7 004 MiB | 6.5 / 6.3 s | 5.2 / 5.5 cores, 22–23 % |
| 0.17.5 | full document, 19.9k tok | 30 000 | none | 35/35 | 6.4 GB, 100 % GPU | 4 995 MiB | 5.6 s | 2.7 cores, 11 % |
| 0.17.5 | full document | 30 000 | Docling, 1.8 GiB | 35/35 | 100 % GPU | 6 812 MiB | 5.6 s | 3.4 cores, 14 % |
| 0.17.5 | full document | 30 000 | Docling + embedder, 6.1 GiB | 12/35 | 6.5 GB, 81 %/19 % CPU/GPU | 7 584 MiB | 10.3 s | 3.3 cores, 14 % |
| 0.35.1 | full document | 30 000 | none | 35/35 | 3.6 GB, 100 % GPU | 4 947 MiB | 5.5 s | 4.5 cores, 19 % |
| 0.35.1 | full document | 30 000 | Docling, 1.8 GiB | 35/35 | 100 % GPU | 6 764 MiB | 5.5 s | 4.5 cores, 19 % |
| 0.35.1 | full document | 30 000 | Docling + embedder, 5.9 GiB | 0/35 | 3.7 GB, 95 %/5 % CPU/GPU | 6 348 MiB | 13.4 s | 5.2 cores, 21 % |

KV cache of gemma3:4b-it-qat in both releases (interleaved sliding-window attention): 5 global layers at
118 MiB each for 30 000 cells (64 MiB for 16 384) plus 29 sliding-window layers at 6 MiB each (1 536 cells), i.e.
764 MiB at 30 000 and 494 MiB at 16 384; both releases enable flash attention for this model by default.
`ollama ps` sizes differ between releases only by accounting (0.17.5 counts the 1.3 GB input embedding, 0.35.1 does
not); the nvidia-smi peaks agree within 50 MiB. The installed service was never modified: 0.17.5 was run from the
cached pacman packages, extracted to a temporary directory, as a second server on port 11435 against the user model
store, and removed afterwards.

Why the two releases split differently at the same pressure (debug logs of both servers, Docling + embedder case,
about 1.7 GiB free): 0.17.5 runs gemma3 on Ollama's own engine and computes the layout itself. It subtracts a
457 MiB minimum and the partial graph (290 MiB) from the free memory, gets 905 MiB of "available layer vram", fills
it with the last 12 repeating layers (22..33) and leaves the 2.2 GB output/embedding block on the CPU, so the GPU ends
up 93 % full. 0.35.1 routes gemma3 to `llama-server` without any `--n-gpu-layers` argument and lets llama.cpp's
automatic fit choose ("using device CUDA0 ... 1781 MiB free" then "offloaded 0/35 layers"), which keeps a much larger
safety margin; it also logs "disabling multimodal projector offload reason=limited-vram". The same policy difference
shows at lighter pressure: with 2.7 GiB and 3.7 GiB occupied, 0.17.5 kept 34/35 layers (dropping only the output
block) while 0.35.1 kept 24/35 and 11/35. Note on the 0.17.5 server: the first instance started on 2026-10-05 kept
running until 13:30 because its command line used a relative path that the stop pattern did not match; later start
attempts could not bind port 11435 and exited, so every 0.17.5 measurement came from that one server (each run
printed the version from /api/version).
