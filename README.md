# Document Slice Contextual Retrieval

Public repository of the *Document Slice Contextual Retrieval* project: a local retrieval-augmented generation (RAG)
pipeline that fits on a consumer GPU with 8 GB of VRAM. It follows the idea of
[Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), where a language model writes a
short summary of each chunk's role before the chunk is embedded, but instead of showing the summariser the whole
document for every chunk it shows a *document slice*: the chunk and its *k* neighbouring chunks. A small router decides
*k* per chunk (0 for self-describing passages, 2 otherwise, with a deterministic rule for tables), which cuts prompt
volume, run time and energy while keeping retrieval quality. Everything runs offline: Docling parses the PDFs, Ollama
runs `gemma3:4b-it-qat` for the summaries and the routing, `nomic-embed-text-v1.5` embeds the enriched chunks, LanceDB
serves hybrid (vector + BM25) search, ColBERTv2 re-ranks, and CodeCarbon measures the energy of every run.

## Publications

This repository is the companion of two publications. Please cite them if you use the code or the data.

1. M. T. Sándor and R. Lakatos, "Efficient Local Retrieval-Augmented Generation via Neighboring Document Slice
   Contextualization," in *Proc. 2026 IEEE 4th Conference on Information Technology and Data Science (CITDS)*, 2026,
   pp. 257–262. DOI: [10.1109/CITDS70571.2026.11706860](https://doi.org/10.1109/CITDS70571.2026.11706860)
2. M. T. Sándor and R. Lakatos, "Sustainable Local RAG Architectures for Energy Governance: A Green AI Approach via
   Document Slice Contextualisation," journal article submitted to a Springer Nature journal (under review, DOI pending).
   The manuscript source is in [`RnD/nature_article/`](RnD/nature_article/).

```bibtex
@inproceedings{sandor2026documentslice,
  author    = {S{\'a}ndor, Martin Tibor and Lakatos, R{\'o}bert},
  title     = {Efficient Local Retrieval-Augmented Generation via Neighboring Document Slice Contextualization},
  booktitle = {2026 IEEE 4th Conference on Information Technology and Data Science (CITDS)},
  year      = {2026},
  pages     = {257--262},
  doi       = {10.1109/CITDS70571.2026.11706860}
}
```

## Repository layout

```
.
├── README.md                        this file
├── LICENSE                          MIT licence of the code
├── LICENSE-DATA                     CC BY 4.0 licence of the data and figures produced by the authors
├── .gitignore
└── RnD/
    ├── README.md                    pipeline notes: workflow, models, peak VRAM
    ├── requirements.txt             pinned Python 3.12 environment
    ├── router_prompt_engineering.md design log and measurement table of the slice-radius router
    ├── build_tobacco_sliding_text_table.py   rebuilds the policy-corpus k=3 slice table (summary + chunk embedded)
    ├── *.ipynb                      the nine notebooks of the reported runs (see "Notebooks" below)
    ├── input/                       21 source PDFs of the scientific corpus (see "Source papers excluded ..." below)
    ├── smoking/                     the European Commission policy corpus: policy/ (4 PDFs) and studies/ (2 PDFs)
    ├── split_documents/             frozen Docling chunk files of the scientific corpus; smoking/ holds the policy corpus
    ├── split_documents_end_to_end/  chunk files written by the end-to-end run that starts from the PDFs
    ├── preprocessed_chunks/         generated contexts of every run: baseline, control, fixed k = 0..4, routed, end-to-end, policy corpus
    ├── db/                          16 LanceDB tables indexed from preprocessed_chunks/ (53 MB)
    ├── q_and_a/                     question sets; Gemini/scientific_multi_chunk_control.json and Gemini/smoking.json are the ones used in the papers
    ├── dynamic_slice_length/        slice_radius_gold.json (hand-labelled radii), predicted_k_all_chunks.json,
    │   │                            router_exemplars_heldout.json (the ten router exemplar passages) and router_experiments/
    │   └── router_experiments/      real_runs/ (live routing records), results/ (per-question retrieval results and policy simulations), scripts/
    ├── benchmarking_results/        outputs and plots of the conference paper: answer_generation/, doc_slice_ablation/, preprocessing/, retrieval/, tobacco/
    ├── emissions_data/emissions.csv CodeCarbon energy ledger of every run
    ├── Models/                      Ollama Modelfiles of the summariser variants
    ├── utils/                       shared code: dynamic_slice_prediction.py (the router), split_corpus.py, metrics, LLM judge
    ├── verification/                scripts that recompute every number reported in the journal manuscript (README inside)
    └── nature_article/              manuscript source (sn_green_rag_article.tex, refs.bib, sn-jnl.cls, sn-mathphys-num.bst) and images/ (figure scripts and PNGs)
```

## Reproducing the experiments

### Environment

All measurements were made on one machine with an NVIDIA RTX 4060 Laptop GPU (8 GB). Python 3.12 and the pinned
packages in `RnD/requirements.txt` (CUDA 12 builds of torch) reproduce the software stack:

```bash
uv venv --python 3.12 .venv                                        # or: python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r RnD/requirements.txt   # or: .venv/bin/pip install -r RnD/requirements.txt
```

The summariser and the router run on [Ollama](https://ollama.com) with `gemma3:4b-it-qat`; the summariser variants
used by the notebooks are Ollama models built from the Modelfiles in `RnD/Models/`:

```bash
ollama pull gemma3:4b-it-qat
ollama create chunker_full_doc   -f RnD/Models/chunker_full_doc.Modelfile    # document-slice summariser, num_ctx 16384
ollama create anthropic_chunking -f RnD/Models/chunker_anthropic.Modelfile   # full-document baseline, num_ctx 30000
```

The embedding model (`nomic-ai/nomic-embed-text-v1.5`) and the re-ranker (`colbert-ir/colbertv2.0`) are downloaded
from the Hugging Face Hub on first use; the verification scripts load the tokenizer from the local Hugging Face cache
in offline mode. The peak GPU memory of the pipeline was 7 695 MiB (see `RnD/README.md`).

### Notebooks

Run them from `RnD/` with the kernel of the environment above, in this order. Each notebook names its input and
output files in its first cells; the files they produced are the frozen artefacts described in the next section.

| # | Notebook | What it does |
|---|---|---|
| 1 | `doc_slice_chunking.ipynb` | Parses the PDFs in `input/` with Docling (layout model, TableFormer, RapidOCR), chunks them with the HybridChunker at 2 000 nomic tokens, cleans the text, writes the chunk files to `split_documents/`, generates the fixed-radius slice contexts with `chunker_full_doc`, embeds and indexes them |
| 2 | `anthropic_traditional_chunking.ipynb` | The full-document baseline (`anthropic_chunking`, the original Contextual Retrieval prompt) and the no-context control, for the scientific and the policy corpus |
| 3 | `ablation_doc_slice_generation.ipynb` | Slice contexts for the fixed radii k = 0..4 (`preprocessed_chunks/ablation_doc_slice_radius_<k>.json`, `db/ablation_doc_slice_radius_<k>.lance`) |
| 4 | `ollama_dynamic_slice_length.ipynb` | The slice-radius router: gold labels, prompt and exemplar evaluation, per-chunk predictions (`dynamic_slice_length/`) |
| 5 | `doc_slice_chunking_dynamic.ipynb` | Slice contexts with a per-chunk radius from the frozen chunk files (first version of the routed pipeline) |
| 6 | `ablation_dynamic_doc_slice_generation.ipynb` | The routed pipeline (table gate + router + summariser) on the frozen chunk files; writes `preprocessed_chunks/ablation_doc_slice_radius_dynamic.json` and the matching table |
| 7 | `end_to_end_dynamic_doc_slice_generation.ipynb` | The routed pipeline from the PDFs (parse, route, summarise) with per-stage timings; writes `split_documents_end_to_end/`, `preprocessed_chunks/end_to_end_doc_slice_radius_dynamic.json` and `benchmarking_results/preprocessing/end_to_end_routed_timings.json` |
| 8 | `benchmarking_retrievals.ipynb` | Recall, MRR and MAR at K of one table against the baseline on the question sets, plus the time and energy plots |
| 9 | `ablation_doc_slice_retrieval.ipynb` | The same metrics for every ablation table (fixed k, routed, baseline, control), per-question results and the ablation plots |

The router experiments of the journal article (prompt variants, offline policy simulator, sweeps, the master
measurement table) are the scripts in `RnD/dynamic_slice_length/router_experiments/scripts/`; their design log is
`RnD/router_prompt_engineering.md`. `RnD/build_tobacco_sliding_text_table.py` rebuilds the policy-corpus slice table
that embeds summary and chunk together.

### Verification scripts

Every number reported in the journal manuscript can be recomputed from the frozen artefacts with the scripts in
`RnD/verification/`; the table in `RnD/verification/README.md` maps each script to the tables and sections it
verifies. Run them from the repository root, for example:

```bash
.venv/bin/python RnD/verification/chunk_character_statistics.py
```

Most scripts only read JSON and CSV files; `policy_corpus_retrieval_metrics.py`, `ollama_runner_vram.py`,
`ollama_offload_under_vram_pressure.py` and `coresident_gpu_footprint.py` need the GPU and a running Ollama server.
Both papers were computed on the full 25-paper corpus; on this public copy, which holds 20 papers, the corpus-level
statistics differ (see "Source papers excluded for licensing reasons"). The two chunk-statistics scripts print the
archived statistics of the router exemplar pool, which is not part of this copy either.

### Figures of the manuscript

The scripts in `RnD/nature_article/images/` read the frozen artefacts and write PNG files only, next to themselves
(`_style.py` holds the shared style and the `save()` helper). Figures 1, 2, 3 and 9 are static images without a script.

```bash
cd RnD/nature_article/images
../../../.venv/bin/python fig04_cost_curves.py      # likewise fig06_energy_emissions.py, fig06_radius_heatmap.py,
                                                    # fig07_runtime_scopes.py, fig08_recall_trajectories.py, fig10_shared_gpu_offload.py
```

The manuscript compiles with TeX Live (the `sn-jnl` class and the bibliography style are included):

```bash
cd RnD/nature_article && latexmk -pdf sn_green_rag_article.tex
```

## Frozen artefacts: do not regenerate in place

The chunk files, generated contexts, LanceDB tables, question sets, gold labels, routing records, per-question
results, timings and the energy ledger are the frozen record of the runs reported in the two papers, and the
verification scripts and figure scripts read them as they are. The notebooks write to exactly these paths, and
language-model output is not bit-reproducible (a regeneration at constant k = 2 reproduced the retrieval metric but
changed most summaries). If you re-run a notebook, change its output names or work on a copy of the repository, and
keep the archived files untouched.

## Source papers excluded for licensing reasons

The benchmark corpus of the article consists of 25 open-access scientific papers. A licence check of every PDF (Crossref metadata, October 2026) showed that five of them may not be redistributed, or not in derived form: four are published under licences that permit reading but not republication, and one is CC BY-NC-ND, which permits sharing the unmodified article but not derivative works such as cleaned text chunks and machine-generated summaries. On the advice of the project supervisor, every copy of those papers' text was removed from the public repository on 2026-10-14. The other 20 papers are CC BY licensed and remain in full.

| Paper | Venue, year | DOI | Licence | Removed from the public repository |
|---|---|---|---|---|
| Machine behaviour (Rahwan et al.) | Nature, 2019 | [10.1038/s41586-019-1138-y](https://doi.org/10.1038/s41586-019-1138-y) | subscription article (Springer Nature text-and-data-mining licence only) | PDF, chunk files, generated contexts, database rows |
| Electron Paramagnetic Resonance Study on 28Si Single Crystal for the Future Realization of the Kilogram | IEEE Transactions on Instrumentation and Measurement, 2019 | [10.1109/TIM.2018.2884044](https://doi.org/10.1109/TIM.2018.2884044) | IEEE Open Access Publishing Agreement (free to read, no redistribution) | PDF, chunk files, generated contexts, database rows |
| Stock Market Prediction via Multi-Source Multiple Instance Learning | IEEE Access, 2018 | [10.1109/ACCESS.2018.2869735](https://doi.org/10.1109/ACCESS.2018.2869735) | IEEE Open Access Publishing Agreement (free to read, no redistribution) | PDF, chunk files, generated contexts, database rows |
| The Graph Database: Jack of All Trades or Just Not SQL? | IT Professional, 2017 | [10.1109/MITP.2017.4241475](https://doi.org/10.1109/MITP.2017.4241475) | subscription article (© IEEE) | PDF, chunk files, generated contexts, database rows |
| The Application of the SOFM Neural Network and Internet of Things in Rural Revitalization | IEEE Access, 2023 | [10.1109/ACCESS.2023.3327131](https://doi.org/10.1109/ACCESS.2023.3327131) | CC BY-NC-ND 4.0 (unmodified copies may be shared, derivatives may not) | chunk files, generated contexts, database rows (the unmodified PDF stays in `RnD/input/`) |

What this means in practice:

- `RnD/input/` holds 21 of the 25 PDFs; `RnD/split_documents/` and `RnD/split_documents_end_to_end/` hold the chunk files of 20 papers; every file under `RnD/preprocessed_chunks/`, the run records under `RnD/dynamic_slice_length/router_experiments/real_runs/`, the gold labels in `RnD/dynamic_slice_length/slice_radius_gold.json`, the older question set `RnD/q_and_a/scientific.json`, and every LanceDB table under `RnD/db/` contain the records of those 20 papers only (292 of the original 382 chunks).
- Everything that carries no article text is complete: the question sets in `RnD/q_and_a/Gemini/` (they reference chunks by content hash), the per-question retrieval results, the routing records, the energy ledger and the notebooks.
- All numbers, tables and figures in the article were computed on the full 25-paper corpus, so the verification scripts in `RnD/verification/` reproduce the per-corpus statistics only after the missing records are restored.
- To reproduce the experiments end to end, download the five PDFs through the DOIs above (institutional access is needed for the two subscription articles), place them in `RnD/input/` under their original file names, and run `RnD/end_to_end_dynamic_doc_slice_generation.ipynb`, which parses, routes and summarises them again; summaries will not be bit-identical to the archived run. The original chunk and context records of the five papers are available from the authors on reasonable request for verification purposes.

### Router exemplar papers (not redistributed)

The slice-radius router (`RnD/utils/dynamic_slice_prediction.py`) takes its ten few-shot exemplar passages from six
papers that are not part of the benchmark corpus (they were held out for test-set hygiene). Their PDFs and chunk
files (`RnD/input_router/`, `RnD/split_documents_router/`) are not in the public repository either, because their
licences do not permit redistribution: two Nature subscription articles, one Nature article available under a
text-and-data-mining licence only, one IEEE Open Access Publishing Agreement paper, one IEEE subscription paper and
the 1976 Lempel–Ziv paper.

| File name used by the pipeline | Paper | Venue, year | DOI |
|---|---|---|---|
| `arute2019.pdf` | Quantum supremacy using a programmable superconducting processor (Arute et al.) | Nature, 2019 | [10.1038/s41586-019-1666-5](https://doi.org/10.1038/s41586-019-1666-5) |
| `Deep-learning.pdf` | Deep learning (LeCun, Bengio, Hinton) | Nature, 2015 | [10.1038/nature14539](https://doi.org/10.1038/nature14539) |
| `Image_Super-Resolution_Using_Deep_Convolutional_Networks.pdf` | Image Super-Resolution Using Deep Convolutional Networks (Dong et al.) | IEEE Transactions on Pattern Analysis and Machine Intelligence, 2016 | [10.1109/TPAMI.2015.2439281](https://doi.org/10.1109/TPAMI.2015.2439281) |
| `lempel1976.pdf` | On the Complexity of Finite Sequences (Lempel and Ziv) | IEEE Transactions on Information Theory, 1976 | [10.1109/TIT.1976.1055501](https://doi.org/10.1109/TIT.1976.1055501) |
| `silver2017.pdf` | Mastering the game of Go without human knowledge (Silver et al.) | Nature, 2017 | [10.1038/nature24270](https://doi.org/10.1038/nature24270) |
| `tan2023.pdf` | Towards Personalized Federated Learning (Tan et al.) | IEEE Transactions on Neural Networks and Learning Systems, 2023 | [10.1109/TNNLS.2022.3160699](https://doi.org/10.1109/TNNLS.2022.3160699) |

The ten exemplar passages themselves, 88 to 906 characters each, are shipped verbatim in
`RnD/dynamic_slice_length/router_exemplars_heldout.json`, exported from the chunk files with `few_shots_to_json`. The
router loads them from that file whenever `RnD/split_documents_router/` is absent, so the routed pipeline and the
router experiments run unchanged on this copy. To rebuild the pool itself, obtain the six papers through the DOIs,
save them under `RnD/input_router/` with the file names above and run
`RnD/dynamic_slice_length/router_experiments/scripts/chunk_router_docs.py`, which writes `RnD/split_documents_router/`
with the corpus recipe; the router then reads the chunk files again. Your own exemplars can be passed with
`few_shots=` or a JSON file through `few_shots_from_json`, and the environment variables `DSL_SPLIT_DIR` and
`DSL_EXEMPLARS_JSON` point the module at other locations. The exemplar pool is described in Sect. 8 of
`RnD/router_prompt_engineering.md`.

## Licence

- **Code** (Python scripts, notebooks, Modelfiles and figure scripts): MIT licence, see [`LICENSE`](LICENSE).
- **Data and figures produced by the authors** (chunk files, generated contexts, LanceDB tables, question sets,
  labels, routing records, results, energy ledger, measurement records and the PNG figures): CC BY 4.0, see
  [`LICENSE-DATA`](LICENSE-DATA); please attribute by citing the two publications above.
- **Not covered**: the source PDFs in `RnD/input/` and `RnD/smoking/` keep their own licences (CC BY for 20 papers,
  CC BY-NC-ND 4.0 for one, the European Commission's reuse policy for the policy documents); the ten exemplar
  passages are short quotations from non-redistributable papers; the manuscript source in `RnD/nature_article/` is
  the authors' submitted article and Springer Nature's class and bibliography style files come with their own terms.
