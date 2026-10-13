# RnD: pipeline notes

Technical notes on the Document Slice Contextual Retrieval pipeline. The directory layout, the two publications,
the reproduction steps and the licence situation are in the [top-level README](../README.md).

The chunking and contextualisation logic follows [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
flow, with the limitation of at most 8 GB of VRAM in mind. The peak GPU memory used during chunking, context
generation and embedding was ***7695 MiB***.

## Workflow

1. The PDF's contents and metadata are extracted with Docling (layout analysis, TableFormer for tables, RapidOCR
   when a page is an image).
2. The contents are chunked with Docling's HybridChunker to the token limit of the embedding model used later
   (2 000 tokens of the `nomic-embed-text-v1.5` tokeniser), then cleaned (Unicode normalisation, URL removal,
   whitespace); the chunk id is the SHA-256 of the cleaned text.
3. Each chunk's contents and role are summarised in one or two sentences by a local Ollama language model
   (`gemma3:4b-it-qat`). The *document slice* variant shows the model the chunk together with its *k* neighbouring
   chunks instead of the full document. *k* is either fixed (0 to 4 in the ablation) or chosen per chunk by the
   router in `utils/dynamic_slice_prediction.py`: a deterministic table gate (Tier 1) and a single forward pass of the
   same model with ten few-shot exemplars (Tier 2), returning *k* = 0 for self-describing passages and *k* = 2 otherwise.
4. The summary is ***prepended*** to the chunk; this enriched text is what gets embedded and indexed (see Anthropic's
   article on why steps 3 and 4 are needed).
5. The enriched chunks are embedded into dense vectors for semantic similarity.
6. They are also indexed for BM25 full-text search.
7. Everything is stored in a LanceDB table (`db/`).
8. A query performs a hybrid search (vector top-20 and BM25 top-20) that returns more than the *N* entries asked for.
9. The candidates are fused and re-ranked with ColBERTv2.
10. The top *N* re-ranked chunks are returned, ready to be used for RAG.

Energy, duration and emissions of every run were recorded with CodeCarbon into `emissions_data/emissions.csv`.

## Models used

- Docling (`docling 2.68.0`)
    - layout model (document layout understanding)
    - TableFormer (table extraction)
    - RapidOCR (in case the PDF is an image)
- Chunking tokeniser: `nomic-ai/nomic-embed-text-v1.5` through Docling's HybridChunker
- Summarising and routing: `gemma3:4b-it-qat` via Ollama (Modelfiles of the summariser variants in `Models/`)
- Embedding: `nomic-ai/nomic-embed-text-v1.5` (768 dimensions)
- Reranking: `colbert-ir/colbertv2.0`

After the documents have been processed, only the embedding and the reranking model are needed for retrieval, so
there is plenty of room on the GPU for the Ollama model that generates the answers.

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

The six papers of the router's few-shot exemplar pool (`input_router/`, `split_documents_router/`) are not
redistributed either. The ten exemplar passages are shipped in `dynamic_slice_length/router_exemplars_heldout.json`,
which the router loads when the chunk files are absent; the top-level README lists the six papers by DOI and explains
how to rebuild or replace the pool.
