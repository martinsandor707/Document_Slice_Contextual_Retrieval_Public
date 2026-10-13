"""Overview of the LanceDB table RnD/db/tobacco_sliding_table.lance (EC tobacco-policy corpus with document-slice
contexts) and how it differs from RnD/db/ablation_doc_slice_radius_dynamic.lance (scientific corpus, routed slice
radius): size and version history, schema with the role of each column (embedding source, dense vector, keyword/BM25
full-text index, scalar index), embedding model and vector norms, documents, lengths in characters and nomic tokens,
how `text` is built from `context` and `original_text`, duplicate ids, one example row, and a side-by-side diff of the
two schemas and of the columns behind the dense and the keyword half of the hybrid search.
Built by: tobacco_sliding_table - cell 9 of RnD/doc_slice_chunking.ipynb from RnD/preprocessed_chunks/tobacco_sliding.json;
ablation_doc_slice_radius_dynamic - cell 4 of RnD/ablation_dynamic_doc_slice_generation.ipynb. The retrieval notebooks
query both with search(query_type="hybrid", vector_column_name="vector", fts_columns="text") and rerank with
ColbertReranker(), whose default column is "text".
Read-only; the embedding model is never loaded. Other tables in RnD/db can be passed as arguments.
Run from the repository root:  .venv/bin/python RnD/verification/tobacco_sliding_table_overview.py [TABLE [OTHER]]
(CPU only, offline)"""
import collections, json, statistics, sys
import lancedb
import numpy as np
import pyarrow as pa
from _common import RND, nomic_tokenizer

TABLE = sys.argv[1] if len(sys.argv) > 1 else "tobacco_sliding_table"
OTHER = sys.argv[2] if len(sys.argv) > 2 else "ablation_doc_slice_radius_dynamic"
TEXT_COLS = ("text", "original_text", "context")

db = lancedb.connect(str(RND / "db"))
tok = nomic_tokenizer()


def spread(xs):
    return f"{min(xs)} / {statistics.median(xs):.0f} / {statistics.mean(xs):.1f} / {max(xs)}"


def clip(v, n=110):
    return repr(v[:n]) + ("..." if len(v) > n else "") if isinstance(v, str) else repr(v)


def describe(name):
    t = db.open_table(name)
    tbl = t.to_arrow()
    cols = tbl.column_names
    vec_cols = [f.name for f in tbl.schema if pa.types.is_fixed_size_list(f.type)]
    rows = tbl.drop_columns(vec_cols).to_pylist()
    # Parsed from the schema metadata because t.embedding_functions would instantiate the HF model (on CUDA).
    emb = json.loads((tbl.schema.metadata or {}).get(b"embedding_functions", b"[]"))
    indices = list(t.list_indices())
    roles = collections.defaultdict(list)
    for f in emb:
        roles[f["source_column"]].append(f"embedding source of {f['vector_column']}")
        roles[f["vector_column"]].append(f"dense vector of {f['source_column']}")
    for ix in indices:
        for c in ix.columns:
            roles[c].append(f"{ix.index_type} index {ix.name}" + (" = keyword/BM25 search" if ix.index_type == "FTS" else ""))
    toks = {c: [len(tok.tokenize(r[c])) for r in rows] for c in TEXT_COLS if c in cols}
    docs = collections.Counter(r["document"] for r in rows)

    path = RND / "db" / f"{name}.lance"
    stats, versions = t.stats(), t.list_versions()
    creations = [v for v in versions if v["metadata"].get("total_rows") == "0"]  # create_table(schema=..., mode="overwrite")
    first, last = creations[-1] if creations else versions[0], versions[-1]
    disk = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    print(f"== {name}  ({path.relative_to(RND.parent)})")
    print(f"   {len(rows)} rows in {stats['fragment_stats']['num_fragments']} fragments; data {stats['total_bytes'] / 1e6:.2f} MB, "
          f"directory incl. old versions {disk / 1e6:.2f} MB")
    print(f"   {len(versions)} versions, table (re)created {len(creations)}x; current build v{first['version']}..v{last['version']}, "
          f"{first['timestamp']:%Y-%m-%d %H:%M} .. {last['timestamp']:%Y-%m-%d %H:%M}")
    print("   schema:")
    for f in tbl.schema:
        print(f"     {f.name:<17} {str(f.type).replace('item: ', ''):<28} {'; '.join(roles[f.name])}".rstrip())
    for ix in indices:
        s = t.index_stats(ix.name)
        print(f"   index {ix.name}: {ix.index_type} on {', '.join(ix.columns)}, {s.num_indexed_rows} rows indexed, {s.num_unindexed_rows} unindexed")
    if not any(c in vec_cols for ix in indices for c in ix.columns):
        print("   no vector index: vector search scans all rows (exact kNN)")
    for f in emb:
        V = np.array([v for v in tbl[f["vector_column"]].to_pylist() if v is not None], dtype=np.float32)
        norms = np.linalg.norm(V, axis=1)
        longest = f"; longest input {max(toks[f['source_column']])} tokens (tokenizer limit {tok.model_max_length})" if f["source_column"] in toks else ""
        print(f"   embedding: {f['name']} {f['model'].get('name', '')} (device {f['model'].get('device')}), "
              f"{f['source_column']} -> {f['vector_column']} [{V.shape[1]}-d]{longest}")
        print(f"     {tbl[f['vector_column']].null_count} null, {int(np.isnan(V).any(axis=1).sum())} with NaN, L2 norm {norms.min():.2f}..{norms.max():.2f} "
              f"({'' if np.allclose(norms, 1, atol=1e-3) else 'not '}unit-normalised; LanceDB's default metric is L2)")
    print(f"   {len(docs)} documents, {min(docs.values())}..{max(docs.values())} rows each")
    if len(docs) <= 10:
        for d, n in docs.most_common():
            print(f"     {n:4d}  {d}")
    for c in cols:
        if c in (*TEXT_COLS, "id", "document", *vec_cols) or pa.types.is_list(tbl.schema.field(c).type):
            continue
        counts = collections.Counter(r[c] for r in rows)
        if len(counts) <= 20:
            print(f"   {c}: {dict(sorted(counts.items()))}")
    print(f"   {'lengths':<17} {'chars min / median / mean / max':<34} nomic tokens min / median / mean / max")
    for c, lens in toks.items():
        print(f"     {c:<15} {spread([len(r[c]) for r in rows]):<34} {spread(lens)}")
    if all(c in cols for c in TEXT_COLS):
        joined = sum(r["text"] == r["context"] + "\n\n" + r["original_text"] for r in rows)
        bare = sum(r["text"] == r["original_text"] for r in rows)
        print(f"   text = context + '\\n\\n' + original_text in {joined} rows, text = original_text in {bare}, other {len(rows) - joined - bare}; "
              f"empty context in {sum(not r['context'].strip() for r in rows)}")
    if "id" in cols:
        dup = {i: n for i, n in collections.Counter(r["id"] for r in rows).items() if n > 1}
        print(f"   ids shared by several rows: {len(dup)}")
        for i, n in dup.items():
            print(f"     {i[:16]}.. in {n} rows, text {clip(next(r['text'] for r in rows if r['id'] == i), 70)}")
    print("   example row:")
    for c, v in next((r for r in rows if r.get("context")), rows[0]).items():
        print(f"     {c:<17} {clip(v)}")

    def idx(keep):
        return ", ".join(f"{','.join(ix.columns)} ({ix.index_type} {ix.name})" for ix in indices if keep(ix))
    return {"name": name, "types": {f.name: str(f.type).replace("item: ", "") for f in tbl.schema},
            "dense (embedding)": "; ".join(f"{f['source_column']} -> {f['vector_column']}" for f in emb) or "none",
            "embedding model": ", ".join(f["model"].get("name", f["name"]) for f in emb) or "none",
            "keyword (FTS/BM25)": idx(lambda ix: ix.index_type == "FTS") or "none",
            "scalar indices": idx(lambda ix: ix.index_type != "FTS" and not set(ix.columns) & set(vec_cols)) or "none",
            "vector index": idx(lambda ix: set(ix.columns) & set(vec_cols)) or "none (exact kNN)",
            "rows / documents": f"{len(rows)} / {len(docs)}"}


def compare(a, b):
    print(f"== {a['name']} vs {b['name']}")
    print(f"   {'':<20} {a['name']:<34} {b['name']}")
    for key in dict.fromkeys([*a["types"], *b["types"]]):
        x, y = a["types"].get(key, "-"), b["types"].get(key, "-")
        print(f"   {key:<20} {x:<34} {y:<34} {'' if x == y else '<- differs'}".rstrip())
    for key in ("dense (embedding)", "embedding model", "keyword (FTS/BM25)", "scalar indices", "vector index", "rows / documents"):
        print(f"   {key:<20} {a[key]:<34} {b[key]:<34} {'' if a[key] == b[key] else '<- differs'}".rstrip())


a = describe(TABLE)
print()
b = describe(OTHER)
print()
compare(a, b)
