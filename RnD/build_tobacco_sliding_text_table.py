"""Build a NEW LanceDB table for the policy (tobacco) corpus from the stored k=3 document-slice contexts, embedding the
ENRICHED text (context + blank line + chunk) exactly as every scientific-corpus table and the policy baseline do.

Background: the existing `tobacco_sliding_table` was built with `original_text` as the embedding source, so its
summaries reached only the BM25 index and the side-by-side comparison with `tobacco_traditional_table` (which embeds
`text`) was not like for like. This script leaves that table untouched and creates `tobacco_sliding_text_table`.

Input (frozen, read-only): RnD/preprocessed_chunks/tobacco_sliding.json (141 records; text == context + "\\n\\n" + original_text).
Output: RnD/db/tobacco_sliding_text_table (schema and indexes as in RnD/ablation_dynamic_doc_slice_generation.ipynb cell 5).
GPU: loads nomic-embed-text-v1.5 on CUDA; do not run while an Ollama model is resident (8 GB card).
Run from the repository root:  .venv/bin/python RnD/build_tobacco_sliding_text_table.py [--overwrite]"""
import json, os, sys, torch, lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

RND = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(RND, "preprocessed_chunks", "tobacco_sliding.json")
TABLE_NAME = "tobacco_sliding_text_table"
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

records = json.load(open(SOURCE, encoding="utf-8"))
assert all(r["text"] == r["context"] + "\n\n" + r["original_text"] for r in records), "unexpected text layout"
db = lancedb.connect(os.path.join(RND, "db"))
if TABLE_NAME in db.table_names() and "--overwrite" not in sys.argv:
    sys.exit(f"{TABLE_NAME} already exists in RnD/db; pass --overwrite to rebuild it")

hf = get_registry().get("huggingface").create(name=EMBEDDING_MODEL_NAME, trust_remote_code=True,
                                              device="cuda" if torch.cuda.is_available() else "cpu")

class PolicyDocument(LanceModel):
    text: str = hf.SourceField()          # <- enriched text is embedded (summary + chunk), as in every other comparison table
    vector: Vector(hf.ndims()) = hf.VectorField()
    original_text: str
    context: str
    document: str
    id: str
    doc_slice_radius: int

db.create_table(TABLE_NAME, schema=PolicyDocument, mode="overwrite")
table = db.open_table(TABLE_NAME)
rows = [{"text": r["text"], "original_text": r["original_text"], "context": r["context"], "document": r["document"],
         "id": r["id"], "doc_slice_radius": 3} for r in records]
for i in range(0, len(rows), 100):
    table.add(rows[i:i + 100])
table.create_scalar_index("id", replace=True)
table.create_fts_index("text", replace=True)
table.wait_for_index(["text_idx"])
meta = table.schema.metadata or {}
print(f"built {TABLE_NAME}: {table.count_rows()} rows; embedding source column: "
      f"{json.loads(meta[b'embedding_functions'])[0]['source_column']}")
