"""Verifies the corpus statistics of the Springer manuscript (sn_green_rag_article.tex):
  Table 1 (tab:parsing) rows 1-4 and 7-11: documents, chunks, distinct ids, chunks per document, character
    quartiles, characters per token, document lengths, formula placeholders, glyph codes, table-gated chunks;
  Table 4 (tab:corpora): chunks, characters, median chunk length, question statistics for both corpora,
    exemplar pool size;  Sect. 3.1 counts (51 / 93 / 42) and the introduction's corpus description.
Run from the repository root:  .venv/bin/python RnD/verification/chunk_character_statistics.py
Read-only: only reads RnD/split_documents*, RnD/q_and_a/Gemini/*.json and, when the exemplar pool is absent (public
copy), RnD/dynamic_slice_length/router_exemplars_heldout.json.  Token totals come from
chunk_token_statistics.py (253 488 scientific tokens)."""
import statistics, sys, collections, json
from _common import RND, SPLIT, SPLIT_POLICY, SPLIT_ROUTER, QNA_SCI, QNA_POLICY, EXEMPLARS, load_docs
sys.path.insert(0, str(RND))
from utils.dynamic_slice_prediction import table_evidence   # the Tier-1 gate itself

def describe(name, docs):
    lens = [len(c["text"]) for ch in docs.values() for c in ch]
    per_doc = {n: len(ch) for n, ch in docs.items()}
    q = statistics.quantiles(lens, n=4, method="inclusive")
    ids = [c["id"] for ch in docs.values() for c in ch]
    print(f"== {name}: {len(docs)} documents, {len(lens)} chunks, {len(set(ids))} distinct ids")
    print(f"   chunks per document min/median/mean/max = {min(per_doc.values())}/{statistics.median(per_doc.values()):.0f}/{statistics.mean(per_doc.values()):.2f}/{max(per_doc.values())}")
    print(f"   chunk chars min/Q1/median/mean/Q3/max = {min(lens)}/{q[0]:.0f}/{statistics.median(lens):.0f}/{statistics.mean(lens):.0f}/{q[2]:.0f}/{max(lens)}   total {sum(lens)}")
    dl = {n: sum(len(c["text"]) for c in ch) for n, ch in docs.items()}
    print(f"   document chars min/median/max = {min(dl.values())}/{statistics.median(dl.values()):.0f}/{max(dl.values())}  (longest: {max(dl, key=dl.get)})")
    print(f"   chunks with '<!-- formula-not-decoded -->': {sum('<!-- formula-not-decoded -->' in c['text'] for ch in docs.values() for c in ch)}")
    print(f"   chunks with 'GLYPH<': {sum('GLYPH<' in c['text'] for ch in docs.values() for c in ch)}")
    print(f"   chunks firing the Tier-1 table gate (table_evidence): {sum(table_evidence(c['text']) is not None for ch in docs.values() for c in ch)}")
    dup = [i for i, n in collections.Counter(ids).items() if n > 1]
    for i in dup:
        print(f"   duplicate id {i[:12]}...: {[(n, k) for n, ch in docs.items() for k, c in enumerate(ch) if c['id'] == i]}  text={next(c['text'] for ch in docs.values() for c in ch if c['id']==i)!r}")
    return per_doc, sum(lens)

def questions(name, path, docs):
    qna = json.load(open(path, encoding="utf-8"))
    sup = [len(q["supporting_chunks"]) for q in qna]
    gold = {c for q in qna for c in q["supporting_chunks"]}
    corpus_ids = {c["id"] for ch in docs.values() for c in ch}
    per_doc = collections.Counter(q["document"] for q in qna)
    print(f"== {name} questions: {len(qna)}; per document {dict(per_doc)}")
    print(f"   supporting chunks per question min/mean/max = {min(sup)}/{statistics.mean(sup):.2f}/{max(sup)}; distribution {dict(sorted(collections.Counter(sup).items()))}")
    print(f"   gold references {sum(sup)}; distinct gold chunks {len(gold)} = {len(gold)/len(corpus_ids):.1%} of {len(corpus_ids)} corpus ids; unresolvable ids {len(gold - corpus_ids)}")

sci = load_docs(SPLIT); pol = load_docs(SPLIT_POLICY); rou = load_docs(SPLIT_ROUTER)
per_doc, total = describe("scientific benchmark (RnD/split_documents)", sci)
print(f"   characters per nomic token = {total} / 253488 = {total/253488:.3f}   (token total from chunk_token_statistics.py)")
print("   chunks per document:", per_doc)
describe("EC policy corpus (RnD/split_documents/smoking)", pol)
if rou:
    describe("router exemplar pool (RnD/split_documents_router)", rou)
else:
    # Public copy: the six held-out papers are not redistributable, so their chunk files are absent.  The pool's statistics
    # as this script printed them on the archived files (2026-10-13, before the removal) are repeated here for Table 4.
    shots = json.load(open(EXEMPLARS, encoding="utf-8"))
    sl = [len(s["text"]) for s in shots]
    print("== router exemplar pool (RnD/split_documents_router): absent in the public copy (the six held-out papers are not redistributable)")
    print("   archived pool: 6 documents, 109 chunks, 109 distinct ids; chunks per document min/median/mean/max = 9/15/18.17/30")
    print("   chunk chars min/Q1/median/mean/Q3/max = 17/1022/2995/3293/5144/9772   total 358891; document chars min/median/max = 31076/61761/96504")
    print("   chunks with '<!-- formula-not-decoded -->': 15; chunks with 'GLYPH<': 0; chunks firing the Tier-1 table gate: 9")
    print(f"   the ten exemplar passages shipped in {EXEMPLARS.relative_to(RND.parent)}: {len(shots)} passages, k distribution {dict(sorted(collections.Counter(s['k'] for s in shots).items()))}, chars min/median/max = {min(sl)}/{statistics.median(sl):.0f}/{max(sl)}, total {sum(sl)}")
questions("scientific", QNA_SCI, sci)
questions("policy", QNA_POLICY, pol)
