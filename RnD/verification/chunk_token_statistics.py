"""Verifies the token-length rows of the Springer manuscript:
  Table 1 (tab:parsing) rows 5-6: chunk tokens min/Q1/median/mean/Q3/max, total, s.d., chunks per token band,
    chunks at >= 1 900 tokens, none above 2 000;
  Table 4 (tab:corpora): 'Chunk length, tokens (mean / median)' for the scientific (663.6 / 450) and the final
    141-chunk policy corpus (446.8 / 246);  Sect. 3.5: the token median used for the 2 890-token typical window.
The call is the one cell 2 of RnD/doc_slice_chunking.ipynb used (tokenizer.tokenize, no special tokens); the
scientific total of 253 488 tokens reproduces that cell's printed mean 663.5811518324607.
Run from the repository root:  .venv/bin/python RnD/verification/chunk_token_statistics.py   (CPU only, offline)"""
import statistics
from _common import SPLIT, SPLIT_POLICY, SPLIT_ROUTER, load_docs, nomic_tokenizer

tok = nomic_tokenizer()
BANDS = [(0, 250), (251, 500), (501, 1000), (1001, 1500), (1501, 2000)]
for name, split in [("scientific benchmark", SPLIT), ("EC policy corpus (final 141 chunks)", SPLIT_POLICY), ("router exemplar pool", SPLIT_ROUTER)]:
    docs = load_docs(split)
    if not docs:   # public copy: the six held-out papers of the exemplar pool are not redistributable; archived values (2026-10-13)
        print(f"== {name}: absent in the public copy; archived run: n=109 total=89704 mean=822.9725 min=3 Q1=234 median=670 Q3=1477 max=1992 sd=676.6")
        print(f"   bands {[f'{a}-{b}' for a, b in BANDS]} = [31, 18, 25, 8, 27]; >=1900: 12; >2000: 0; with the two special tokens: mean 825.0")
        continue
    lens = [len(tok.tokenize(c["text"])) for ch in docs.values() for c in ch]
    q = statistics.quantiles(lens, n=4, method="inclusive")
    print(f"== {name}: n={len(lens)} total={sum(lens)} mean={statistics.mean(lens):.4f} min={min(lens)} Q1={q[0]:.0f} median={statistics.median(lens):.0f} Q3={q[2]:.0f} max={max(lens)} sd={statistics.pstdev(lens):.1f}")
    print(f"   bands {[f'{a}-{b}' for a, b in BANDS]} = {[sum(a <= l <= b for l in lens) for a, b in BANDS]}; >=1900: {sum(l >= 1900 for l in lens)}; >2000: {sum(l > 2000 for l in lens)}")
    enc = [len(tok.encode(c["text"])) for ch in docs.values() for c in ch]
    print(f"   with the two special tokens (tokenizer.encode): mean {statistics.mean(enc):.1f}")
