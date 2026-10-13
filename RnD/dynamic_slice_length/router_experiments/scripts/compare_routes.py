"""Compare two router passes over the benchmark chunks (letters + probabilities). Usage: python compare_routes.py old new"""
import json
import os
import sys
from collections import Counter

W = os.path.dirname(os.path.abspath(__file__))
RND = "/home/martin/projects/TDK/Document_Slice_Contextual_Retrieval/RnD"
a = json.load(open(os.path.join(W, f"routes_{sys.argv[1]}.json")))
b = json.load(open(os.path.join(W, f"routes_{sys.argv[2]}.json")))
text = {}
import glob
for f in glob.glob(f"{RND}/split_documents/*.json"):
    for c in json.load(open(f)):
        text[c["id"]] = c["text"]
ids = [i for i in a if i in b and a[i]["letter"] != "T"]
agree = sum(1 for i in ids if a[i]["letter"] == b[i]["letter"])
print(f"non-table chunks compared: {len(ids)} | same letter: {agree} ({agree/len(ids):.1%})")
print(f"letters {sys.argv[1]}: {dict(Counter(a[i]['letter'] for i in a))} | {sys.argv[2]}: {dict(Counter(b[i]['letter'] for i in b))}")
flips = Counter((a[i]["letter"], b[i]["letter"]) for i in ids if a[i]["letter"] != b[i]["letter"])
print("flips (old -> new):", dict(flips))
# confidence of the flipped chunks
for i in ids:
    if a[i]["letter"] != b[i]["letter"]:
        pa = max(a[i]["probs"].values()); pb = max(b[i]["probs"].values())
        print(f"  {a[i]['letter']}->{b[i]['letter']}  p_old {pa:.2f} p_new {pb:.2f} | {len(text[i]):5d} ch | {text[i][:95].replace(chr(10),' ')!r}")
# gold chunks affected
qna = json.load(open(f"{RND}/q_and_a/Gemini/scientific_multi_chunk_control.json"))
gold = {c for q in qna for c in q["supporting_chunks"]}
gf = [i for i in ids if a[i]["letter"] != b[i]["letter"] and i in gold]
print(f"flipped chunks that are gold supporting chunks: {len(gf)} (of {len(gold)} gold chunks)")
