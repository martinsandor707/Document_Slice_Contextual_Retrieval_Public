"""Run router PROMPT variants over all 382 chunks (Ollama, gemma3:4b-it-qat) and save per-chunk letter log-probs.

Usage: python router_variants.py <variant> [...]   -> writes routes_<variant>.json  {id: {"probs": {letter: p}, "letter": L}}
The mapping letter -> k is applied later in the simulator, so one router run can be evaluated under several mappings.
"""
import glob
import json
import math
import os
import sys
import time
from typing import Dict, List

RND = "/home/martin/projects/TDK/Document_Slice_Contextual_Retrieval/RnD"
S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RND)
from utils.dynamic_slice_prediction import DynamicSlicePredictor, clip, default_few_shots, GENRE_SYSTEM_PROMPT as GENRE_PROMPT

ANCHOR_PROMPT = (
    "You help a search engine over scientific papers. Each passage below will be indexed together with a short "
    "sentence that situates it in its paper; you decide how much of the paper the writer of that sentence must read. "
    "Judge one thing only: does the passage ALONE already contain the words a reader would search for, i.e. the "
    "paper's subject, the named method, system, material or dataset, the concrete quantities, entities or findings it "
    "talks about?\n\n"
    "A - Anchored. The passage names its subject and specifics explicitly: an abstract; an introduction that states "
    "the problem and the approach; related work naming methods and authors; a dataset, hardware or setup description "
    "with proper names; a conclusion restating the contribution; also reference lists, acknowledgements and other "
    "boilerplate. A searcher would find it from its own words.\n"
    "B - Partly anchored. The passage carries real content (results, equations, procedures, discussion) but refers to "
    "its subject through generic words and stand-ins: 'the proposed method', 'the model', 'this scenario', 'Fig. 3', "
    "symbols, unexplained acronyms, 'where x is ...'. The neighbouring paragraphs are needed to know what it is about.\n"
    "C - Unanchored. A table of numbers, a list of bullet items, pseudo-code, a caption, a formula block, or a fragment "
    "cut mid-sentence; on its own it gives almost no clue what the paper is about.\n\n"
    "Reply with the letter only."
)

# Few-shot exemplars for the anchor scheme: same held-out Gaze chunks as the genre scheme, re-labelled by anchoring.
def anchor_few_shots():
    gaze = json.load(open(glob.glob(f"{RND}/split_documents/A_Hybrid_Gaze*")[0]))
    nat = json.load(open(glob.glob(f"{RND}/split_documents/s41586*")[0]))
    refs = "\n".join(gaze[13]["text"].split("\n")[:3])
    return [
        (clip(gaze[1]["text"], 900, 0), "A"),    # abstract -> anchored
        (clip(gaze[8]["text"], 700, 0), "B"),    # confidence-measure formulas with 'where' -> partly
        (clip(gaze[2]["text"], 650, 0), "A"),    # introduction (XR, VAC named) -> anchored
        (nat[12]["text"], "C"),                  # two bullet questions -> unanchored
        (refs, "A"),                             # reference list -> anchored (boilerplate)
        (clip(gaze[11]["text"], 600, 0), "B"),   # results pointing to Fig. 7, Eq. 1/3, segments -> partly
        (clip(gaze[5]["text"], 650, 0), "A"),    # method overview naming Pupil-Labs / RealSense -> anchored
        (clip(gaze[7]["text"], 600, 0), "B"),    # activation-function formulas -> partly
    ]


class Router(DynamicSlicePredictor):
    """DynamicSlicePredictor with an arbitrary letter set and optional zero-shot prompting."""

    def __init__(self, system_prompt, letters, shots=None, label="Answer", **kw):
        super().__init__(system_prompt=system_prompt, few_shots=[("x", 0)], letter_to_k={l: i for i, l in enumerate(letters)},
                         answer_label=label, **kw)
        self._raw_shots = shots or []
        self._letters = letters

    def build_messages(self, chunk_text):
        parts = [f"Passage:\n{t}\n{self.answer_label}: {l}\n" for t, l in self._raw_shots]
        parts.append(f"Passage:\n{clip(chunk_text, self.query_head, self.query_tail)}\n{self.answer_label}:")
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": "\n".join(parts)}]


def genre_router():
    return Router(GENRE_PROMPT, ["A", "B", "C", "D"], shots=[(s.text, "ABCD"[s.k]) for s in default_few_shots()], label="Genre")


GENRE3_PROMPT = (
    "Classify a passage taken from a scientific paper into one of three genres. Answer with a single letter.\n\n"
    "A - Whole-paper summary or boilerplate. The abstract; a conclusion or summary of the whole paper (these typically open "
    "with 'In this paper', 'In conclusion', 'In summary', 'This paper has proposed', 'This article has proposed', 'We have "
    "proposed'); or non-content matter: reference list, acknowledgements, funding statement, author biographies or "
    "contributions, competing-interests statement, publication metadata, licence text, layout garbage. A conclusion is A "
    "even when it is full of numbers, acronyms and technical terms.\n"
    "B - Self-contained exposition. Introduction, motivation, background or related work, or a description of a dataset, "
    "device, system, software or experimental setup that names its subject in full and is understandable without the rest "
    "of the paper. Contains no equation placeholders. Introductions and related-work paragraphs are B even when technical; "
    "they typically cite many references such as [3], [7]-[9] and describe what other authors did.\n"
    "C - Everything that depends on its surroundings. Methods, derivations, results or discussion that continue an argument "
    "of their section ('<!-- formula-not-decoded -->' placeholders with 'where ...' definitions, references to figures, "
    "tables, scenarios or equations, 'the proposed method', undefined acronyms or symbols, text that starts or stops "
    "mid-sentence), as well as fragments: a couple of bullet items, pseudo-code, a bare list of inputs, a dangling caption, "
    "a one- or two-sentence stub.\n\n"
    "Reply with the letter only."
)

BINARY_PROMPT = (
    "You are shown a passage taken from a scientific paper. Decide whether it describes itself. Answer with a single letter.\n\n"
    "S - Self-describing. The passage already tells the reader what the paper is about or needs no such information: the "
    "abstract; a conclusion or summary of the whole paper (typically opening with 'In this paper', 'In conclusion', 'In "
    "summary', 'This paper has proposed', 'We have proposed'); or non-content matter such as a reference list, "
    "acknowledgements, funding, author biographies or contributions, competing interests, publication metadata, licence text, "
    "layout garbage. A conclusion is S even when it is full of numbers and technical terms.\n"
    "N - Needs its neighbours. Any other passage: introduction, background, related work, methods, derivations, equations, "
    "experimental setup, results, discussion, figure or table captions, bullet lists, fragments.\n\n"
    "Reply with the letter only."
)

SPLIT5_PROMPT = (
    "Classify a passage taken from a scientific paper into one of five classes. Answer with a single letter.\n\n"
    "A - Abstract or conclusion. The paper's abstract, or a conclusion / summary of the whole paper (these typically open "
    "with 'In this paper', 'In conclusion', 'In summary', 'This paper has proposed', 'We have proposed'). A conclusion is A "
    "even when it is full of numbers, acronyms and technical terms.\n"
    "B - Boilerplate without scientific content: reference list, acknowledgements, funding statement, author biographies or "
    "contributions, competing-interests statement, publication metadata, licence text, layout garbage.\n"
    "C - Self-contained exposition. Introduction, motivation, background or related work, or a description of a dataset, "
    "device, system, software or experimental setup that names its subject in full and is understandable without the rest "
    "of the paper. Contains no equation placeholders; typically cites many references such as [3], [7]-[9].\n"
    "D - Section-internal technical body. Methods, derivations, results or discussion that continue an argument of their "
    "section: '<!-- formula-not-decoded -->' placeholders with 'where ...' definitions, references to figures, tables, "
    "scenarios or equations, 'the proposed method', undefined acronyms or symbols, a one- or two-sentence section stub, or "
    "text that starts or stops mid-sentence.\n"
    "E - Fragment. A couple of bullet items, pseudo-code, a bare list of inputs, a dangling caption, or a stub that only "
    "names undefined acronyms; meaningless without its surroundings.\n\n"
    "Reply with the letter only."
)


def relabelled_shots(mapping):
    """The module's ten genre exemplars (k-labelled) relabelled for another letter scheme; mapping: k -> letter."""
    return [(s.text, mapping[s.k]) for s in default_few_shots()]


def split5_shots():
    # genre exemplars carry k only; split the k=0 ones by content: abstract / conclusion -> A, references / contributions -> B
    out = []
    for s in default_few_shots():
        if s.k == 0:
            letter = "B" if (s.note.startswith("reference") or s.note.startswith("author")) else "A"
        else:
            letter = {1: "C", 2: "D", 3: "E"}[s.k]
        out.append((s.text, letter))
    return out


VARIANTS = {
    "genre": genre_router,                                                          # current module prompt (control)
    "anchor0": lambda: Router(ANCHOR_PROMPT, ["A", "B", "C"], shots=None),            # zero-shot
    "anchor8": lambda: Router(ANCHOR_PROMPT, ["A", "B", "C"], shots=anchor_few_shots()),
    # round 3: prompts for a router that never needs k=3 (tables are gated to k=1)
    "genre3": lambda: Router(GENRE3_PROMPT, ["A", "B", "C"], shots=relabelled_shots({0: "A", 1: "B", 2: "C", 3: "C"}), label="Genre"),
    "binary": lambda: Router(BINARY_PROMPT, ["S", "N"], shots=relabelled_shots({0: "S", 1: "N", 2: "N", 3: "N"}), label="Answer"),
    # same prompt; exemplars are whatever the module's default_few_shots() currently returns (held-out papers since the
    # exemplar re-sourcing) - saved under a separate name so it can be compared with routes_binary.json (old exemplars)
    "binary_heldout": lambda: Router(BINARY_PROMPT, ["S", "N"], shots=relabelled_shots({0: "S", 1: "N", 2: "N", 3: "N"}), label="Answer"),
    "split5": lambda: Router(SPLIT5_PROMPT, ["A", "B", "C", "D", "E"], shots=split5_shots(), label="Class"),
}


def run(variant: str):
    router = VARIANTS[variant]()
    chunks = [c for p in sorted(glob.glob(f"{RND}/split_documents/*.json")) for c in json.load(open(p))]
    out: Dict[str, dict] = {}
    t0 = time.time()
    n_gate = 0
    for c in chunks:
        if c["id"] in out:
            continue
        from utils.dynamic_slice_prediction import table_evidence
        if table_evidence(c["text"]):
            out[c["id"]] = {"letter": "T", "probs": {}, "table": True}
            n_gate += 1
            continue
        scores = router.slm_scores(c["text"])          # {idx: logprob}
        z = {router._letters[i]: v for i, v in scores.items()}
        m = max(z.values()); probs = {l: math.exp(v - m) for l, v in z.items()}; s = sum(probs.values())
        probs = {l: p / s for l, p in probs.items()}
        out[c["id"]] = {"letter": max(probs, key=probs.get), "probs": probs, "table": False}
    json.dump(out, open(f"{S}/routes_{variant}.json", "w"))
    from collections import Counter
    print(f"{variant:10s}: {len(out)} ids in {time.time()-t0:.0f}s | letters {dict(Counter(v['letter'] for v in out.values()))} | gated {n_gate}", flush=True)


if __name__ == "__main__":
    for v in sys.argv[1:]:
        run(v)
