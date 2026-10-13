"""Dynamic document-slice radius prediction (k in {0, 1, 2, 3}) for the Document Slice contextual retrieval pipeline.

Production counterpart of ``RnD/ollama_dynamic_slice_length.ipynb``.  The prompts, exemplars, clipping and decision
rule are *identical* to the notebook; everything is exposed as configuration so the Ollama model, the few-shot
exemplars, the clip lengths or the calibration bias can be swapped without touching the code.

Two tiers, both cheap (Green AI):

* **Tier 1 - table gate** (host CPU, regex, ~0 ms).  Docling's HybridChunker serialises a table as one line of
  ``<row>, <column> = <value>.`` cells and emits captions on their own line.  Any chunk that carries a table gets
  ``k = TABLE_K`` without calling the model.  ``TABLE_K`` is **2** (3 in the first version, 1 in the confirmation
  runs): with a wide slice the summariser describes the prose around the table or degenerates into a system-prompt
  example, with a narrow slice it describes the table itself.  Whether 1 or 2 is better depends on the summary
  sample; 3 lost in every measurement (RnD/router_prompt_engineering.md).
* **Tier 2 - single forward pass** of the SLM with ``num_predict = 1``.  The module default (round 3 of the router
  study) is a **2-class router**: S = self-describing (abstract, conclusion, boilerplate) -> ``k = 0`` (no summary),
  N = needs its neighbours -> ``k = 2``.  The 4-class **genre router** used in the confirmation runs (A whole-paper
  summary / boilerplate -> 0, B self-contained exposition -> 1, C section-internal technical body -> 2, D fragment -> 3)
  is kept under the ``GENRE_*`` names.  The decision is the argmax over the letter tokens read from Ollama's log-probs,
  so parsing can never fail; if log-probs are unavailable the emitted token is parsed, and if that fails too
  ``k_default`` (2, "if unsure use 2") is returned.

Semantics of k: 0 = the chunk is already a summary or boilerplate -> skip the contextualisation call (free for
recall, best MRR/MAR, 27 % fewer summariser calls); 1 = light context (also tables); 2 = default; 3 = widest window
(fragments only).  Widening prose classes to 3 or narrowing technical body text to 1 both lowered recall; the
alternative "topical anchoring" prompt classified worse than the genre prompt.  Full measurements and the
reasoning are in ``RnD/router_prompt_engineering.md``.  ``FIXED_K`` (module constant / predictor field) bypasses
the router and returns a constant radius for control runs.

Determinism: ``temperature = 0`` makes the argmax deterministic for a given set of logits, but Ollama's logits carry a
little numerical noise depending on whether the shared prompt prefix came from its KV cache or was recomputed
(observed: ~0.1-0.3 nats on the first call after a cache change).  Chunks whose top two letters are that close can
therefore flip between runs (2 of 119 gold chunks did).  ``tie_margin`` (off by default) resolves such near-ties to
``k_default``; ``predict_with_details`` exposes the letter log-probs so the margin can be inspected.

Usage::

    from utils.dynamic_slice_prediction import predict
    k = predict(chunk_text)                                   # notebook defaults (gemma3:4b-it-qat)
    k = predict(chunk_text, model="qwen3.5:4b")               # another Ollama model
    k = predict(chunk_text, few_shots=[(text, 0), (text, 2)]) # custom exemplars, (text, k) pairs
    k = predict(chunk_text, bias=NOTEBOOK_CALIBRATION_BIAS)   # optional contextual calibration
    k = predict(chunk_text, tie_margin=0.2)                   # near-ties involving k=2 resolve to 2

or build a reusable predictor once::

    predictor = DynamicSlicePredictor(model="gemma3:4b-it-qat", few_shots=few_shots_from_json("my_shots.json"))
    k = predictor.predict(chunk_text)

Requires a running Ollama server (``ollama serve``) with the model pulled.  The default exemplars are read from the
frozen Docling chunk JSONs of the six held-out papers in ``RnD/split_documents_router`` (override ``SPLIT_DIR`` or the
``DSL_SPLIT_DIR`` environment variable if they live elsewhere, or pass ``few_shots`` explicitly).  When that directory
is absent, as in the public repository, the ten exemplar passages are loaded verbatim from
``RnD/dynamic_slice_length/router_exemplars_heldout.json`` (``EXEMPLARS_JSON``, override with ``DSL_EXEMPLARS_JSON``).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import ollama

__all__ = [
    "predict",
    "predict_with_details",
    "DynamicSlicePredictor",
    "FewShot",
    "default_few_shots",
    "few_shots_from_json",
    "few_shots_to_json",
    "is_structural_table",
    "table_evidence",
    "clip",
    "SYSTEM_PROMPT",
    "BINARY_SYSTEM_PROMPT",
    "GENRE_SYSTEM_PROMPT",
    "LETTER_TO_K",
    "K_TO_LETTER",
    "GENRE_LETTER_TO_K",
    "GENRE_ANSWER_LABEL",
    "genre_few_shots",
    "binary_few_shots",
    "TABLE_K",
    "FIXED_K",
    "NOTEBOOK_CALIBRATION_BIAS",
    "SPLIT_DIR",
    "EXEMPLARS_JSON",
]

# =====================================================================================================================
# Defaults (identical to the notebook)
# =====================================================================================================================
OLLAMA_MODEL_NAME = "gemma3:4b-it-qat"          # same model as the contextualiser -> nothing extra in VRAM
K_MAX = 3
K_DEFAULT = 2                                   # fallback when nothing can be parsed ("if unsure, 2")
QUERY_HEAD, QUERY_TAIL = 1500, 500              # characters of the chunk shown to the SLM: head [...] tail
ANSWER_LABEL = "Answer"                         # answer prefix in the prompt; the model is sensitive to this word
GENRE_ANSWER_LABEL = "Genre"                    # answer prefix of the 4-class genre router
OLLAMA_OPTIONS: Dict[str, Any] = {"temperature": 0.0, "num_predict": 1, "top_p": 1.0}
TOP_LOGPROBS = 20

#: Docling chunk JSONs the few-shot exemplars are taken from.  These are the six HELD-OUT papers of ``RnD/input_router``
#: (chunked with the corpus recipe into ``RnD/split_documents_router``), which are not part of the retrieval benchmark;
#: the benchmark corpus itself (``RnD/split_documents``) must never be used for exemplars.
SPLIT_DIR = Path(os.environ.get("DSL_SPLIT_DIR", Path(__file__).resolve().parent.parent / "split_documents_router"))

#: The ten exemplar passages exported from those chunk files with ``few_shots_to_json(default_few_shots(), ...)``.  The six
#: exemplar papers are not redistributable, so the public repository ships this file instead of ``split_documents_router``;
#: ``default_few_shots()`` falls back to it when ``split_dir`` holds no chunk JSON.
EXEMPLARS_JSON = Path(os.environ.get("DSL_EXEMPLARS_JSON",
                                     Path(__file__).resolve().parent.parent / "dynamic_slice_length" / "router_exemplars_heldout.json"))

# ---- routing policy (module defaults, used by the notebooks) ------------------------------------------------------
# Round-3 default: the 2-class router.  S = self-describing (abstract, conclusion, boilerplate) -> k = 0 (no summary),
# N = needs its neighbours -> k = 2; Tier-1 tables -> k = 2.  Offline estimate on real summaries of this pipeline:
# R@20 0.9507 (fixed k=2: 0.9467); see RnD/router_prompt_engineering.md, section 4b.
LETTER_TO_K: Dict[str, int] = {"S": 0, "N": 2}
K_TO_LETTER: Dict[int, str] = {v: k for k, v in LETTER_TO_K.items()}
TABLE_K = 2                                     # radius assigned by the Tier-1 table gate (3 in the first version, 1 in the confirmation runs)
# The 4-class genre router of the confirmation runs (A -> 0, B -> 1, C -> 2, D -> 3, TABLE_K = 1) is kept selectable:
# DynamicSlicePredictor(system_prompt=GENRE_SYSTEM_PROMPT, letter_to_k=GENRE_LETTER_TO_K, answer_label=GENRE_ANSWER_LABEL,
#                       few_shots=genre_few_shots(), table_k=1)
GENRE_LETTER_TO_K: Dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}
FIXED_K: Optional[int] = None                   # experiment switch: if set, every chunk gets this radius (no gate, no SLM call)

#: Additive log-prob bias fitted on the gold set in the notebook (class 0 anchored).  Off by default: its
#: leave-one-document-out gain was ~+3 points and it pushes away from k = 2, against the paper's prior.
NOTEBOOK_CALIBRATION_BIAS: Tuple[float, float, float, float] = (0.0, 0.308, -2.802, 0.023)

BINARY_SYSTEM_PROMPT = (
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

GENRE_SYSTEM_PROMPT = (
    "Classify a passage taken from a scientific paper into one of four genres. Answer with a single letter.\n\n"
    "A - Whole-paper summary or boilerplate. The abstract; a conclusion or summary of the whole paper (these typically open "
    "with 'In this paper', 'In conclusion', 'In summary', 'This paper has proposed', 'This article has proposed', 'We have "
    "proposed'); or non-content matter: reference list, acknowledgements, funding statement, author biographies or "
    "contributions, competing-interests statement, publication metadata, licence text, layout garbage. A conclusion is A "
    "even when it is full of numbers, acronyms and technical terms.\n"
    "B - Self-contained exposition. Introduction, motivation, background or related work, or a description of a dataset, "
    "device, system, software or experimental setup that names its subject in full and is understandable without the rest "
    "of the paper. Contains no equation placeholders. Introductions and related-work paragraphs are B even when technical; "
    "they typically cite many references such as [3], [7]-[9] and describe what other authors did.\n"
    "C - Section-internal technical body. Methods, derivations, results or discussion that continue an argument of their "
    "section: '<!-- formula-not-decoded -->' placeholders with 'where ...' definitions, references to figures, tables, "
    "scenarios or equations, 'the proposed method', undefined acronyms or symbols, a one- or two-sentence section stub "
    "('This section presents ...'), or text that starts or stops mid-sentence.\n"
    "D - Fragment. A couple of bullet items, pseudo-code, a bare list of inputs, a dangling caption, or a stub that only "
    "names undefined acronyms; meaningless without its surroundings.\n\n"
    "Reply with the letter only."
)

SYSTEM_PROMPT = BINARY_SYSTEM_PROMPT            # the module default

# =====================================================================================================================
# Tier 1 - deterministic table gate
# =====================================================================================================================
_CELL_RE = re.compile(r", [^,=\n]{1,80} = [^=\n]{0,80}\S\.(?: |$)")
_CAPTION_RE = re.compile(
    r"(?m)^[ \t]*(?:TABLE|Table)[ \t]+(?:\d{1,2}|[IVXLC]{1,5})\b"
    r"(?:[ \t]*[.:|][ \t]*\S"      # 'TABLE 1. Title' | 'Table 1: Title' | 'Table 1 | Title'
    r"|[ \t]+[A-Z][A-Za-z]"        # 'TABLE I SAMPLE BINS ...' | 'Table 1 Profits and ...'
    r"|[ \t]*$)"                   # 'TABLE I' alone, all-caps title on the next line (IEEE Trans.)
)
_PIPE_CAPTION_RE = re.compile(r"\bTable[ \t]+\d{1,2}[ \t]*\|")   # Nature style, may be glued to the first cell


def table_evidence(text: str) -> Optional[str]:
    """Return a short human-readable reason if ``text`` contains a Docling-serialised table, else ``None``.

    Fires on (1) three or more ``<row>, <column> = <value>.`` cells on one line with at most 100 characters per
    ``=``, or (2) a table caption at the start of a line.  In-text mentions ("... shown in Table 1.", "Table 1 shows")
    and maths prose ("V = { 1, . . . , m }", "gx = 2.0087(3), gy = ...") do not match.
    """
    for line in text.split("\n"):
        cells = _CELL_RE.findall(line)
        if len(cells) >= 3 and len(line) / max(1, line.count(" = ")) <= 100:
            return f"{len(cells)} serialised cells on one line: {line[:60]!r}"
    m = _CAPTION_RE.search(text)
    if m:
        return f"caption: {text[m.start():m.end() + 40].splitlines()[0]!r}"
    m = _PIPE_CAPTION_RE.search(text)
    if m:
        return f"pipe caption: {text[m.start():m.start() + 40]!r}"
    return None


def is_structural_table(text: str) -> bool:
    """True if the chunk carries a table (caption and/or serialised cells) -> Tier 1 assigns ``k = TABLE_K``."""
    return table_evidence(text) is not None


# =====================================================================================================================
# Tier 2 - prompt construction
# =====================================================================================================================
def clip(text: str, head: int = QUERY_HEAD, tail: int = QUERY_TAIL) -> str:
    """Show the first ``head`` and last ``tail`` characters of a chunk (``tail = 0`` -> head only)."""
    text = text.strip()
    if len(text) <= head + tail + 20:
        return text
    return text[:head].rstrip() + (" [...] " + text[-tail:].lstrip() if tail else " [...]")


@dataclass(frozen=True)
class FewShot:
    """One in-context exemplar: the text exactly as it is shown to the model and its gold radius ``k``."""
    text: str
    k: int
    note: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.k <= K_MAX:
            raise ValueError(f"few-shot k must be in 0..{K_MAX}, got {self.k}")


FewShotLike = Sequence[Any]  # FewShot | (text, k) | (text, k, note) | {"text":..., "k":...}


def _coerce_few_shots(shots: FewShotLike) -> Tuple[FewShot, ...]:
    out: List[FewShot] = []
    for s in shots:
        if isinstance(s, FewShot):
            out.append(s)
        elif isinstance(s, Mapping):
            out.append(FewShot(str(s["text"]), int(s["k"]), str(s.get("note", ""))))
        else:
            text, k, *rest = s
            out.append(FewShot(str(text), int(k), str(rest[0]) if rest else ""))
    if not out:
        raise ValueError("at least one few-shot exemplar is required")
    return tuple(out)


def _load_doc(pattern: str, split_dir: Path = SPLIT_DIR) -> List[dict]:
    matches = sorted(glob.glob(str(Path(split_dir) / pattern)))
    if not matches:
        raise FileNotFoundError(
            f"no chunk JSON matching {pattern!r} in {split_dir} - set SPLIT_DIR / DSL_SPLIT_DIR or pass few_shots explicitly"
        )
    with open(matches[0], encoding="utf-8") as f:
        return json.load(f)


def default_few_shots(split_dir: Path = SPLIT_DIR) -> Tuple[FewShot, ...]:
    """Ten exemplars (k in 0..3) taken verbatim from Docling chunks of the held-out papers in ``split_dir``, or loaded from
    ``EXEMPLARS_JSON`` when ``split_dir`` holds no chunk JSON (the public repository ships the passages, not the papers);
    the routers relabel them through genre_few_shots() / binary_few_shots().  Roles, clip lengths and order are those of the original
    exemplar set (which came from benchmark papers and was replaced for test-set hygiene):
    abstract | formulas with 'where' | related work | fragment | references | results pointing to a figure |
    method overview naming the device | conclusion | experimental setup | acknowledgement."""
    if not glob.glob(str(Path(split_dir) / "*.json")):
        if Path(EXEMPLARS_JSON).is_file():
            return few_shots_from_json(EXEMPLARS_JSON)
        raise FileNotFoundError(
            f"no chunk JSON in {split_dir} and no exemplar file at {EXEMPLARS_JSON} - set SPLIT_DIR / DSL_SPLIT_DIR or "
            f"DSL_EXEMPLARS_JSON, or pass few_shots explicitly"
        )
    tan = _load_doc("tan2023*", split_dir)                   # IEEE survey on personalised federated learning
    srcnn = _load_doc("Image_Super-Resolution*", split_dir)  # IEEE TPAMI, SRCNN
    arute = _load_doc("arute2019*", split_dir)               # Nature, quantum supremacy (Sycamore)
    silver = _load_doc("silver2017*", split_dir)             # Nature, AlphaGo Zero
    lempel = _load_doc("lempel1976*", split_dir)             # IEEE Trans. Inf. Theory, Lempel-Ziv complexity
    refs = "\n".join(srcnn[25]["text"].split("\n")[:3])
    return (
        FewShot(clip(tan[0]["text"], 900, 0), 0, "author line + Abstract"),
        FewShot(clip(srcnn[10]["text"], 700, 0), 2, "MSE loss formula with 'where n is ...'"),
        FewShot(clip(srcnn[4]["text"], 650, 0), 1, "related work, methods named (MLP, CNN denoising, Cui et al.)"),
        FewShot(silver[14]["text"], 3, "dangling figure caption with layout junk (148 chars)"),
        FewShot(refs, 0, "reference list excerpt"),
        FewShot(clip(srcnn[22]["text"], 600, 0), 2, "results pointing to Fig. 12, running-time comparison"),
        FewShot(clip(arute[3]["text"], 650, 0), 1, "device description: the Sycamore processor"),
        FewShot(clip(srcnn[24]["text"], 700, 0), 0, "conclusion 'We have presented ...'"),
        FewShot(clip(srcnn[12]["text"], 600, 0), 1, "experimental setup: training sets, network settings"),
        FewShot(lempel[7]["text"], 0, "acknowledgement (88 chars)"),
    )


def genre_few_shots(split_dir: Path = SPLIT_DIR) -> Tuple[FewShot, ...]:
    """Exemplars for the 4-class genre router (k in 0..3, letters A-D)."""
    return default_few_shots(split_dir)


def binary_few_shots(split_dir: Path = SPLIT_DIR) -> Tuple[FewShot, ...]:
    """The same ten exemplars relabelled for the 2-class router: k = 0 stays S, everything else becomes N (k = 2)."""
    return tuple(FewShot(s.text, 0 if s.k == 0 else 2, s.note) for s in default_few_shots(split_dir))


DEFAULT_FEW_SHOTS = binary_few_shots              # exemplar set that matches the module-default prompt / letters


def few_shots_from_json(path: os.PathLike | str) -> Tuple[FewShot, ...]:
    """Load exemplars from a JSON list of ``{"text": ..., "k": ..., "note": ...}`` objects (text is shown verbatim)."""
    with open(path, encoding="utf-8") as f:
        return _coerce_few_shots(json.load(f))


def few_shots_to_json(shots: FewShotLike, path: os.PathLike | str) -> None:
    """Save exemplars (e.g. ``default_few_shots()``) so they can be edited and reloaded with ``few_shots_from_json``."""
    data = [{"text": s.text, "k": s.k, "note": s.note} for s in _coerce_few_shots(shots)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


# =====================================================================================================================
# Predictor
# =====================================================================================================================
@dataclass
class DynamicSlicePredictor:
    """Configurable two-tier radius predictor.  The defaults are the module-level policy (2-class router, tables -> 2);
    every field can be overridden per instance, e.g. to run the 4-class genre router (see the GENRE_* constants)."""

    model: str = OLLAMA_MODEL_NAME
    system_prompt: str = SYSTEM_PROMPT
    few_shots: Optional[FewShotLike] = None            # None -> DEFAULT_FEW_SHOTS(split_dir) (loaded lazily, matches the default prompt)
    query_head: int = QUERY_HEAD
    query_tail: int = QUERY_TAIL
    answer_label: str = ANSWER_LABEL
    letter_to_k: Mapping[str, int] = field(default_factory=lambda: dict(LETTER_TO_K))
    k_default: int = K_DEFAULT
    bias: Optional[Sequence[float]] = None             # additive log-prob bias per k (contextual calibration)
    tie_margin: float = 0.0                            # >0: if the best and second-best letters are within this many
                                                       # log-prob units and one of them is k_default, return k_default
                                                       # ("if unsure, 2"); 0.0 = plain argmax as in the notebook
    table_gate: bool = True                            # Tier 1 on/off
    table_k: int = TABLE_K                             # radius returned by the table gate
    fixed_k: Optional[int] = FIXED_K                   # if set: constant radius for every chunk (ablation / control runs)
    options: Dict[str, Any] = field(default_factory=lambda: dict(OLLAMA_OPTIONS))
    top_logprobs: int = TOP_LOGPROBS
    host: Optional[str] = None                         # Ollama host, e.g. "http://localhost:11434"
    client: Optional[ollama.Client] = None             # or a pre-built client (takes precedence over host)
    split_dir: Path = SPLIT_DIR                        # where default exemplars are read from

    # ---------------------------------------------------------------------------------------------- helpers
    def _client(self) -> ollama.Client:
        if self.client is None:
            self.client = ollama.Client(host=self.host) if self.host else ollama.Client()
        return self.client

    def _shots(self) -> Tuple[FewShot, ...]:
        if self.few_shots is None:
            self.few_shots = DEFAULT_FEW_SHOTS(self.split_dir)
        elif not (isinstance(self.few_shots, tuple) and all(isinstance(s, FewShot) for s in self.few_shots)):
            self.few_shots = _coerce_few_shots(self.few_shots)
        return self.few_shots  # type: ignore[return-value]

    @property
    def k_to_letter(self) -> Dict[int, str]:
        return {v: k for k, v in self.letter_to_k.items()}

    # ---------------------------------------------------------------------------------------------- prompt
    def build_messages(self, chunk_text: str) -> List[Dict[str, str]]:
        """The exact chat messages sent to Ollama (system prompt + few-shots + clipped query)."""
        k2l = self.k_to_letter
        parts = [f"Passage:\n{s.text}\n{self.answer_label}: {k2l[s.k]}\n" for s in self._shots()]
        parts.append(f"Passage:\n{clip(chunk_text, self.query_head, self.query_tail)}\n{self.answer_label}:")
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": "\n".join(parts)}]

    # ---------------------------------------------------------------------------------------------- tier 2
    def slm_scores(self, chunk_text: str) -> Dict[int, float]:
        """One forward pass with ``num_predict = 1``; returns ``{k: logprob}`` over the answer-letter tokens."""
        resp = self._client().chat(
            model=self.model,
            messages=self.build_messages(chunk_text),
            options=dict(self.options),
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )
        scores: Dict[int, float] = {}
        if getattr(resp, "logprobs", None):
            for tl in resp.logprobs[0].top_logprobs:
                tok = tl.token.strip().upper()
                if tok in self.letter_to_k and self.letter_to_k[tok] not in scores:
                    scores[self.letter_to_k[tok]] = tl.logprob
        if not scores:  # no log-probs from this server/model -> parse the emitted token -> default
            letters = "".join(sorted(self.letter_to_k))
            m = re.search(f"[{letters}{letters.lower()}]", resp.message.content or "")
            scores = {self.letter_to_k[m.group(0).upper()] if m else self.k_default: 0.0}
        return scores

    def decide(self, scores: Mapping[int, float]) -> int:
        """Argmax over letter log-probs, optionally shifted by the additive calibration bias.

        With ``tie_margin > 0`` a near-tie that involves ``k_default`` resolves to ``k_default``.  Ollama's logits
        carry a little run-to-run noise, so chunks whose top two letters are within ~0.1-0.2 nats can flip between
        runs; the margin makes those cases land on the paper's safe default instead.
        """
        adj = {k: v + (self.bias[k] if self.bias is not None else 0.0) for k, v in scores.items()}  # type: ignore[index]
        ranked = sorted(adj, key=adj.__getitem__, reverse=True)
        best = ranked[0]
        if self.tie_margin > 0 and len(ranked) > 1 and self.k_default in ranked[:2] and best != self.k_default:
            if adj[best] - adj[ranked[1]] <= self.tie_margin:
                return self.k_default
        return best

    # ---------------------------------------------------------------------------------------------- pipeline
    def predict_with_details(self, chunk_text: str) -> Tuple[int, str, Dict[int, float], Optional[str]]:
        """Full pipeline; returns ``(k, tier, scores, table_evidence)`` for inspection and logging."""
        if self.fixed_k is not None:
            return int(self.fixed_k), "fixed", {int(self.fixed_k): 0.0}, None
        if self.table_gate:
            evidence = table_evidence(chunk_text)
            if evidence is not None:
                return int(self.table_k), "table", {int(self.table_k): 0.0}, evidence
        scores = self.slm_scores(chunk_text)
        return int(self.decide(scores)), "slm", scores, None

    def predict(self, chunk_text: str) -> int:
        """Full pipeline (Tier 1 table gate, then Tier 2 SLM); returns only the suggested slice radius."""
        return self.predict_with_details(chunk_text)[0]

    __call__ = predict

    def with_(self, **overrides: Any) -> "DynamicSlicePredictor":
        """A copy with some settings changed, e.g. ``predictor.with_(model="qwen3.5:4b")``."""
        return replace(self, **overrides)


# =====================================================================================================================
# Module-level convenience API
# =====================================================================================================================
_DEFAULT_PREDICTOR: Optional[DynamicSlicePredictor] = None


def _predictor(**overrides: Any) -> DynamicSlicePredictor:
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = DynamicSlicePredictor()
    if not overrides:
        return _DEFAULT_PREDICTOR
    if "few_shots" not in overrides:          # reuse the already-loaded exemplars instead of re-reading the JSONs
        overrides["few_shots"] = _DEFAULT_PREDICTOR._shots()
    if "client" not in overrides and "host" not in overrides:
        overrides["client"] = _DEFAULT_PREDICTOR._client()
    return _DEFAULT_PREDICTOR.with_(**overrides)


def predict(chunk_text: str, **overrides: Any) -> int:
    """Suggest the document-slice radius ``k`` (0..3) for one Docling chunk.

    Runs the full pipeline: the deterministic table gate first (``k = TABLE_K`` for any chunk carrying a table), then a
    single ``num_predict = 1`` forward pass of the SLM with the module-default prompt and exemplars.  Any
    ``DynamicSlicePredictor`` field can be overridden per call, e.g. ``model="qwen3.5:4b"``,
    ``few_shots=[(text, k), ...]``, ``table_k=1``, ``bias=NOTEBOOK_CALIBRATION_BIAS``, ``query_head=900``.
    """
    return _predictor(**overrides).predict(chunk_text)


def predict_with_details(chunk_text: str, **overrides: Any) -> Tuple[int, str, Dict[int, float], Optional[str]]:
    """Like :func:`predict` but returns ``(k, tier, scores, table_evidence)``."""
    return _predictor(**overrides).predict_with_details(chunk_text)


# =====================================================================================================================
# CLI:  python dynamic_slice_prediction.py chunk.txt   |   echo "text" | python dynamic_slice_prediction.py -
# =====================================================================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Predict the document-slice radius k (0..3) for a chunk of text.")
    ap.add_argument("source", help="path to a text file, or '-' to read the chunk from stdin")
    ap.add_argument("--model", default=OLLAMA_MODEL_NAME)
    ap.add_argument("--few-shots", help="JSON file with [{'text':..., 'k':...}, ...] exemplars (default: notebook exemplars)")
    ap.add_argument("--calibrate", action="store_true", help="apply the notebook's calibration bias")
    ap.add_argument("--details", action="store_true", help="also print tier, letter log-probs and table evidence")
    a = ap.parse_args()

    text = sys.stdin.read() if a.source == "-" else Path(a.source).read_text(encoding="utf-8")
    kwargs: Dict[str, Any] = {"model": a.model}
    if a.few_shots:
        kwargs["few_shots"] = few_shots_from_json(a.few_shots)
    if a.calibrate:
        kwargs["bias"] = NOTEBOOK_CALIBRATION_BIAS
    if a.details:
        k, tier, scores, evidence = predict_with_details(text, **kwargs)
        print(f"k={k} tier={tier} scores={ {kk: round(v, 3) for kk, v in sorted(scores.items())} } evidence={evidence!r}")
    else:
        print(predict(text, **kwargs))
