"""Static consistency checks for RnD/nature_article/sn_green_rag_article.tex (no TeX toolchain is installed):
balanced braces, matched environments, every \\ref/\\eqref target defined and every label referenced (including
listings' label={...}), every \\cite key present in refs.bib, every \\includegraphics file exists under RnD/nature_article/,
counts of \\todo and \\verify notes, rendered word counts of the abstract (<= 200), introduction (700-800),
discussion (300-400) and conclusion, and a heuristic list of sentences that state measured-looking numbers
without an adjacent \\verify/\\todo note (review manually; citation keys with underscores are false positives).
Run from the repository root:  .venv/bin/python RnD/verification/latex_manuscript_consistency_check.py [tex] [bib]"""
import re, sys, collections
from pathlib import Path
here = Path(__file__).resolve().parent
tex_path = Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent / "nature_article" / "sn_green_rag_article.tex"
bib_path = Path(sys.argv[2]) if len(sys.argv) > 2 else here.parent / "nature_article" / "refs.bib"
tex = open(tex_path, encoding="utf-8").read(); bib = open(bib_path, encoding="utf-8").read()
problems = []
body = re.sub(r"(?<!\\)%.*", "", tex)
stripped = body.replace(r"\{", "").replace(r"\}", "")
depth = 0; line = 1
for ch in stripped:
    if ch == "\n": line += 1
    if ch == "{": depth += 1
    elif ch == "}":
        depth -= 1
        if depth < 0: problems.append(f"negative brace depth at line {line}"); depth = 0
if depth: problems.append(f"unbalanced braces: final depth {depth}")
stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", body):
    if m.group(1) == "begin": stack.append(m.group(2))
    elif not stack or stack[-1] != m.group(2): problems.append(f"environment mismatch: end{{{m.group(2)}}} with stack {stack[-3:]}")
    else: stack.pop()
if stack: problems.append(f"unclosed environments: {stack}")
labels = re.findall(r"\\label\{([^}]*)\}", body) + re.findall(r"label=\{([^}]*)\}", body)
refs = re.findall(r"\\(?:ref|eqref|autoref|pageref)\{([^}]*)\}", body)
for l, c in collections.Counter(labels).items():
    if c > 1: problems.append(f"duplicate label: {l}")
for r in sorted(set(refs) - set(labels)): problems.append(f"undefined reference: {r}")
for l in labels:
    if l not in refs: problems.append(f"label never referenced: {l}")
bibkeys = set(re.findall(r"^@\w+\{([^,\s]+)", bib, flags=re.M))
cites = [k.strip() for m in re.finditer(r"\\cite[pt]?\{([^}]*)\}", body) for k in m.group(1).split(",")]
for k in sorted(set(cites) - bibkeys): problems.append(f"unknown citation key: {k}")
for img in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body):
    if not (tex_path.parent / img).exists(): problems.append(f"missing image file: {img}")
NESTED = r"\\(?:todo|verify)\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}"
def words(s):
    s = re.sub(NESTED, " ", s, flags=re.S)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s); s = s.replace("\\Rec{20}", "Recall@20").replace("\\gco{}", "gCO2eq")
    s = s.replace("\\,", "").replace("\\%", "%").replace("``", "").replace("''", "")
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s); s = re.sub(r"[{}$\\]", " ", s)
    return len(s.split())
def section(pat, nxt):
    m = re.search(r"\\section\{" + pat + r"\}.*?(?=" + nxt + ")", body, flags=re.S); return m.group(0) if m else ""
abs_m = re.search(r"\\abstract\{(.*?)\}\s*\n\s*\\keywords", body, flags=re.S)
print("abstract words:", words(abs_m.group(1)) if abs_m else "n/a", "(limit 200)")
print("introduction words:", words(section("Introduction", r"\\section\{Methodology")), "(target 700-800)")
print("discussion words:", words(section("Discussion[^}]*", r"\\section\{Conclusion")), "(target 300-400)")
print("conclusion words:", words(section("Conclusion", r"\\backmatter")))
nfig = len(re.findall(r"\\begin\{figure\}", body)); ntab = len(re.findall(r"\\begin\{(?:table|sidewaystable)\}", body))
n_todo = len(re.findall(r"\\todo\{", body)) - 1      # minus the \verify definition in the preamble
n_verify = len(re.findall(r"\\verify\{", body))
print(f"figures {nfig}, tables {ntab}, plain \\todo {n_todo}, \\verify {n_verify}, labels {len(labels)}, distinct citation keys {len(set(cites))}")
prose = re.sub(r"\\begin\{(table|figure|lstlisting|algorithm|equation|align)\}.*?\\end\{\1\}", " ", body, flags=re.S)
prose = re.sub(NESTED, "<<NOTE>>", prose, flags=re.S)
missing = []
for para in re.split(r"\n\s*\n", prose):
    if "\\section" in para or "\\subsection" in para or "\\abstract" in para or "\\lstset" in para or "\\newcommand" in para: continue
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", para.strip()):
        if re.search(r"\d", s) and "<<NOTE>>" not in s and not re.search(r"\\(cite|ref|eqref|label)", s):
            if re.search(r"\d[\d,.\\]*\s*(%|\\%|Wh|kWh|min|s\b|ms|GB|MiB|tokens|chunks|questions|papers|characters|points|pt)", s) or re.search(r"0\.\d{2,}", s):
                missing.append(s[:120].replace("\n", " "))
print("sentences with measured-looking numbers and no adjacent note (review manually):", len(missing))
for s in missing: print("   -", s)
print("PROBLEMS:" if problems else "no structural problems")
for p in problems: print("  *", p)
sys.exit(1 if problems else 0)
