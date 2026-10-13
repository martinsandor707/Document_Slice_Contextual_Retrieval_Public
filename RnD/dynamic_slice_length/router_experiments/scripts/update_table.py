"""Regenerate the master measurement table (make_table.py) and swap it into RnD/router_prompt_engineering.md."""
import os
import subprocess

W = os.path.dirname(os.path.abspath(__file__))
PY = "/home/martin/projects/TDK/Document_Slice_Contextual_Retrieval/.venv/bin/python"
MD = "/home/martin/projects/TDK/Document_Slice_Contextual_Retrieval/RnD/router_prompt_engineering.md"

out = subprocess.run([PY, os.path.join(W, "make_table.py")], capture_output=True, text=True, cwd=W).stdout
table = [l for l in out.splitlines() if l.startswith("|")]
assert table and table[0].startswith("| Configuration | Kind |"), out[-500:]

lines = open(MD).read().split("\n")
start = next(i for i, l in enumerate(lines) if l.startswith("| Configuration | Kind |"))
end = start
while end < len(lines) and lines[end].startswith("|"):
    end += 1
new = lines[:start] + table + lines[end:]
open(MD, "w").write("\n".join(new))
print(f"master table replaced: {end - start} -> {len(table)} lines")
