import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = (
    ROOT / "RnD" / "preprocessed_chunks" / "chunks_with_metadata.json",
    ROOT / "RnD" / "preprocessed_chunks" / "ablation_doc_slice_radius_dynamic.json",
)


def text_values(value):
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            yield text
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from text_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from text_values(child)


def main():
    for path in FILES:
        print(f"Processing {path}...")
        with path.open(encoding="utf-8") as file:
            texts = list(text_values(json.load(file)))

        formula_count = sum("<!-- formula-not-decoded -->" in text for text in texts)
        glyph_count = sum(
            "GLYPH<" in text and ">" in text.split("GLYPH<", 1)[1] for text in texts
        )
        print(f'{path.name}: "<!-- formula-not-decoded -->" count: {formula_count}')
        print(f'{path.name}: "GLYPH<...>" count: {glyph_count}')


if __name__ == "__main__":
    main()
