#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import List, Tuple

# Models you asked for
TARGET_MODELS: List[Tuple[str, str]] = [
    # (module_and_class, short_tag_for_edge_suffix)
    ("GCN.GCN", "GCN"),
    ("GIN.GIN", "GIN"),
    ("GraphSAGE.GraphSAGE", "GraphSAGE"),
    ("SGC_CGU.SGC_CGU", "SGC_CGU"),
    ("SGC.SGC", "SGC"),
]

MODEL_ANCHOR = "erasure.model.graphs.GAT.GAT"
EDGE_TAG_ANCHOR = "GAT_edge"  # what appears inside strings like "..._GAT_edge.json"

def make_out_name(src: Path, short: str) -> Path:
    """
    Create an output filename for the given model.
    Tries to replace _GAT_edge / _GAT. / _GAT_ in the filename first.
    Falls back to a single 'GAT' replace, and finally (if nothing) appends _{short}.
    """
    name = src.name

    # Try the most common patterns first
    patterns = [
        ("_GAT_edge", f"_{short}_edge"),
        ("_GAT.", f"_{short}."),
        ("_GAT_", f"_{short}_"),
    ]

    new_name = name
    for old, new in patterns:
        if old in new_name:
            new_name = new_name.replace(old, new)
            break
    else:
        # No pattern matched; try a single 'GAT' replacement
        if "GAT" in new_name:
            new_name = new_name.replace("GAT", short, 1)
        else:
            stem = src.stem
            new_name = f"{stem}_{short}{src.suffix}"

    return src.with_name(new_name)

def transform_content(txt: str, module_and_class: str, short: str) -> str:
    """
    Replace model class and any *_GAT_edge occurrences.
    Does global replacements to catch every occurrence in the file.
    """
    out = txt.replace(MODEL_ANCHOR, f"erasure.model.graphs.{module_and_class}")
    # Replace any GAT_edge tokens in paths or identifiers
    out = out.replace(EDGE_TAG_ANCHOR, f"{short}_edge")
    return out

def process_file(path: Path, out_dir: Path = None) -> List[Path]:
    """
    For a single file, write one output file per target model.
    Returns list of created file paths.
    """
    text = path.read_text(encoding="utf-8")
    created = []
    for module_and_class, short in TARGET_MODELS:
        new_text = transform_content(text, module_and_class, short)
        out_path = make_out_name(path, short)
        if out_dir:
            out_path = out_dir / out_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(new_text, encoding="utf-8")
        created.append(out_path)
    return created

def main():
    ap = argparse.ArgumentParser(
        description="Clone JSONC configs by swapping GAT with other graph models."
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Input file(s) or globs (e.g., configs/*_GAT_edge.jsonc)",
    )
    ap.add_argument(
        "-o", "--out-dir", default=None,
        help="Optional output directory (defaults to same directory as input).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None

    # Expand globs
    files: List[Path] = []
    for pattern in args.inputs:
        matches = list(Path().glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        files.extend(matches)

    if not files:
        raise SystemExit("No matching input files found.")

    all_created = []
    for f in files:
        if not f.is_file():
            continue
        created = process_file(f.resolve(), out_dir)
        for c in created:
            print(c)
        all_created.extend(created)

if __name__ == "__main__":
    main()
