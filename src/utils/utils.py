import json

def save_dict_to_json(d: dict, filepath: str):
    """Save dictionary to JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def load_dict_from_json(filepath: str) -> dict:
    """Load dictionary from JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


from pathlib import Path
from typing import List

def tree_str(
    root: str | Path,
    max_files: int = 3,
    show_hidden: bool = False,
) -> str:
    """
    Return a string that resembles the output of the `tree` command.
    - For each directory, at most `max_files` files are shown.
    - If more files exist, an indicator line like "... N more files" is shown.
    - Directories are always listed (no per-directory limit for subdirs).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root!s} does not exist")

    lines: List[str] = []
    lines.append(f"{root.name}/")

    def _iter_entries(d: Path):
        # optionally hide dotfiles
        ents = [p for p in sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
                if show_hidden or not p.name.startswith('.')]
        dirs = [p for p in ents if p.is_dir()]
        files = [p for p in ents if p.is_file()]
        return dirs, files

    def _recurse(d: Path, prefix: str):
        dirs, files = _iter_entries(d)

        # We need to know how many printed "items" this level will produce,
        # to decide which connector should be '└──' (last) vs '├──' (not last).
        printed_file_count = min(len(files), max_files)
        has_more_files = len(files) > max_files
        total_items = len(dirs) + printed_file_count + (1 if has_more_files else 0)

        idx = 0
        # directories first
        for sub in dirs:
            idx += 1
            is_last = idx == total_items
            connector = "└── " if is_last else "├── "
            lines.append(prefix + connector + sub.name + "/")
            _recurse(sub, prefix + ("    " if is_last else "│   "))

        # limited files
        for f in files[:max_files]:
            idx += 1
            is_last = idx == total_items
            connector = "└── " if is_last else "├── "
            lines.append(prefix + connector + f.name)

        # "more files" indicator
        if has_more_files:
            idx += 1
            is_last = idx == total_items
            connector = "└── " if is_last else "├── "
            remain = len(files) - max_files
            lines.append(prefix + connector + f"... {remain} more file{'s' if remain != 1 else ''}")

    _recurse(root, "")
    return "\n".join(lines)


# Example usage:
# if __name__ == "__main__":
#     print(tree_str("/mnt/data-vol/ASR-VIVOS-HCMUS-15H/raw/", max_files=3, show_hidden=False))

