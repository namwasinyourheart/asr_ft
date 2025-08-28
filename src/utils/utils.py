import json

def save_dict_to_json(d: dict, filepath: str):
    """Save dictionary to JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def load_dict_from_json(filepath: str) -> dict:
    """Load dictionary from JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
