import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


def load_extracted_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a SEC extracted JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _extract_section_text(section_value: Any) -> str:
    if isinstance(section_value, str):
        return section_value.strip()

    if isinstance(section_value, dict):
        text = section_value.get("text")
        if isinstance(text, str):
            return text.strip()

    return ""


def concatenate_json_values(
    data: Dict[str, Any],
    separator: str = "\n\n",
    include_titles: bool = True,
) -> str:
    """
    Concatenate all section values from an extracted 10-K JSON into one string.

    The tree is built over the full filing narrative rather than per key/value pair.
    """
    parts: List[str] = []

    for key in sorted(data.keys(), key=lambda item: (len(item), item)):
        section = data[key]
        text = _extract_section_text(section)
        if not text:
            continue

        if include_titles and isinstance(section, dict) and section.get("title"):
            parts.append(f"[{section['title']}]\n{text}")
        else:
            parts.append(text)

    return separator.join(parts)


def load_concatenated_text(
    json_path: Union[str, Path],
    separator: str = "\n\n",
    include_titles: bool = True,
) -> str:
    """Load a JSON filing and return all section text concatenated as one string."""
    data = load_extracted_json(json_path)
    return concatenate_json_values(
        data,
        separator=separator,
        include_titles=include_titles,
    )


def discover_extracted_json_files(directory: Union[str, Path]) -> List[Path]:
    """Return all *_extracted.json files in a directory, sorted by name."""
    directory = Path(directory)
    return sorted(directory.glob("*_extracted.json"))
