import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, create_model


_DATATYPE_MAP: Dict[str, Any] = {
    "STRING": str,
    "INTEGER": int,
    "NUMBER": float,
    "BOOLEAN": bool,
    "DATETIME": datetime.datetime,
    "DATE": datetime.date,
    "TIME": datetime.time,
    "OBJECT": dict,
    "ARRAY": list,
}


COMIC_ASPECT_MAPPING: Dict[str, str] = {
    "Issue Number": "issue_number",
    "Features": "features",
    "Series Title": "series_title",
    "Publication Year": "publication_year",
    "Publisher": "publisher",
    "Artist/Writer": "writer",
    "Genre": "genre",
    "Character": "characters",
    "Country/Region of Manufacture": "country",
    "Language": "language",
    "Style": "style",
    "Era": "era",
    "Cover Artist": "cover_artist",
}


def sanitize_enum(values: List[Union[str, int, float]]) -> List[Union[str, int, float]]:
    """Sanitize enum values to avoid invalid characters in strict JSON schemas."""
    sanitized: List[Union[str, int, float]] = []
    for v in values:
        if isinstance(v, str) and '"' in v:
            v = v.replace('"', "'").strip()
        sanitized.append(v)
    return sanitized


def tokenize(text: str) -> List[str]:
    """Split normalized text into lowercase tokens."""
    return text.lower().split()


def get_relevant_enum(
    comic_aspect_value: Union[str, int, float, List[Union[str, int, float]]],
    ebay_aspect_values: List[str],
) -> List[str]:
    """
    Find relevant eBay aspect values for the given comic_aspect_value.
    - Works with string, numeric, or list input.
    - Matches based on token overlap (case-insensitive).
    """
    if not comic_aspect_value:
        return []

    # Normalize input into list of strings
    if isinstance(comic_aspect_value, (int, float)):
        comic_values = [str(comic_aspect_value)]
    elif isinstance(comic_aspect_value, str):
        comic_values = [comic_aspect_value]
    elif isinstance(comic_aspect_value, list):
        comic_values = [str(v) for v in comic_aspect_value]
    else:
        comic_values = [str(comic_aspect_value)]

    # Preprocess ebay aspect values
    normalized_ebay: Dict[str, str] = {v.lower(): v for v in ebay_aspect_values}
    enums: set[str] = set()

    for comic_val in comic_values:
        tokens = tokenize(comic_val)
        for ebay_key, original in normalized_ebay.items():
            ebay_tokens = ebay_key.split()
            if any(token in ebay_tokens for token in tokens):
                enums.add(original)

    return sorted(enums)


def create_pydantic_model(
    aspects: List[Dict[str, Any]],
    comic_data: Dict[str, Any],
    model_name: str = "ComicAspects",
    max_aspect_values: int = 150,
    example_preview_count: int = 10,
) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic model based on aspects and comic data.
    """
    fields: Dict[str, Tuple[Any, Field]] = {}

    for aspect in aspects:
        # Validate input
        name: str = aspect.get("aspect_name")
        if not name:
            raise ValueError("Each aspect must have an 'aspect_name'.")

        datatype: str = aspect.get("aspect_datatype", "STRING").upper()
        aspect_mode: str = aspect.get("aspect_mode", "")
        base_type = _DATATYPE_MAP.get(datatype, str)
        constraint: str = aspect.get("aspect_value_constraint", "SINGLE")

        original_values = aspect.get("aspect_values", []) or []
        values = sanitize_enum(original_values)

        is_required: bool = bool(aspect.get("is_required", False))

        comic_aspect_name = COMIC_ASPECT_MAPPING.get(name)
        comic_aspect_value = comic_data.get(comic_aspect_name)

        enums = get_relevant_enum(comic_aspect_value, values) if comic_aspect_value else []

        literal_candidates = enums if bool(enums) else values
        use_literal = (
            (len(literal_candidates) <= max_aspect_values and literal_candidates)
            or aspect_mode == "SELECTION_ONLY"
        )

        # Decide annotation type
        if use_literal:
            literal_values = tuple(sorted(set(literal_candidates)))
            if constraint == "MULTI":
                annotation = List[Literal[literal_values]]  # type: ignore
            else:
                annotation = Literal[literal_values]  # type: ignore
        else:
            annotation = List[base_type] if constraint == "MULTI" else base_type

        # Default handling
        default_value = ... if is_required else None

        # Wrap optional types
        if not is_required:
            annotation = Optional[annotation]  # type: ignore

        # Generate field description
        preview = ", ".join(map(str, values[:example_preview_count]))
        if len(values) > example_preview_count:
            preview += ", ... etc."
        description = f"{name} (type: {datatype}, constraint: {constraint}) | Examples: {preview}"

        fields[name] = (annotation, Field(default_value, description=description))

    model = create_model(model_name, __base__=BaseModel, **fields)
    return model


if __name__ == '__main__':
    from pathlib import Path
    import json

    BASE_DIR = Path(__name__).parent.parent.resolve()
    ASPECT_DATA_DIR = BASE_DIR / "extracted_comic.json"
    CLIENT_DATA_DIR = BASE_DIR / "client_rules.json"
    COMIC_DATA_DIR = BASE_DIR / "extracted_comic_data.json"

    with open(COMIC_DATA_DIR, 'r', encoding="utf-8") as f:
        comic_data = json.load(f)

    with open(ASPECT_DATA_DIR, 'r') as f:
        aspect_data = json.load(f)
    
    Comic = create_pydantic_model(aspects=aspect_data, comic_data=comic_data)

    # print(Comic.model_json_schema())
