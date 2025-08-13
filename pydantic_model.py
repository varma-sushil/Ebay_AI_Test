import datetime
from typing import Any, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, create_model


# Map your domain datatypes to real Python types
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

def create_pydantic_model(
    aspects: List[Dict[str, Any]],
    model_name: str = "ProductAspects",
    max_aspect_values: int = 50,
    example_preview_count: int = 10,
    client_rules:dict | None = None
) -> type[BaseModel]:
    """
    Dynamically generate a Pydantic model from a list of aspect definitions.

    Each aspect dict must contain:
      - aspect_name: str
      - aspect_datatype: str (currently only STRING supported)
      - is_required: bool
      - aspect_values: List[str]

    Fields with <= max_aspect_values will be typed as Literal choices; others default to str.

    Args:
        aspects: list of aspect metadata dictionaries.
        model_name: name of the generated Pydantic model.
        max_aspect_values: max values for using Literal typing.
        example_preview_count: how many example values to show in the field description.

    Returns:
        A Pydantic BaseModel subclass with one field per aspect.
    """
    fields: Dict[str, Tuple[Any, Field]] = {}

    for aspect in aspects:
        # Validate input
        name = aspect.get("aspect_name")
        if not name:
            raise ValueError("Each aspect must have an 'aspect_name'.")
        
        datatype = aspect.get("aspect_datatype", "STRING").upper()
        aspect_mode = aspect.get("aspect_mode", "")
        base_type = _DATATYPE_MAP.get(datatype, None)
        constraint = aspect.get("aspect_value_constraint", "SINGLE")

        values = aspect.get("aspect_values", []) or []
        is_required = bool(aspect.get("is_required", False))

        rule = client_rules.get(name) if client_rules else {}
        rule = rule or {}
        
        help_text = rule.get("help_text", "")
        client_values = rule.get("value_to_use", [])
        client_default_value = rule.get("default_value", None)
        

        final_values = client_values if client_values else values
        use_literal = (len(final_values) <= max_aspect_values and final_values) or aspect_mode == "SELECTION_ONLY"

        if use_literal:
            # Ensure no duplicate values
            literal_values = tuple(sorted(set(final_values)))
            
            if constraint == "MULTI":
                annotation = List[Literal[literal_values]]  # type: ignore
            else:
                annotation = Literal[literal_values]  # type: ignore
        else:
            if constraint == "MULTI":
                annotation = List[base_type]
            else:
                annotation = base_type

        default_value = client_default_value if client_default_value is not None else (None if not is_required else ...)

        # Wrap optional types
        if not is_required:
            annotation = Optional[annotation]  # type: ignore
            default_value = client_default_value
        
        field_default = default_value

        # Generate description with preview of examples
        preview = ", ".join(final_values[:example_preview_count])
        if len(final_values) > example_preview_count:
            preview += ", ... etc."
        
        description = f"{name}: {help_text.strip()}"
        if preview and not help_text:
            description += f" Example values: {preview}"

        fields[name] = (annotation, Field(field_default, description=description))

    # Create model
    model = create_model(model_name, __base__=BaseModel, **fields)
    return model

if __name__ == '__main__':

    from pathlib import Path
    import json

    BASE_DIR = Path(__name__).parent.resolve()
    ASPECT_DATA_DIR = BASE_DIR / "extracted_suits_aspects.json"
    CLIENT_RULES_DIR = BASE_DIR / "client_rules.json"

    with open(ASPECT_DATA_DIR, 'r') as f:
        aspect_data = json.load(f)

    with open(CLIENT_RULES_DIR, 'r') as f:
        client_rules = json.load(f)
    
    client_rules = client_rules['Suits and suits separate']
    Product = create_pydantic_model(aspects=aspect_data, client_rules=client_rules)

    print(Product.model_json_schema())
    