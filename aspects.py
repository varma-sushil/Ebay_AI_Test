import json
from pathlib import Path


BASE_DIR = Path(__name__).parent.resolve()
ASPECTS_JSON = BASE_DIR / "comic.json"
OUTPUT_JSON_PATH = BASE_DIR / "extracted_comic.json"

with open(ASPECTS_JSON, "r") as f:
    aspects_data = json.load(f)

def extract_aspects_details(data:list[dict]):

    aspects_extracted_data:list[dict[str, str|list|bool|None]] = []
    for aspect in data:
        aspect_name:str = aspect.get("localizedAspectName")
        constraints:dict[str,str|bool] = aspect.get("aspectConstraint")
        data_type:str = constraints.get("aspectDataType")
        value_count:str = constraints.get("itemToAspectCardinality")
        aspect_mode:str = constraints.get("aspectMode")
        is_required:bool = constraints.get("aspectRequired")
        aspect_values:list[dict[str,str]] = aspect.get("aspectValues")

        values:list[str] = []
        if aspect_values:
            for aspect_value in aspect_values:
                value:str = aspect_value.get("localizedValue")
                if aspect_name == "Size" and value:
                    try:
                        value = str(int(value))
                    except Exception:
                        continue
                values.append(value)
        
        aspects_extracted_data.append(
            {
                "aspect_name": aspect_name,
                "aspect_datatype": data_type,
                "aspect_value_constraint": value_count,
                "aspect_mode": aspect_mode,
                "is_required": is_required,
                "aspect_values": values
            }
        )
    
    return aspects_extracted_data

extracted_aspects = extract_aspects_details(aspects_data)

with open(OUTPUT_JSON_PATH, 'w', encoding="utf-8") as f:
    json.dump(extracted_aspects, f, indent=4)

prompt_statement = ""
statements = []
demo = {}
for aspect_details in extracted_aspects:
    name = aspect_details.get("aspect_name")
    constraint = aspect_details.get("aspect_value_constraint")
    values = aspect_details.get("aspect_values")

    if len(values) <=20 and len(values) > 0:
        value_stmt = values[:20]
    else:
        value_stmt = "provide relevant value for the field from images if found else set it null."
    
    statement = f"{name} - {constraint}: {value_stmt}"
    
    statements.append(statement)

    demo[name] = None

prompt_statement = ",\n".join(statements)

print(prompt_statement)


final_data = {
    "comic": demo
}

with open("comic_aspect.json", 'w') as f:
    json.dump(final_data, f, indent=4)
