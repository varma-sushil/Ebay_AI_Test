import json
import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, create_model
from pathlib import Path

# from pydantic_model import create_pydantic_model
from comic.comic_pydantic_creator import create_pydantic_model

load_dotenv()

OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_PROMPT_TEMPLATE = """
You are an expert in product attribute mapping.

TASK
- Analyze the provided comic JSON and map each value to the corresponding eBay aspect enum values from the target schema.
- Produce an enum-safe JSON output.

MANDATORY MAPPING RULES
1. Always attempt to fill as many fields as possible:
   - Use direct comic data when present.
   - If not explicitly provided, infer from general knowledge of comics.
   - Only return null when the value is truly unknowable.
2. Use only values that appear in the schema enums. Do NOT invent new enum values. If you cannot confidently map to an enum value, return null for that aspect.
3. Normalization steps (apply in order):
   a. Case-insensitive exact match to enum.
   b. Remove common corporate suffixes/words (e.g., "Inc", "LLC", "Ltd", "Co", "Worldwide", "Publications", "Entertainment", "Press") and punctuation, then retry match.
   c. Normalize common tokens (e.g., "US" ↔ "United States", plurals/singulars, basic British/American spelling) and retry.
   d. Substring/token overlap / fuzzy match: select the enum with the highest token overlap. If token overlap is low or ambiguous, return null.
4. If an input value contains multiple concepts (e.g., "Second Printing Variant"), split and map each concept to the appropriate aspect(s) using schema enums (e.g., Features and Variant Type). Only infer additional detail (like variant letter "B") if that exact token appears in the input.
5. For lists (Characters, Features, etc.): include only items that map to schema enums. If none map, return null for that aspect.
6. Prefer higher-precision matches: if two enum values both match, choose the one with the most tokens in common or the more canonical/shorter schema value.
7. For missing aspects: use common defaults if strongly implied by data:
   - Type → "Comic Book"  
   - Tradition → "US Comics" (if publisher is Marvel/DC/etc. and country = US)  
   - Unit of Sale → "Single Unit" (unless clearly multiple issues/lot)  
   - Signed/Personalized/Inscribed → "No" unless explicitly stated  
   - Vintage → "No" if Modern Age or later  
   - Universe → map to schema (e.g., Marvel → "Marvel (MCU)") 
8. Always explain (one short sentence) any non-obvious mapping choices in the "reason" field.

INPUT
{comic_data}
"""

DEFAULT_TITLE_PROMPT = """
You are an expert in SEO-friendly comic title writing.

TASK
- Generate a concise title (≤80 chars) based on the mapped comic data.

TITLE RULES
- Include: Series, #Issue, Year (if present), one or two strongest features (e.g., "2nd Printing", "Variant"), and 1 key creator (writer or cover artist) if space allows.
- Keep punctuation minimal.
- Use proper title casing.
- Avoid promotional words like "Amazing" or "Lot".
"""

DEFAULT_DESCRIPTION_PROMPT = """
TASK
- Generate an HTML-formatted description (≤867 chars) from the mapped comic data.

DESCRIPTION RULES
- Opening <p> summary line (series + edition + issue + key feature).
- A short <ul> list of key attributes:
  Series, Issue, Year, Publisher, Writer, Cover Artist, Characters (top 3 if many), Language, Page Count, Style, Features, Barcode (if present).
- A short paragraph noting "Condition: Not specified" if no condition info provided.
- A one-line call-to-action for collectors.
- Keep tone neutral and factual; don’t promise grading or certification unless provided.
"""


def get_pydantic_model(aspect_file_path:Path|str, comic_data_file:Path|str|None=None):

    with open(aspect_file_path, 'r') as f:
        data = json.load(f)
    
    if comic_data_file:
        with open(comic_data_file, 'r') as f:
            comic_data = json.load(f)
        
        # client_rules = client_rules["Suits and suits separate"]
    else:
        comic_data = None
    
    return create_pydantic_model(data, comic_data)


def llm_model() -> ChatOpenAI:
    """Initialize ChatOpenAI client."""
    return ChatOpenAI(api_key=OPEN_AI_KEY, model="gpt-4.1-mini", temperature=0.1)

def pick_prompt(prompts: dict[str, any]|None, key: str, default: str) -> str:
    """
    Return a clean prompt string.
    Falls back to `default` if value is None, non-string, or blank/whitespace.
    """
    val = (prompts or {}).get(key)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return default.strip()

def build_response_model(schema_model: type[BaseModel]) -> type[BaseModel]:
    """Create a structured response model for the LLM output."""
    return create_model(
        "LLMResponse",
        data=(schema_model, ...),
        title=(str, Field(..., description="SEO-friendly title for the eBay listing (≤80 chars).")),
        description=(str, Field(..., description="HTML-formatted description for the listing (≤867 chars).")),
        price=(float|None, Field(default=None, description="Comic book price.")),
        confident_level=(
            float,
            Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0."),
        ),
        reason=(str, Field(..., description="Explanation for confidence level and non-obvious mappings.")),
        __base__=BaseModel,
    )


def extract_features_of_comic(
    comic_data:dict, schema_model:type[BaseModel], custom_prompts: dict[str, str|None]|None = None
):

    if not comic_data:
        raise ValueError("Provide extracted comic data for mapping and content generation.")
    
    if not (isinstance(schema_model, type) and issubclass(schema_model, BaseModel)):
        raise TypeError("schema_model must be a Pydantic BaseModel subclass.")
    
    llm = llm_model()

    ResponseModel = build_response_model(schema_model)

    template = pick_prompt(custom_prompts,"feature_extraction", DEFAULT_PROMPT_TEMPLATE)
    title_prompt = pick_prompt(custom_prompts, "title_prompt", DEFAULT_TITLE_PROMPT)
    description_prompt = pick_prompt(custom_prompts, "description_prompt", DEFAULT_DESCRIPTION_PROMPT)
    combined_prompt = template.strip() + '\n' + title_prompt.strip() + '\n' + description_prompt.strip()

    print("\nCombined Title & Description Prompt:\n",combined_prompt)
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", combined_prompt),
    ])

    human_msg = prompt.format_messages(
        comic_data=json.dumps(comic_data),
        format_instructions=""
    )[0]

    print("\nLLM PROMPT: ",human_msg.content)

    feature_llm = llm.with_structured_output(ResponseModel, strict=True, include_raw=True)
    response = feature_llm.invoke([human_msg])
    raw = response.get('raw')
    parsed = response.get('parsed')

    # print("\nRaw response: ", raw)
    # print("\nParsed response: ", parsed)

    data = parsed.model_dump()
    print("LLM response:", data)



if __name__ == '__main__':

    BASE_DIR = Path(__name__).parent.parent.resolve()
    ASPECT_DATA_DIR = BASE_DIR / "extracted_comic.json"
    CLIENT_DATA_DIR = BASE_DIR / "client_rules.json"
    COMIC_DATA_DIR = BASE_DIR / "extracted_comic_data.json"

    with open(COMIC_DATA_DIR, 'r', encoding="utf-8") as f:
        comic_data = json.load(f)

    Product = get_pydantic_model(ASPECT_DATA_DIR, COMIC_DATA_DIR)

    with open(BASE_DIR/"model_schema.json", "w") as f:
        json.dump(Product.model_json_schema(), f, indent=4)

    extracted_data = extract_features_of_comic(comic_data, Product)

