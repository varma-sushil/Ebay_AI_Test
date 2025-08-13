import json
import os
import base64
import io

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from PIL import Image
from PIL.Image import Resampling
from pydantic import BaseModel, Field, ValidationError, create_model
from pathlib import Path

from pydantic_model import create_pydantic_model

load_dotenv()

OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
IMAGE_RESOLUTION = 1600


def get_pydantic_model(aspect_file_path:Path|str, client_rules_file_path:Path|str|None=None):

    with open(aspect_file_path, 'r') as f:
        data = json.load(f)
    
    if client_rules_file_path:
        with open(client_rules_file_path, 'r') as f:
            client_rules = json.load(f)
        
        client_rules = client_rules["Suits and suits separate"]
    else:
        client_rules = None
    
    return create_pydantic_model(data, client_rules=client_rules)


def resize_keep_aspect(img: Image.Image, max_dim: int) -> Image.Image:
    img_copy = img.copy()
    img_copy.thumbnail((max_dim, max_dim), Resampling.LANCZOS)
    return img_copy

def encode_image_to_base64(img: Image.Image):
    """Helper to convert a PIL image to a Base64."""
    fmt = img.format or "JPEG"
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    try:
        mime_type = Image.MIME.get(fmt, f"image/{fmt.lower()}")
    except KeyError:
        raise ValueError(f"Unknown MIME type for format: {fmt!r}")
    
    return b64, mime_type


def load_images_from_directory(directory_path: Path|str) -> list[Image.Image]:
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    images = []
    for image_path in directory_path.iterdir():
        if image_path.suffix.lower() in image_extensions:
            try:
                img = Image.open(image_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Failed to load {image_path.name}: {e}")
    
    return images


def llm_model():
    llm = ChatOpenAI(api_key=OPEN_AI_KEY, model="gpt-4.1-mini", temperature=0.1)
    # llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model="gemini-2.5-flash")
    return llm

def generate_title_and_description(llm, aspects_data):

    PROMPT = """
    You are an expert eBay listing writer.
    You are given structured product details extracted from images:
    {aspect_details}

    title: Write SEO-friendly eBay title with brand, item type, size, color, and key selling points. Max 80 characters.
    Write a clear, professional eBay description following these rules:
    description should be in max 867 characters.

    Format:
    Team: [If applicable — sports fan gear only, e.g., Pitt Panthers]
    Brand: [e.g., Nike]
    Item: [e.g., Hoodie]
    Gender: [e.g., Mens, Womens, Boys, Girls]
    Size: [Full label, e.g., Extra Large (XL)]
    Measurements:
    Please see photos above for all measurements.
    Note: Item(s) may have been altered from the original tag size, so please confirm measurements in photos to ensure desired fit.
    Color Notes or Fit Notes (only applicable if confident is low.):
    - If color is borderline: "Color appears beige/tan/brown — see photos for best representation."
    - If size tag doesn’t match actual fit: "Tag says Mens XL but fits closer to Mens Small — please refer to measurements."
    Condition:
    Select from: New With Tags, New Without Tags, New With Defects, Excellent Pre-Owned, Good Pre-Owned, Fair Pre-Owned.
    If pre-owned and flawed, include notes like:
    “This item has [hole(s), pilling, faint stain(s), fading, loose threads, etc.].”
    Shipping & Customer Support:
    I will ship this item out with a tracking number for confirmation.
    Please feel free to ask any questions you may have.
    If there are any problems with your order, please message us ASAP so I can help resolve the issue quickly.
    
    If Not Clothing (General Items):
        Omit gender/fit language and sizing notes. Instead:
        Replace “Measurements” section with:
        “Please see photos for details and dimensions.”
        Replace any fit/alteration text with notes on:
        Functionality
        Missing parts
        Scratches/wear
        Original packaging presence/absence
    
    Output Format:
    {format_instructions}
    """.strip()

    ResponseModel = create_model(
        "LLMResponse",
        title = (str, Field(..., description="Generate title based on extracted field values for ebay listing.")),
        description = (str, Field(..., description="Generated description.")),
        __base__=BaseModel
    )
    parser = PydanticOutputParser(pydantic_object=ResponseModel)

    # description_prompt = ChatPromptTemplate.from_messages([
    #     ("human", PROMPT),
    # ])

    # system_msg = description_prompt.format_messages(
    #     aspect_details=aspects_data,
    #     format_instructions=parser.get_format_instructions()
    # )[0]
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT)
    chain = prompt_template | llm

    response = chain.invoke({
        "aspect_details": aspects_data,
        "format_instructions": parser.get_format_instructions()
    })
    print("final response contet",response.content)
    result = json.loads((response.content).replace("```json", "").replace("```", ""))

    return result


def extract_features_from_images(images_list:list[Image.Image], schema_model: type[BaseModel],custom_prompt: str = None) -> BaseModel:

    if not images_list:
        raise ValueError("Image URL is required.")
    
    if not (isinstance(schema_model, type) and issubclass(schema_model, BaseModel)):
        raise TypeError("schema_model must be a Pydantic BaseModel subclass.")
    
    llm = llm_model()

    ResponseModel = create_model(
        "LLMResponse",
        title = (str, Field(..., description="Generate title based on extracted field values for ebay listing.")),
        description = (str, Field(..., description="Generated description.")),
        data=(schema_model, ...),
        confident_level=(
            float,
            Field(
                ...,
                ge=0.0,
                le=1.0,
                description="Confidence score between 0.0 and 1.0",
            ),
        ),
        reason=(str, Field(..., description="Explanation for confidence level")),
        __base__=BaseModel,
    )
    print("pydatic model schema", ResponseModel.model_json_schema())
    parser = PydanticOutputParser(pydantic_object=ResponseModel)

    field_descriptions = "\n".join(
        f'- {field.description}'
        for _, field in schema_model.model_fields.items()
    ).strip()

    # print("field description ", field_descriptions)

    system_msg = SystemMessage(
    content=f"""
    You are an expert in product attribute extraction.
    Analyze all provided images of a single product (tags, labels, imprints, barcodes, care tags, linings, printed text, measurements). Treat all images as one item.
    Instructions:
    - Use measuring scales in images to determine size-related fields (e.g., Size, Chest Size, Inseam, etc.) if not explicitly on tags.
    - Only extract values explicitly visible and clearly legible — no guessing or inference.
    - Map extracted values exactly to the provided JSON schema and enums.
    - If a value appears in short form (e.g., country abbreviations), convert it to the full allowed enum form.
    - Return null for missing, unclear, or obscured fields.
    - Include a confidence score (0.0–1.0) with a detailed reason.
    - Output must be valid JSON per the provided schema.

    Fields to extract:
    {field_descriptions}

    title: Write SEO-friendly eBay title with brand, item type, size, color, and key selling points. Max 80 characters.
    Write a clear, professional eBay description using the information provided. Follow this format and logic:
    description should be in max 867 characters.
    Format:
    Team: [If applicable — sports fan gear only, e.g., Pitt Panthers]
    Brand: [e.g., Nike]
    Item: [e.g., Hoodie]
    Gender: [e.g., Mens, Womens, Boys, Girls]
    Size: [Full label, e.g., Extra Large (XL)]
    Measurements:
    Please see photos above for all measurements.
    Note: Item(s) may have been altered from the original tag size, so please confirm measurements in photos to ensure desired fit.
    Color Notes or Fit Notes (only applicable if confident is low.):
    - If color is borderline: "Color appears beige/tan/brown — see photos for best representation."
    - If size tag doesn’t match actual fit: "Tag says Mens XL but fits closer to Mens Small — please refer to measurements."
    Condition:
    Select from: New With Tags, New Without Tags, New With Defects, Excellent Pre-Owned, Good Pre-Owned, Fair Pre-Owned.
    If pre-owned and flawed, include notes like:
    “This item has [hole(s), pilling, faint stain(s), fading, loose threads, etc.].”
    Shipping & Customer Support:
    I will ship this item out with a tracking number for confirmation.
    Please feel free to ask any questions you may have.
    If there are any problems with your order, please message us ASAP so I can help resolve the issue quickly.
    
    If Not Clothing (General Items):
        Omit gender/fit language and sizing notes. Instead:
        Replace “Measurements” section with:
        “Please see photos for details and dimensions.”
        Replace any fit/alteration text with notes on:
        Functionality
        Missing parts
        Scratches/wear
        Original packaging presence/absence

    Output Format:
    {parser.get_format_instructions()}
    """.strip()
    )

    print(system_msg.content)

    content_blocks = []

    for image in images_list:
        
        compressed_image = resize_keep_aspect(image, IMAGE_RESOLUTION)
        encoded_image, mime_type = encode_image_to_base64(compressed_image)
        if encoded_image:
            content_blocks.append({
            "type": "image",
            "source_type": "base64",
            "mime_type":mime_type,
            "data": encoded_image,
        })
    
    user_msg = HumanMessage(content=content_blocks)

    try:
        raw = llm.invoke([system_msg, user_msg])
        print("LLM raw response:", raw)
        # print("Parsed pydantic model response:", raw.model_dump())
        parsed = raw.content.replace("```json", "").replace("```", "")
        # parsed = parser.parse(raw.content)
        print("LLM response:", parsed)

        # title_description = generate_title_and_description(llm, parsed)
        # final_data = json.loads(parsed)
        # final_data["title"] = title_description.get("title")
        # final_data["description"] = title_description.get("description")

        # print("final data: ", final_data)
    except ValidationError as ve:
        raise RuntimeError(f"Response did not match schema: {ve}") from ve
    except Exception as e:
        raise RuntimeError(f"LLM invocation failed: {e}") from e

    return parsed


if __name__ == '__main__':

    BASE_DIR = Path(__name__).parent.resolve()
    ASPECT_DATA_DIR = BASE_DIR / "extracted_suits_aspects.json"
    CLIENT_DATA_DIR = BASE_DIR / "client_rules.json"
    IMAGE_DIR = BASE_DIR / "Archive 1" / "020754"
    IMAGE_DIR = BASE_DIR / "Archive 1" / "020755"
    # IMAGE_DIR = BASE_DIR / "Archive 1" / "020758"

    Product = get_pydantic_model(ASPECT_DATA_DIR, CLIENT_DATA_DIR)

    images = load_images_from_directory(IMAGE_DIR)

    extracted_data = extract_features_from_images(images, Product)

