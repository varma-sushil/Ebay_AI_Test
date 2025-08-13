from bs4 import BeautifulSoup
import requests

import filetype
import base64

from typing import Literal, Optional, List, Dict
from pydantic import BaseModel, Field, create_model
import re


def snake_case(name: str) -> str:
    s = re.sub(r"[^\w]+", "_", name).lower()
    return re.sub(r"_+", "_", s).strip("_")

def create_pydantic_model(aspects:List[Dict], max_aspect_values:int=15):

    fields = {}
    for aspect in aspects:
        field_name = snake_case(aspect["aspect_name"])
        aspect_value = aspect.get("aspect_values")

        if len(aspect_value) <= max_aspect_values:
            literals = aspect["aspect_values"]
        else:
            literals = None
        
        if literals:
            annotation = Literal[tuple(literals)]
        else:
            annotation = str
        
        default = ... if aspect["is_required"] else None

        if len(aspect_value) <= 5:
            description_eg = ",".join(value for value in aspect_value)
        else:
            description_eg = ",".join(value for value in aspect_value[0:5]) + "... etc."

        description = f"{aspect['aspect_name']} (eg. {description_eg})"
        fields[field_name] = (Optional[annotation] if not aspect["is_required"] else annotation,
                            Field(default, description=description))
    
    ProductAspects = create_model(
        "ProductAspects",
        __base__=BaseModel,
        **fields
    )
    return ProductAspects


def get_image_urls(urls: list):
    image_urls = []
    
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            img_container = soup.select_one("#image-viewer-container")

            if img_container:
                img_tag = img_container.find("img")
                if img_tag and img_tag.get("src"):
                    image_urls.append(img_tag["src"])
                else:
                    print(f"No <img> tag found in container for URL: {url}")
            else:
                print(f"No #image-viewer-container found for URL: {url}")

        except Exception as e:
            print(f"Error fetching image from {url}: {e}")
    
    return image_urls


def get_base64_image(image_url:str):

    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            kind = filetype.guess(response.content)
            if kind is None:
                raise ValueError("Unknown file type")
            mime_type = kind.mime
            base64_encoded_img = base64.b64encode(response.content).decode()

            return base64_encoded_img, mime_type
    except Exception as e:
        print(f"Got an error for {image_url} | {e}")
        return None


if __name__ == "__main__":
    import json

    # links = [
    #     "https://ibb.co/YT15134y",
    #     "https://ibb.co/pr42X2Jw",
    #     "https://ibb.co/QFDYSyNM",
    #     "https://ibb.co/4RHqFr77",
    #     "https://ibb.co/JW6Z1Hmf",
    #     "https://ibb.co/ccQDRtRt",
    #     "https://ibb.co/mKFvq6b",
    #     "https://ibb.co/mVX3tZjg",
    #     "https://ibb.co/fYPFGpWn",
    #     "https://ibb.co/xKS50rxt",
    #     "https://ibb.co/5gZR7K19",
    #     "https://ibb.co/7xQN83Tz",
    #     "https://ibb.co/j9XXLbVz",
    #     "https://ibb.co/pjJHn8h9",
    #     "https://ibb.co/7t7ZwBNV",
    #     "https://ibb.co/5XZQ8Gpt",
    #     "https://ibb.co/3mjNtWVz",
    #     "https://ibb.co/7tFzn3ty",
    #     "https://ibb.co/HDpM2bZs",
    #     "https://ibb.co/yB6WQdvp",
    #     "https://ibb.co/TxvRxNRy",
    #     "https://ibb.co/LzY7P9cy",
    #     "https://ibb.co/YFft804X",
    #     "https://ibb.co/NnKcXjGq",
    #     "https://ibb.co/Xk3r5cP5",
    # ]

    # print(get_image_urls(links))

    with open(r"C:\Users\VARMA\Downloads\Work\ebay_ai\aspects_data.json", 'r') as f:
        data = json.load(f)
    
    ProductAttributes = create_pydantic_model(data)

    print(ProductAttributes.model_json_schema())
