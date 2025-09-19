import re
import time
import logging

from curl_cffi import requests
from bs4 import BeautifulSoup
from comic.mapper import iso_language_to_name, iso_country_to_name, map_era


BASE_COMIC_URL = "https://www.comics.org"
BASE_COMIC_API_URL = "https://www.comics.org/api"
AUTH = ("sushilvarma@reluconsultancy.in", "Sushil@123")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_session(auth: tuple[str, str]|None = None) -> requests.Session:
    """Create and return a reusable session with Chrome impersonation."""
    return requests.Session(
        impersonate="chrome",
        allow_redirects=True,
        auth=auth,
        timeout=20, 
    )


def safe_request(
    session: requests.Session, url: str, params: dict|None = None, retries: int = 3
) -> requests.Response | None:
    """Make a safe GET request with retry and exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params)
            if resp.status_code == 200:
                return resp
            logger.warning("Request failed (status=%s) attempt=%s", resp.status_code, attempt)
        except Exception as e:
            logger.warning("Request error on attempt %s: %s", attempt, e)

        time.sleep(2**attempt)  # exponential backoff
    logger.error("Failed to fetch URL after %s attempts: %s", retries, url)
    return None


def extract_year(value: str) -> str|None:
    """Extract year (YYYY) from text like 'August 2019', '2019-08-01', '2019'."""
    if not value:
        return None
    match = re.search(r"(19|20)\d{2}", value)
    return int(match.group(0)) if match else None


def extract_characters(names: str) -> list[str]:
    """Extract clean list of characters from semicolon-separated input."""
    if not names:
        return []
    characters = []
    for name in names.split(";"):
        clean = name.split("[")[0].strip()
        if clean:
            characters.append(clean)
    return characters

def to_float(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value.split(" ")[0].strip())
    except Exception:
        return None

def search_comic_issue_by_barcode(barcode: str) -> str|None:
    """
    Search comics.org for an issue by barcode.
    Returns the first issue link (e.g., /issue/1666841/) or None.
    """
    session = get_session()
    params = {"search_object": "issue", "q": barcode, "sort": "relevance"}

    logger.info("Searching for barcode: %s", barcode)
    resp = safe_request(session, f"{BASE_COMIC_URL}/searchNew/", params=params)
    if not resp:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table", {"class": "border"})

    for table in tables:
        preview_tag = table.find("a", {"title": "preview"})
        if not preview_tag:
            continue

        if preview_tag.get_text(strip=True) == "[ISSUE]":
            issue_link = table.find("a", href=re.compile(r"^/issue/\d+/$"))
            if issue_link:
                link = issue_link["href"]
                logger.info("Found issue link: %s", link)
                return link

    logger.warning("No issue found for barcode %s", barcode)
    return None


def issue_details(link: str, auth: tuple[str, str]) -> dict[str, any]|None:
    """
    Fetch issue details from the comics.org API.
    Requires authentication.
    """
    session = get_session(auth)
    params = {"format": "json"}

    url = f"{BASE_COMIC_API_URL}{link}" if not link.startswith(BASE_COMIC_API_URL) else link
    logger.info("Fetching issue details: %s", url)

    resp = safe_request(session, url, params=params)
    if not resp:
        return None

    try:
        return resp.json()
    except Exception as e:
        logger.exception("Failed to parse JSON: %s", e)
        return None


def extract_data(issue_data: dict[str, any]) -> dict[str, str|list|None]:
    """Normalize and extract structured comic issue data."""
    descriptor: str = issue_data.get("descriptor", "") or ""
    issue_number = ""
    features = []

    try:
        issue_number = str(int(descriptor.split("[")[0]))
    except Exception:
        issue_number = ""

    if "[" in descriptor:
        features.append(descriptor.split("[")[-1].replace("[", "").replace("]", ""))

    # Handle variants
    variant_of = issue_data.get("variant_of")
    if variant_of:
        issue_data = issue_details(link=variant_of, auth=AUTH) or issue_data
    
    series_name = (issue_data.get("series_name") or "").split("(")[0].strip()
    publication_date = issue_data.get("publication_date", "")
    publication_year = extract_year(publication_date) if publication_date else None
    era = map_era(publication_year) if publication_year else None
    
    publisher = issue_data.get("indicia_publisher", "")
    barcode = issue_data.get("barcode", "")
    price = to_float(issue_data.get("price", ""))
    page_count = to_float(issue_data.get("page_count", ""))


    writer = None
    genre = None
    characters: list[str] = []

    for story in issue_data.get("story_set", []):
        if story.get("type") != "comic story":
            continue
        writer = story.get("script", "") or writer
        genre = story.get("genre", "") or genre
        characters = extract_characters(story.get("characters", "")) or characters

    country:str|None = None
    language:str|None = None
    style:str|None = None

    series:str = issue_data.get("series")
    if series:
        series_data:dict = issue_details(link=series, auth=AUTH)
        if series_data:
            raw_country:str = series_data.get("country", "")
            raw_language:str = series_data.get("language", "")
            country = iso_country_to_name(raw_country)
            language = iso_language_to_name(raw_language)
            style = (series_data.get("color") or "").capitalize()

    return {
        "issue_number": issue_number,
        "features": features,
        "series_name": series_name,
        "publication_year": publication_year,
        "publisher": publisher,
        "barcode": barcode,
        "writer": writer,
        "genre": genre,
        "characters": characters,
        "country": country,
        "language": language,
        "style": style,
        "era":era,
        "price":price,
        "page_count": page_count,
    }

if __name__ == "__main__":
    barcode = "75960608309100112"
    link = search_comic_issue_by_barcode(barcode)

    if link:
        details = issue_details(link, AUTH)
        if details:
            logger.info("Raw issue details retrieved")
            structured = extract_data(details)
            print("\nExtracted Data:\n", structured)
        else:
            logger.error("Failed to fetch issue details for %s", link)
    else:
        logger.error("No issue link found for barcode %s", barcode)
