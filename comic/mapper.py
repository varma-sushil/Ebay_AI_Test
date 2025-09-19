import pycountry

def map_era(publication_year: str|None) -> str|None:
    """Map publication year to eBay Era aspect value."""
    if not publication_year:
        return None

    try:
        year = int(publication_year)
    except ValueError:
        return None

    if 1897 <= year <= 1937:
        return "Platinum Age (1897-1937)"
    elif 1938 <= year <= 1955:
        return "Golden Age (1938-55)"
    elif 1956 <= year <= 1969:
        return "Silver Age (1956-69)"
    elif 1970 <= year <= 1983:
        return "Bronze Age (1970-83)"
    elif 1984 <= year <= 1991:
        return "Copper Age (1984-1991)"
    elif year >= 1992:
        return "Modern Age (1992-Now)"
    return None

def iso_country_to_name(code: str) -> str | None:
    """Convert ISO 2-letter country code to official full name."""
    if not code:
        return None
    try:
        country = pycountry.countries.get(alpha_2=code.upper())
        return country.name if country else None
    except Exception:
        return None

def iso_language_to_name(code: str) -> str | None:
    """Convert ISO 2- or 3-letter language code to full name."""
    if not code:
        return None
    try:
        lang = pycountry.languages.get(alpha_2=code.lower()) or pycountry.languages.get(alpha_3=code.lower())
        return lang.name if lang else None
    except Exception:
        return None
