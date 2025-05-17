"""
HTML processing utilities for Open Practice Library content.

This module contains functions for extracting and processing HTML content
from the Open Practice Library website.
"""

import importlib
import logging
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

# Configure module logger
_log = logging.getLogger(__name__)

# Constants
HTML_PARSER = "html.parser"


def _load_markdown_processing():
    """Load markdown processing functions, handling different import scenarios."""
    try:
        from .markdown_processing import (
            clean_markdown,
            clean_text,
            convert_html_to_markdown,
        )

        return clean_markdown, clean_text, convert_html_to_markdown
    except ImportError:
        try:
            from .markdown_processing import (
                clean_markdown,
                clean_text,
                convert_html_to_markdown,
            )

            return clean_markdown, clean_text, convert_html_to_markdown
        except ImportError:
            import sys
            from pathlib import Path

            current_dir = Path(__file__).parent
            if str(current_dir) not in sys.path:
                sys.path.append(str(current_dir))
            markdown_module = importlib.import_module("markdown_processing")
            return (
                markdown_module.clean_markdown,
                markdown_module.clean_text,
                markdown_module.convert_html_to_markdown,
            )


def _extract_title_subtitle(soup, clean_text):
    """Extract title and subtitle from soup."""
    title_elem = soup.select_one('[data-testid="title"]')
    subtitle_elem = soup.select_one('[data-testid="subtitle"]')

    if not title_elem or not subtitle_elem:
        _log.warning("Could not find title or subtitle elements")
        return "Unknown Title", "Unknown Subtitle"

    return clean_text(title_elem.text.strip()), clean_text(subtitle_elem.text.strip())


def _find_main_container(soup):
    """Find the main content container in the soup."""
    for container in soup.select(".MuiContainer-root.MuiContainer-maxWidthMd"):
        box = container.select_one(".MuiBox-root")
        if box and box.select("h4.MuiTypography-h4"):
            return box
    _log.error("Could not find main content container")
    return None


def _process_section_content(content_divs, convert_html_to_markdown, clean_markdown):
    """Process a section's content divs into cleaned markdown."""
    section_html = "".join(str(div) for div in content_divs)
    section_html = f"<div>{section_html}</div>"
    section_content = convert_html_to_markdown(section_html)
    section_content = clean_markdown(section_content)
    section_content = re.sub(r"#+\s*\n+", "", section_content)
    return section_content.strip()


def extract_opl_content(html_content):
    """
    Extract the relevant practice content from Open Practice Library HTML.
    Returns a dict with title, subtitle, sections and their content.

    Args:
        html_content: HTML content as a string

    Returns:
        dict: Dictionary containing title, subtitle, and sections with their content
    """
    clean_markdown, clean_text, convert_html_to_markdown = _load_markdown_processing()
    soup = BeautifulSoup(html_content, HTML_PARSER)

    title, subtitle = _extract_title_subtitle(soup, clean_text)
    result = {"title": title, "subtitle": subtitle, "sections": {}}

    main_container = _find_main_container(soup)
    if not main_container:
        return result

    current_heading = None
    section_elements = {}

    for element in main_container.children:
        if not hasattr(element, "name"):
            continue

        if element.name == "h4" and "MuiTypography-h4" in element.get("class", []):
            current_heading = clean_text(element.get_text(strip=True))
            section_elements[current_heading] = []
        elif current_heading and element.name == "div" and "MuiTypography-body1" in element.get("class", []):
            section_elements[current_heading].append(element)

    for heading, content_divs in section_elements.items():
        if not content_divs:
            continue
        result["sections"][heading] = _process_section_content(content_divs, convert_html_to_markdown, clean_markdown)

    return result


def get_all_practice_urls(homepage_url="https://openpracticelibrary.com/"):
    """
    Fetch all practice URLs from the Open Practice Library homepage.

    Args:
        homepage_url: URL of the OPL homepage

    Returns:
        list: List of practice URLs
    """
    _log.info(f"Fetching practice URLs from {homepage_url}")

    # Import here to avoid circular imports
    import requests

    response = requests.get(homepage_url, timeout=30)
    soup = BeautifulSoup(response.text, HTML_PARSER)

    # Find all practice card links
    practice_links = soup.select('div[data-testid="practicecardgrid"] a')

    _log.info(f"Found {len(practice_links)} practice links")

    # Extract and normalize URLs
    practice_urls = []
    for link in practice_links:
        href = link.get("href")
        if href and href.startswith("/practice/"):
            full_url = urljoin(homepage_url, href)
            practice_urls.append(full_url)

    _log.info(f"Extracted {len(practice_urls)} practice URLs")
    return practice_urls
