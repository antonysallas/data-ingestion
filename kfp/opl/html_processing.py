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


def extract_opl_content(html_content):
    """
    Extract the relevant practice content from Open Practice Library HTML.
    Returns a dict with title, subtitle, sections and their content.

    Args:
        html_content: HTML content as a string

    Returns:
        dict: Dictionary containing title, subtitle, and sections with their content
    """
    # Import functions from markdown_processing, handling both package and direct imports
    try:
        # Try package import first
        from opl.markdown_processing import (
            clean_markdown,
            clean_text,
            convert_html_to_markdown,
        )
    except ImportError:
        # Fall back to direct import
        try:
            from markdown_processing import (
                clean_markdown,
                clean_text,
                convert_html_to_markdown,
            )
        except ImportError:
            # If both fail, try to load the module dynamically
            import sys
            from pathlib import Path

            # Add current directory to path if needed
            current_dir = Path(__file__).parent
            if str(current_dir) not in sys.path:
                sys.path.append(str(current_dir))

            markdown_module = importlib.import_module("markdown_processing")
            clean_text = markdown_module.clean_text
            clean_markdown = markdown_module.clean_markdown
            convert_html_to_markdown = markdown_module.convert_html_to_markdown

    soup = BeautifulSoup(html_content, HTML_PARSER)

    # Extract title and subtitle
    title_elem = soup.select_one('[data-testid="title"]')
    subtitle_elem = soup.select_one('[data-testid="subtitle"]')

    if not title_elem or not subtitle_elem:
        _log.warning("Could not find title or subtitle elements")
        title = "Unknown Title"
        subtitle = "Unknown Subtitle"
    else:
        title = clean_text(title_elem.text.strip())
        subtitle = clean_text(subtitle_elem.text.strip())

    result = {"title": title, "subtitle": subtitle, "sections": {}}

    # Find the main content container
    main_container = None
    for container in soup.select(".MuiContainer-root.MuiContainer-maxWidthMd"):
        box = container.select_one(".MuiBox-root")
        if box and box.select("h4.MuiTypography-h4"):
            main_container = box
            break

    if not main_container:
        _log.error("Could not find main content container")
        return result

    # Use document flow to collect sections and their content
    current_heading = None
    section_elements = {}

    for element in main_container.children:
        if not hasattr(element, "name"):  # Skip non-tag elements like NavigableString
            continue

        if element.name == "h4" and "MuiTypography-h4" in element.get("class", []):
            # Save the heading as a new section
            current_heading = clean_text(element.get_text(strip=True))
            section_elements[current_heading] = []

        elif current_heading and element.name == "div" and "MuiTypography-body1" in element.get("class", []):
            # Save this content div as part of the current section
            section_elements[current_heading].append(element)

    # Now process each section's content
    for heading, content_divs in section_elements.items():
        if len(content_divs) == 0:
            continue

        # Combine all content divs for this section
        section_html = ""
        for div in content_divs:
            section_html += str(div)

        # Wrap in a div for proper parsing
        section_html = f"<div>{section_html}</div>"

        # Convert to markdown with proper formatting
        section_content = convert_html_to_markdown(section_html)

        # Clean and normalize the markdown
        section_content = clean_markdown(section_content)

        # Remove any empty headings that might have been added
        section_content = re.sub(r"#+\s*\n+", "", section_content)

        # Remove any leading/trailing whitespace
        section_content = section_content.strip()

        result["sections"][heading] = section_content

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

    response = requests.get(homepage_url)
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
