"""
Markdown processing utilities for Open Practice Library content.

This module contains functions for converting HTML to Markdown and
cleaning/formatting markdown content.
"""

import importlib
import logging
import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Import html_processing functions
try:
    from .html_processing import extract_opl_content
except ImportError:
    try:
        from . import html_processing

        extract_opl_content = html_processing.extract_opl_content
    except ImportError:
        import sys

        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.append(str(current_dir))
        html_module = importlib.import_module("html_processing")
        extract_opl_content = html_module.extract_opl_content

# Configure module logger
_log = logging.getLogger(__name__)

# Constants
HTML_PARSER = "html.parser"
MULTIPLE_NEWLINES_PATTERN = r"\n{3,}"
SECTION_WHAT_IS = "What Is"
SECTION_WHY_DO = "Why Do"
SECTION_HOW_TO = "How to do"


def _process_heading(element, soup):
    """Process a heading element into markdown."""
    level = int(element.name[1])
    if soup.select_one("div.MuiTypography-body1") and element.parent.name != "body":
        level += 1
    heading_text = element.get_text().strip()
    return f"{'#' * level} {heading_text}\n"


def _process_paragraph(element, processed_elements):
    """Process a paragraph element into markdown."""
    p_content = ""
    for child in element.children:
        if child.name == "strong" or (isinstance(child, str) and child.parent.name == "strong"):
            p_content += f"**{child.get_text().strip()}**"
        elif child.name == "em" or (isinstance(child, str) and child.parent.name == "em"):
            p_content += f"**{child.get_text().strip()}**"
        elif isinstance(child, str):
            p_content += child
        else:
            processed_elements.add(id(child))
            p_content += child.get_text()
    return p_content.strip()


def _should_process_element(element, processed_elements):
    """Determine if an element should be processed."""
    if id(element) in processed_elements:
        return False
    if element.parent.name in ["li", "ul", "ol"] and element.parent.parent.name != "body":
        return False
    if element.name == "p" and element.parent.name == "li" and len(element.find_next_siblings()) > 0:
        return False
    return True


def _process_element(element, soup, processed_elements):
    """Process a single element into markdown lines."""
    if not _should_process_element(element, processed_elements):
        return []

    processed_elements.add(id(element))
    markdown_lines = []

    if element.name.startswith("h"):
        markdown_lines.append(_process_heading(element, soup))
        markdown_lines.append("")
    elif element.name == "p":
        paragraph_text = _process_paragraph(element, processed_elements)
        if paragraph_text:
            markdown_lines.append(paragraph_text)
            markdown_lines.append("")
    elif element.name == "ul" and element.parent.name != "li":
        list_items = process_list(element, global_processed=processed_elements)
        if list_items:
            markdown_lines.extend(list_items)
            markdown_lines.append("")
    elif element.name == "ol" and element.parent.name != "li":
        list_items = process_ordered_list(element, global_processed=processed_elements)
        if list_items:
            markdown_lines.extend(list_items)
            markdown_lines.append("")

    return markdown_lines


def convert_html_to_markdown(html_content):
    """
    Convert HTML content to Markdown format.

    Args:
        html_content: HTML content as a string

    Returns:
        str: Markdown formatted content
    """
    soup = BeautifulSoup(html_content, HTML_PARSER)
    processed_elements = set()
    markdown_output = []

    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "em", "strong"], recursive=True):
        markdown_output.extend(_process_element(element, soup, processed_elements))

    result = "\n".join(markdown_output)
    return re.sub(MULTIPLE_NEWLINES_PATTERN, "\n\n", result)


def _process_paragraph_content(p_child, processed):
    """Process content within a paragraph element."""
    if p_child.name == "strong" or (isinstance(p_child, str) and p_child.parent.name == "strong"):
        return f"**{p_child.get_text().strip()}**"
    elif p_child.name == "em" or (isinstance(p_child, str) and p_child.parent.name == "em"):
        return f"**{p_child.get_text().strip()}**"
    elif isinstance(p_child, str):
        return p_child
    else:
        processed.add(id(p_child))
        return p_child.get_text()


def _process_list_item_text(li, processed):
    """Extract and process text content from a list item."""
    content = []
    direct_text = ""

    for child in li.children:
        if isinstance(child, str):
            direct_text += child.strip() + " "
        elif child.name == "p":
            processed.add(id(child))
            p_content = "".join(_process_paragraph_content(p_child, processed) for p_child in child.children)
            if p_content.strip():
                content.append(p_content.strip())

    if direct_text.strip():
        content.insert(0, direct_text.strip())
    return content


def _format_list_item(content, indent):
    """Format list item content with proper indentation."""
    if not content:
        return []
    lines = ["  " * indent + "* " + content[0]]
    lines.extend("  " * indent + "  " + line for line in content[1:])
    return lines


def process_list(ul_element, indent=0, global_processed=None):
    """
    Process an unordered list element and its children recursively.

    Args:
        ul_element: BeautifulSoup ul element
        indent: Current indentation level
        global_processed: Set of processed element IDs shared across function calls

    Returns:
        list: Lines of markdown formatted list
    """
    markdown_lines = []
    processed = global_processed if global_processed is not None else set()

    for li in ul_element.find_all("li", recursive=False):
        if id(li) in processed:
            continue

        processed.add(id(li))
        content = _process_list_item_text(li, processed)
        markdown_lines.extend(_format_list_item(content, indent))

        # Process nested lists
        for nested_ul in li.find_all("ul", recursive=False):
            processed.add(id(nested_ul))
            markdown_lines.extend(process_list(nested_ul, indent + 1, processed))

        for nested_ol in li.find_all("ol", recursive=False):
            processed.add(id(nested_ol))
            markdown_lines.extend(process_ordered_list(nested_ol, indent + 1, processed))

    return markdown_lines


def _format_ordered_list_item(content, indent, number):
    """Format ordered list item content with proper indentation and numbering."""
    if not content:
        return []
    lines = ["  " * indent + f"{number}. " + content[0]]
    lines.extend("  " * indent + "   " + line for line in content[1:])
    return lines


def process_ordered_list(ol_element, indent=0, global_processed=None):
    """
    Process an ordered list element and its children recursively.

    Args:
        ol_element: BeautifulSoup ol element
        indent: Current indentation level
        global_processed: Set of processed element IDs shared across function calls

    Returns:
        list: Lines of markdown formatted ordered list
    """
    markdown_lines = []
    processed = global_processed if global_processed is not None else set()

    for i, li in enumerate(ol_element.find_all("li", recursive=False), 1):
        if id(li) in processed:
            continue

        processed.add(id(li))
        content = _process_list_item_text(li, processed)
        markdown_lines.extend(_format_ordered_list_item(content, indent, i))

        # Process nested lists
        for nested_ul in li.find_all("ul", recursive=False):
            processed.add(id(nested_ul))
            markdown_lines.extend(process_list(nested_ul, indent + 1, processed))

        for nested_ol in li.find_all("ol", recursive=False):
            processed.add(id(nested_ol))
            markdown_lines.extend(process_ordered_list(nested_ol, indent + 1, processed))

    return markdown_lines


def clean_text(text):
    """
    Clean up text by removing special/invisible characters.

    Args:
        text: Text to clean

    Returns:
        str: Cleaned text
    """
    if not text:
        return text

    # Remove Zero Width No-Break Space (U+FEFF, BOM)
    text = text.replace("\ufeff", "")

    # Remove other problematic invisible characters
    invisible_chars = [
        "\u200b",  # Zero Width Space
        "\u200c",  # Zero Width Non-Joiner
        "\u200d",  # Zero Width Joiner
        "\u2060",  # Word Joiner
        "\u2061",  # Function Application
        "\u2062",  # Invisible Times
        "\u2063",  # Invisible Separator
        "\u2064",  # Invisible Plus
        "\u2065",  # Invisible Times
        "\u2066",  # Left-to-Right Isolate
        "\u2067",  # Right-to-Left Isolate
        "\u2068",  # First Strong Isolate
        "\u2069",  # Pop Directional Isolate
        "\u206a",  # Inhibit Symmetric Swapping
        "\u206b",  # Activate Symmetric Swapping
        "\u206c",  # Inhibit Arabic Form Shaping
        "\u206d",  # Activate Arabic Form Shaping
        "\u206e",  # National Digit Shapes
        "\u206f",  # Nominal Digit Shapes
    ]

    for char in invisible_chars:
        text = text.replace(char, "")

    # Replace non-breaking spaces with regular spaces
    text = text.replace("\xa0", " ")

    # Replace multiple spaces with a single space
    text = re.sub(r" +", " ", text)

    return text


def clean_markdown(markdown_text):
    """
    Clean and normalize markdown text.

    Args:
        markdown_text: Markdown text to clean

    Returns:
        str: Cleaned markdown text
    """
    if not markdown_text:
        return markdown_text

    # Clean special characters
    markdown_text = clean_text(markdown_text)

    # Fix common markdown issues

    # Ensure proper line breaks before headers
    markdown_text = re.sub(r"([^\n])(\n#{1,6}\s)", r"\1\n\n\2", markdown_text)

    # Ensure proper line breaks after headers
    markdown_text = re.sub(r"(#{1,6}\s.+?)(\n[^#\n])", r"\1\n\2", markdown_text)

    # Normalize bullet lists (ensure consistent spacing)
    markdown_text = re.sub(r"(\n\s*)\*\s", r"\1* ", markdown_text)

    # Do NOT normalize numbered lists - this allows correct incrementation
    # markdown_text = re.sub(r"(\n\s*)\d+\.\s", r"\g<1>1. ", markdown_text)

    # Remove excessive blank lines (more than 2)
    markdown_text = re.sub(MULTIPLE_NEWLINES_PATTERN, r"\n\n", markdown_text)

    # Convert underscores to double asterisks for consistent emphasis
    markdown_text = re.sub(r"_(.*?)_", r"**\1**", markdown_text)

    # Ensure document ends with newline
    if not markdown_text.endswith("\n"):
        markdown_text += "\n"

    return markdown_text


def _format_markdown_section(heading, content, title):
    """Format a section in markdown format."""
    if SECTION_WHAT_IS in heading:
        clean_heading = f"### {SECTION_WHAT_IS} {title}"
    elif SECTION_WHY_DO in heading:
        clean_heading = f"### {SECTION_WHY_DO} {title}"
    elif SECTION_HOW_TO in heading:
        clean_heading = f"### {SECTION_HOW_TO} {title}"
    else:
        clean_heading = f"### {heading}"

    return [clean_heading, "", re.sub(MULTIPLE_NEWLINES_PATTERN, "\n\n", content), ""]


def _format_text_section(heading, content, title):
    """Format a section in plain text format."""
    if SECTION_WHAT_IS in heading:
        clean_heading = f"{SECTION_WHAT_IS} {title}"
    elif SECTION_WHY_DO in heading:
        clean_heading = f"{SECTION_WHY_DO} {title}"
    elif SECTION_HOW_TO in heading:
        clean_heading = f"{SECTION_HOW_TO} {title}"
    else:
        clean_heading = heading

    # Clean up content for plain text
    content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)  # Remove bold markers
    content = re.sub(r"_(.*?)_", r"\1", content)  # Remove italic markers
    content = re.sub(r"#{1,6}\s+", "", content)  # Remove markdown headings

    return [clean_heading, "", content, ""]


def _get_section_formatter(output_format, title):
    """Get the appropriate section formatter function based on output format."""
    if output_format == "markdown":

        def format_section(heading, content):
            return _format_markdown_section(heading, content, title)
    else:  # text format

        def format_section(heading, content):
            return _format_text_section(heading, content, title)

    return format_section


def format_output(practice_content, output_format="markdown"):
    """Format practice content into markdown output."""
    output = []

    if output_format == "markdown":
        # Document title and subtitle with proper spacing
        output.append("")  # Blank line before title
        output.append(f"# {practice_content['title']}")
        output.append("")  # Blank line after title
        output.append(f"## {practice_content['subtitle']}")
        output.append("")  # Blank line after subtitle

        # Process each section with proper headings
        for heading, content in practice_content["sections"].items():
            # Format the main section headings
            if SECTION_WHAT_IS in heading:
                clean_heading = f"### {SECTION_WHAT_IS} {practice_content['title']}"
            elif SECTION_WHY_DO in heading:
                clean_heading = f"### {SECTION_WHY_DO} {practice_content['title']}"
            elif SECTION_HOW_TO in heading:
                clean_heading = f"### {SECTION_HOW_TO} {practice_content['title']}"
            else:
                clean_heading = f"### {heading}"

            output.append("")  # Blank line before heading
            output.append(clean_heading)
            output.append("")  # Blank line after heading

            # Add content
            import re

            content = re.sub(MULTIPLE_NEWLINES_PATTERN, "\n\n", content)
            output.append(content)
            output.append("")  # Blank line after content

    # Final cleanup to ensure consistent formatting
    result = "\n".join(output)

    # Remove any extra blank lines while preserving heading spacing
    import re

    result = re.sub(MULTIPLE_NEWLINES_PATTERN, "\n\n", result)

    # Ensure the file ends with a newline
    if not result.endswith("\n"):
        result += "\n"

    return result


def _get_html_content(source):
    """Get HTML content from a source (URL or file path)."""
    if isinstance(source, Path) or (isinstance(source, str) and os.path.exists(source)):
        file_path = Path(source) if isinstance(source, str) else source
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    else:
        response = requests.get(source, timeout=30)
        return response.text


def _get_practice_title(practice_content, source):
    """Get or generate a practice title."""
    if practice_content["title"] and practice_content["title"] != "Unknown Title":
        return practice_content["title"]

    if isinstance(source, Path) or (isinstance(source, str) and os.path.exists(source)):
        file_path = Path(source) if isinstance(source, str) else source
        filename_title = file_path.stem.replace("_", " ").replace("-", " ").title()
        _log.info(f"Using filename as title: {filename_title}")
        return filename_title

    _log.warning(f"Could not extract title for {source}")
    return None


def _save_markdown_file(practice_content, output_dir):
    """Save practice content as a markdown file."""
    practice_name = practice_content["title"].lower().replace(" ", "-").replace("$", "dollar-")
    practice_name = re.sub(r"[^\w\-]", "", practice_name)

    # Clean all content
    for section_key, section_content in practice_content["sections"].items():
        practice_content["sections"][section_key] = clean_text(section_content)

    markdown_content = format_output(practice_content, "markdown")
    output_path = output_dir / f"{practice_name}.md"

    with output_path.open("w", encoding="utf-8") as fp:
        fp.write(markdown_content)

    return output_path


def process_practice(source, output_dir):
    """
    Process a single practice source (URL or file path) and generate a markdown file.

    Args:
        source: URL or Path to the HTML file to process
        output_dir: Directory to save the markdown file

    Returns:
        bool: Success status
    """
    try:
        _log.info(f"Processing {source}")

        # Get HTML content
        html_content = _get_html_content(source)

        # Extract and process content
        practice_content = extract_opl_content(html_content)
        title = _get_practice_title(practice_content, source)

        if not title:
            return False

        practice_content["title"] = title

        # Save markdown file
        output_path = _save_markdown_file(practice_content, output_dir)
        _log.info(f"Successfully processed {title} -> {output_path}")
        return True

    except Exception as e:
        _log.error(f"Error processing {source}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False
