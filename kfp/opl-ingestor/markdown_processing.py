"""
Markdown processing utilities for Open Practice Library content.

This module contains functions for converting HTML to Markdown and
cleaning/formatting markdown content.
"""

import importlib
import logging
import os
import re

from bs4 import BeautifulSoup

# Configure module logger
_log = logging.getLogger(__name__)

# Constants
HTML_PARSER = "html.parser"


def convert_html_to_markdown(html_content):
    """
    Convert HTML content to Markdown format.

    Args:
        html_content: HTML content as a string

    Returns:
        str: Markdown formatted content
    """
    soup = BeautifulSoup(html_content, HTML_PARSER)
    markdown_output = []
    processed_elements = set()  # Track which elements we've already processed

    # Process elements in document order for better content flow
    for element in soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "em", "strong"], recursive=True
    ):
        # Skip already processed elements to avoid duplication
        if id(element) in processed_elements:
            continue

        processed_elements.add(id(element))

        # Skip elements that are part of a parent we'll process separately
        if element.parent.name in ["li", "ul", "ol"] and element.parent.parent.name != "body":
            continue

        # Skip paragraph elements that are just headers of list items (these create duplicates)
        if element.name == "p" and element.parent.name == "li" and len(element.find_next_siblings()) > 0:
            # Don't process paragraph text separately when it's a list item header with content
            continue

        if element.name.startswith("h"):
            # Get heading level
            level = int(element.name[1])
            # For headings inside content sections (h3, h4, etc.), increase level by 1
            if soup.select_one("div.MuiTypography-body1") and element.parent.name != "body":
                level += 1
            heading_text = element.get_text().strip()
            markdown_output.append("#" * level + " " + heading_text)
            markdown_output.append("")  # Empty line after heading

        elif element.name == "p":
            # Handle paragraphs with formatting
            p_content = ""
            for child in element.children:
                if child.name == "strong" or (isinstance(child, str) and child.parent.name == "strong"):
                    # Handle strong text
                    p_content += f"**{child.get_text().strip()}**"
                elif child.name == "em" or (isinstance(child, str) and child.parent.name == "em"):
                    # Handle emphasized text - use double asterisks for all emphasis
                    p_content += f"**{child.get_text().strip()}**"
                elif isinstance(child, str):
                    # Regular text
                    p_content += child
                else:
                    # Other elements inside paragraph
                    processed_elements.add(id(child))  # Mark child elements as processed
                    p_content += child.get_text()

            paragraph_text = p_content.strip()
            if paragraph_text:
                markdown_output.append(paragraph_text)
                markdown_output.append("")  # Empty line after paragraph

        elif element.name == "ul" and element.parent.name != "li":
            # Only process top-level lists to avoid duplication
            list_items = process_list(element, global_processed=processed_elements)
            if list_items:
                markdown_output.extend(list_items)
                markdown_output.append("")  # Empty line after list

        elif element.name == "ol" and element.parent.name != "li":
            # Process ordered lists
            list_items = process_ordered_list(element, global_processed=processed_elements)
            if list_items:
                markdown_output.extend(list_items)
                markdown_output.append("")  # Empty line after list

    # Clean up: remove consecutive empty lines
    result = "\n".join(markdown_output)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result


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
    # Use a shared set of processed elements if provided, or create a new one
    processed = global_processed if global_processed is not None else set()

    for li in ul_element.find_all("li", recursive=False):
        if id(li) in processed:
            continue

        processed.add(id(li))
        # Get all content from the list item
        content = []

        # Handle direct text first
        direct_text = ""
        for child in li.children:
            if isinstance(child, str):
                direct_text += child.strip() + " "
            elif child.name == "p":
                # Track as processed to avoid duplication
                processed.add(id(child))
                # Handle formatted content within paragraphs
                p_content = ""
                for p_child in child.children:
                    if p_child.name == "strong" or (
                        isinstance(p_child, str) and p_child.parent.name == "strong"
                    ):
                        # Handle strong text
                        p_content += f"**{p_child.get_text().strip()}**"
                    elif p_child.name == "em" or (isinstance(p_child, str) and p_child.parent.name == "em"):
                        # Handle emphasized text - use bold formatting
                        p_content += f"**{p_child.get_text().strip()}**"
                    elif isinstance(p_child, str):
                        # Regular text
                        p_content += p_child
                    else:
                        # Other elements
                        processed.add(id(p_child))  # Mark content as processed
                        p_content += p_child.get_text()

                if p_content.strip():
                    content.append(p_content.strip())

        if direct_text.strip():
            content.insert(0, direct_text.strip())

        # Add the list item with proper indentation
        if content:
            # For the first line of content
            markdown_lines.append("  " * indent + "* " + content[0])

            # For additional lines of content (if any)
            for line in content[1:]:
                markdown_lines.append("  " * indent + "  " + line)

        # Process nested unordered lists
        for nested_ul in li.find_all("ul", recursive=False):
            processed.add(id(nested_ul))
            markdown_lines.extend(process_list(nested_ul, indent + 1, processed))

        # Process nested ordered lists
        for nested_ol in li.find_all("ol", recursive=False):
            processed.add(id(nested_ol))
            markdown_lines.extend(process_ordered_list(nested_ol, indent + 1, processed))

    return markdown_lines


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
    # Use a shared set of processed elements if provided, or create a new one
    processed = global_processed if global_processed is not None else set()

    for i, li in enumerate(ol_element.find_all("li", recursive=False), 1):
        if id(li) in processed:
            continue

        processed.add(id(li))
        # Get all content from the list item
        content = []

        # Handle direct text first
        direct_text = ""
        for child in li.children:
            if isinstance(child, str):
                direct_text += child.strip() + " "
            elif child.name == "p":
                # Track as processed to avoid duplication
                processed.add(id(child))
                # Handle formatted content within paragraphs
                p_content = ""
                for p_child in child.children:
                    if p_child.name == "strong" or (
                        isinstance(p_child, str) and p_child.parent.name == "strong"
                    ):
                        # Handle strong text
                        p_content += f"**{p_child.get_text().strip()}**"
                    elif p_child.name == "em" or (isinstance(p_child, str) and p_child.parent.name == "em"):
                        # Handle emphasized text - use bold formatting
                        p_content += f"**{p_child.get_text().strip()}**"
                    elif isinstance(p_child, str):
                        # Regular text
                        p_content += p_child
                    else:
                        # Other elements
                        processed.add(id(p_child))  # Mark content as processed
                        p_content += p_child.get_text()

                if p_content.strip():
                    content.append(p_content.strip())

        if direct_text.strip():
            content.insert(0, direct_text.strip())

        # Add the list item with proper indentation and numbering
        if content:
            # For the first line of content - use actual item number for correct numbering
            markdown_lines.append("  " * indent + f"{i}. " + content[0])

            # For additional lines of content (if any)
            for line in content[1:]:
                markdown_lines.append("  " * indent + "   " + line)

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
    markdown_text = re.sub(r"\n{3,}", r"\n\n", markdown_text)

    # Convert underscores to double asterisks for consistent emphasis
    markdown_text = re.sub(r"_(.*?)_", r"**\1**", markdown_text)

    # Ensure document ends with newline
    if not markdown_text.endswith("\n"):
        markdown_text += "\n"

    return markdown_text


def format_output(practice_content, output_format="markdown"):
    """
    Format the practice content into the specified output format.
    Formats: 'markdown', 'text'

    Args:
        practice_content: Dictionary containing practice content
        output_format: Output format ('markdown' or 'text')

    Returns:
        str: Formatted content
    """
    output = []

    if output_format == "markdown":
        # Document title and subtitle
        output.append(f"# {practice_content['title']}")
        output.append(f"## {practice_content['subtitle']}")
        output.append("")  # Empty line

        # Process each section with proper headings
        for heading, content in practice_content["sections"].items():
            # Format the main section headings
            if "What Is" in heading:
                clean_heading = f"### What Is {practice_content['title']}"
            elif "Why Do" in heading:
                clean_heading = f"### Why Do {practice_content['title']}"
            elif "How to do" in heading:
                clean_heading = f"### How to do {practice_content['title']}"
            else:
                clean_heading = f"### {heading}"

            output.append(clean_heading)
            output.append("")  # Empty line after heading

            # Add content with proper formatting
            # Ensure there's no duplicate line breaks
            content = re.sub(r"\n{3,}", "\n\n", content)
            output.append(content)
            output.append("")  # Empty line after content

    elif output_format == "text":
        # Plain text format
        output.append(f"{practice_content['title']}")
        output.append(f"{practice_content['subtitle']}")
        output.append("")

        for heading, content in practice_content["sections"].items():
            # Format section headings for plain text
            if "What Is" in heading:
                clean_heading = f"What Is {practice_content['title']}"
            elif "Why Do" in heading:
                clean_heading = f"Why Do {practice_content['title']}"
            elif "How to do" in heading:
                clean_heading = f"How to do {practice_content['title']}"
            else:
                clean_heading = heading

            output.append(clean_heading)
            output.append("")

            # Clean up the content for plain text
            content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)  # Remove bold markers
            content = re.sub(r"_(.*?)_", r"\1", content)  # Remove italic markers
            content = re.sub(r"#{1,6}\s+", "", content)  # Remove markdown headings

            output.append(content)
            output.append("")

    # Final cleanup to ensure consistent formatting
    result = "\n".join(output)

    # Remove any extra blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)

    # Ensure the file ends with a newline
    if not result.endswith("\n"):
        result += "\n"

    return result


def process_practice(source, output_dir):
    """
    Process a single practice source (URL or file path) and generate a markdown file.

    Args:
        source: URL or Path to the HTML file to process
        output_dir: Directory to save the markdown file

    Returns:
        bool: Success status
    """
    # Import here to avoid circular imports
    from pathlib import Path

    import requests

    # Attempt to import html_processing - try different approaches
    try:
        # Try package import first
        from opl_ingestor.html_processing import extract_opl_content
    except ImportError:
        try:
            # Fall back to direct import
            from html_processing import extract_opl_content
        except ImportError:
            # If both fail, try to load the module dynamically
            import sys

            # Add current directory to path if needed
            current_dir = Path(__file__).parent
            if str(current_dir) not in sys.path:
                sys.path.append(str(current_dir))

            html_module = importlib.import_module("html_processing")
            extract_opl_content = html_module.extract_opl_content

    try:
        _log.info(f"Processing {source}")

        # Determine if source is a file path or URL
        if isinstance(source, Path) or (isinstance(source, str) and os.path.exists(source)):
            # Local file
            file_path = Path(source) if isinstance(source, str) else source

            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            source_id = str(file_path)
        else:
            # Assume URL
            response = requests.get(source)
            html_content = response.text
            source_id = source

        # Extract content
        practice_content = extract_opl_content(html_content)

        if not practice_content["title"] or practice_content["title"] == "Unknown Title":
            # For local files, try to extract a title from the filename
            if isinstance(source, Path) or (isinstance(source, str) and os.path.exists(source)):
                file_path = Path(source) if isinstance(source, str) else source
                filename_title = file_path.stem.replace("_", " ").replace("-", " ").title()
                practice_content["title"] = filename_title
                _log.info(f"Using filename as title: {filename_title}")
            else:
                _log.warning(f"Could not extract title for {source_id}")
                return False

        # Generate markdown filename
        practice_name = practice_content["title"].lower().replace(" ", "-").replace("$", "dollar-")
        practice_name = re.sub(r"[^\w\-]", "", practice_name)  # Remove any non-alphanumeric/hyphen chars

        # Clean all content again to ensure no special characters remain
        for section_key, section_content in practice_content["sections"].items():
            practice_content["sections"][section_key] = clean_text(section_content)

        # Generate markdown content
        markdown_content = format_output(practice_content, "markdown")

        # Write markdown file
        output_path = output_dir / f"{practice_name}.md"
        with output_path.open("w", encoding="utf-8") as fp:
            fp.write(markdown_content)

        _log.info(f"Successfully processed {practice_content['title']} -> {output_path}")
        return True

    except Exception as e:
        _log.error(f"Error processing {source}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False
