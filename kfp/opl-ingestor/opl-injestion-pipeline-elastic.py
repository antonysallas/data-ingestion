import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

HTML_PARSER = "html.parser"

_log = logging.getLogger(__name__)


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


def extract_opl_content(html_content):
    """
    Extract the relevant practice content from Open Practice Library HTML.
    Returns a dict with title, subtitle, sections and their content.
    """
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


def format_output(practice_content, output_format="markdown"):
    """
    Format the practice content into the specified output format.
    Formats: 'markdown', 'text'
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


def get_all_practice_urls(homepage_url="https://openpracticelibrary.com/"):
    """
    Fetch all practice URLs from the Open Practice Library homepage.

    Args:
        homepage_url: URL of the OPL homepage

    Returns:
        list: List of practice URLs
    """
    _log.info(f"Fetching practice URLs from {homepage_url}")

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


def ingest_to_elasticsearch(input_artifact):
    """
    Function to ingest the processed content into Elasticsearch.

    This can be called directly or used as a KFP component.

    Args:
        input_artifact: Either a path to a JSON file or a list of document splits
    """
    import json
    import logging
    import os

    try:
        from elasticsearch import Elasticsearch
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        from langchain_elasticsearch import ElasticsearchStore
    except ImportError:
        logging.error(
            "Required libraries not found. Please install 'elasticsearch', 'langchain', and 'langchain-elasticsearch'"
        )
        return

    logging.basicConfig(level=logging.INFO)
    _log = logging.getLogger("ingest_to_elasticsearch")

    # Process the input artifact, which can be a file path or a list of document splits
    document_splits = []

    if isinstance(input_artifact, str) or hasattr(input_artifact, "path"):
        # If it's a path string or has a path attribute (like KFP artifact)
        path = input_artifact.path if hasattr(input_artifact, "path") else input_artifact
        try:
            with open(path) as input_file:
                splits_artifact = input_file.read()
                document_splits = json.loads(splits_artifact)
        except Exception as e:
            _log.error(f"Error reading input artifact: {str(e)}")
            return
    elif isinstance(input_artifact, list):
        # If it's already a list of document splits
        document_splits = input_artifact
    else:
        _log.error(f"Unsupported input_artifact type: {type(input_artifact)}")
        return

    # Get Elasticsearch credentials from environment variables
    es_user = os.environ.get("ES_USER")
    es_pass = os.environ.get("ES_PASS")
    es_host = os.environ.get("ES_HOST")

    if not es_user or not es_pass or not es_host:
        _log.error(
            "Elasticsearch config not present. Please set ES_USER, ES_PASS, and ES_HOST environment variables."
        )
        return

    # Initialize Elasticsearch client
    _log.info(f"Connecting to Elasticsearch at {es_host}")
    try:
        es_client = Elasticsearch(
            es_host, basic_auth=(es_user, es_pass), request_timeout=30, verify_certs=False
        )
    except Exception as e:
        _log.error(f"Failed to initialize Elasticsearch client: {str(e)}")
        return

    # Health check for Elasticsearch client connection
    try:
        health = es_client.cluster.health()
        _log.info(f"Elasticsearch cluster status: {health['status']}")
    except Exception as e:
        _log.error(f"Error connecting to Elasticsearch: {str(e)}")
        return

    def ingest(index_name, splits):
        """
        Ingest documents into Elasticsearch with embeddings

        Args:
            index_name: Name of the Elasticsearch index
            splits: List of document splits to ingest
        """
        # Initialize embedding model - using Nomic AI's embedding model
        model_name = "nomic-ai/nomic-embed-text-v1"

        # Set model parameters for CPU usage
        model_kwargs = {"trust_remote_code": True, "device": "cpu"}

        _log.info(f"Initializing embeddings model: {model_name}")

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                show_progress=True,
                encode_kwargs={"normalize_embeddings": True},
            )

            # Initialize Elasticsearch store with embeddings
            _log.info(f"Creating/updating index: {index_name}")
            db = ElasticsearchStore(
                index_name=index_name.lower(),  # index names in elastic must be lowercase
                embedding=embeddings,
                es_connection=es_client,
            )

            # Convert to Document objects
            _log.info(f"Converting {len(splits)} splits to Document objects")
            documents = [
                Document(page_content=split["page_content"], metadata=split["metadata"]) for split in splits
            ]

            # Add documents in batches to prevent overwhelming ES
            batch_size = 50
            total_batches = (len(documents) + batch_size - 1) // batch_size

            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                _log.info(
                    f"Uploading batch {batch_num}/{total_batches} ({len(batch)} documents) to index {index_name}"
                )
                db.add_documents(batch)

            _log.info(f"Successfully uploaded all documents to index {index_name}")

        except Exception as e:
            _log.error(f"Error ingesting documents to Elasticsearch: {str(e)}")
            import traceback

            traceback.print_exc()

    # Process each index and its documents
    if isinstance(document_splits, list):
        # If it's a list of dictionaries with index_name and splits
        for index_data in document_splits:
            if isinstance(index_data, dict) and "index_name" in index_data and "splits" in index_data:
                index_name = index_data["index_name"]
                splits = index_data["splits"]
                _log.info(f"Processing index {index_name} with {len(splits)} documents")
                ingest(index_name=index_name, splits=splits)
    elif isinstance(document_splits, dict):
        # If it's a single dictionary
        for index_name, splits in document_splits.items():
            _log.info(f"Processing index {index_name} with {len(splits)} documents")
            ingest(index_name=index_name, splits=splits)

    _log.info("Document ingestion complete!")


# We don't need the process_html_file function anymore since we're only processing from URLs


def prepare_documents_for_es(output_dir):
    """
    Prepare processed markdown documents for Elasticsearch ingestion.

    Args:
        output_dir: Directory containing the markdown files

    Returns:
        dict: Dictionary mapping index name to list of document splits
    """
    _log.info(f"Preparing documents from {output_dir} for Elasticsearch ingestion")

    # Check if output directory exists and contains files
    if not output_dir.exists() or not output_dir.is_dir():
        _log.error(f"Output directory '{output_dir}' not found")
        return None

    # Find all markdown files
    md_files = list(output_dir.glob("*.md"))

    if not md_files:
        _log.error(f"No markdown files found in '{output_dir}'")
        return None

    _log.info(f"Found {len(md_files)} markdown files to ingest")

    # Create index name for Open Practice Library
    index_name = "open_practice_library_en_us_2024".lower()

    # Initialize documents list
    documents = []

    # Process each markdown file
    for md_file in md_files:
        try:
            # Read markdown content
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract title from first line (assumes markdown starts with # Title)
            title_match = re.match(r"^# (.+)$", content.split("\n")[0])
            title = title_match.group(1) if title_match else md_file.stem

            # Create metadata
            metadata = {
                "source": "open_practice_library",
                "source_full_name": "Open Practice Library",
                "title": title,
                "filename": md_file.name,
                "version": "2024",
                "language": "en-US",
                "url": f"https://openpracticelibrary.com/practice/{md_file.stem}",
            }

            # Create document
            document = {"page_content": content, "metadata": metadata}

            documents.append(document)

        except Exception as e:
            _log.error(f"Error processing {md_file}: {str(e)}")

    # Return document splits for Elasticsearch ingestion
    return {index_name: documents}


def main():
    """
    Main function for the Open Practice Library ingestion script.

    This script fetches practice content from the Open Practice Library website,
    processes it into markdown, and ingests it into Elasticsearch.

    Environment variables:
      - ES_USER: Elasticsearch username
      - ES_PASS: Elasticsearch password
      - ES_HOST: Elasticsearch host URL
      - KUBEFLOW_ENDPOINT: Kubeflow pipeline endpoint (optional, for pipeline usage)
      - BEARER_TOKEN: Authentication token for Kubeflow (optional, for pipeline usage)

    Usage in OpenShift Pipeline:
      - All content is fetched from the OPL website (https://openpracticelibrary.com/)
      - Processed markdown files are saved to the "practices" directory
      - Content is ingested into Elasticsearch using the provided credentials
    """
    import os
    import sys
    import traceback

    # Configure logging with timestamp and log level
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Configuration based on redhat-product-documentation-ingestor
    # -----------------------------------------------------------------
    # Base URL for the Open Practice Library
    opl_base_url = "https://openpracticelibrary.com/"

    # Directory for output files
    output_dir = Path("practices")

    # Limit the number of practices to process for testing
    # Set to None for production use
    max_practices = None

    # Uncomment for testing with a limited number of practices
    # max_practices = 10

    # Delay between requests to avoid overloading the server
    request_delay = 1.0

    # In standalone mode, we always attempt to ingest to Elasticsearch
    # using the same environment variables used by RedHat product documentation ingestor
    ingest_to_es = True
    # -----------------------------------------------------------------

    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        _log.info(f"Output directory: {output_dir.absolute()}")

        # Process practices from the website
        _log.info(f"Starting to process practices from {opl_base_url}")

        # Get all practice URLs
        practice_urls = get_all_practice_urls(opl_base_url)

        if not practice_urls:
            _log.error("No practice URLs found from website. Exiting.")
            sys.exit(1)

        # Limit the number of practices to process if MAX_PRACTICES is set
        if max_practices > 0:
            _log.info(f"Limiting to {max_practices} practices for processing")
            practice_urls = practice_urls[:max_practices]

        _log.info(f"Found {len(practice_urls)} practices to process")

        # Process each practice
        successful = 0
        failed = 0

        for i, url in enumerate(practice_urls):
            _log.info(f"Processing practice {i+1}/{len(practice_urls)}: {url}")

            if process_practice(url, output_dir):
                successful += 1
            else:
                failed += 1

            # Add a delay to avoid overloading the server
            if i < len(practice_urls) - 1:  # Don't delay after the last request
                time.sleep(request_delay)

        _log.info(
            f"Website processing complete. Successfully processed {successful} practices. Failed: {failed}"
        )
        _log.info(f"Markdown files saved to {output_dir.absolute()}")

        # Ingest documents into Elasticsearch
        if ingest_to_es:
            _log.info("Preparing to ingest documents into Elasticsearch")

            # Get Elasticsearch credentials from environment variables
            es_user = os.environ.get("ES_USER")
            es_pass = os.environ.get("ES_PASS")
            es_host = os.environ.get("ES_HOST")

            if not es_user or not es_pass or not es_host:
                _log.error(
                    "Elasticsearch config not present. Check ES_USER, ES_PASS, and ES_HOST environment variables."
                )
                _log.error("Exiting without ingesting to Elasticsearch.")
                sys.exit(1)

            try:
                # Prepare documents for Elasticsearch
                document_splits = prepare_documents_for_es(output_dir)
                if document_splits:
                    # Ingest into Elasticsearch
                    _log.info("Ingesting processed documents into Elasticsearch...")
                    ingest_to_elasticsearch(document_splits)
                    _log.info("Successfully ingested documents into Elasticsearch.")
                else:
                    _log.error("No documents prepared for ingestion. Skipping Elasticsearch ingestion.")
            except Exception as e:
                _log.error(f"Error during Elasticsearch ingestion: {str(e)}")
                _log.debug(traceback.format_exc())
                sys.exit(1)

        # Return success/failure count for pipeline
        return successful, failed

    except Exception as e:
        _log.error(f"Error in main execution: {str(e)}")
        _log.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
