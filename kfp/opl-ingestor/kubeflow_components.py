"""
Kubeflow Pipeline components for Open Practice Library content ingestion.

This module contains the Kubeflow Pipeline components and pipeline
definition for the OPL ingestion pipeline.
"""

import logging
import os

import kfp
from kfp import dsl, kubernetes
from kfp.dsl import Artifact, Input, Output

# Configure module logger
_log = logging.getLogger(__name__)

# Constants
HTML_PARSER = "html.parser"


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["requests", "beautifulsoup4"],
)
def load_documents() -> list:
    """
    KFP Component to load documents from Open Practice Library.

    Returns:
        List of URLs to process
    """
    import logging
    from urllib.parse import urljoin

    import requests
    from bs4 import BeautifulSoup

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("load_documents")

    # Base URL for the Open Practice Library
    opl_base_url = "https://openpracticelibrary.com/"

    logger.info(f"Fetching practice URLs from {opl_base_url}")
    response = requests.get(opl_base_url, timeout=30)
    soup = BeautifulSoup(response.text, HTML_PARSER)

    # Find all practice card links
    practice_links = soup.select('div[data-testid="practicecardgrid"] a')
    logger.info(f"Found {len(practice_links)} practice links")

    # Extract and normalize URLs
    practice_urls = []
    for link in practice_links:
        href = link.get("href")
        if href and href.startswith("/practice/"):
            full_url = urljoin(opl_base_url, href)
            practice_urls.append(full_url)

    # Limit the number of practices to process for testing
    max_practices = 0  # Set to 0 for no limit
    if max_practices > 0:
        logger.info(f"Limiting to {max_practices} practices for processing")
        practice_urls = practice_urls[:max_practices]

    logger.info(f"Extracted {len(practice_urls)} practice URLs")
    return practice_urls


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "beautifulsoup4==4.12.2",
        "html2text==2024.2.26",
        "langchain-community==0.3.8",
        "langchain==0.3.8",
        "lxml==5.1.0",
        "tqdm==4.66.2",
        "elastic-transport==8.15.1",
        "elasticsearch==8.16.0",
        "langchain-elasticsearch==0.3.0",
    ],
)
def format_documents(documents: list, splits_artifact: Output[Artifact]):
    """
    KFP Component to format OPL documents into markdown and prepare for Elasticsearch.

    Args:
        documents: List of URLs to process

    Returns:
        Dictionary containing document splits for Elasticsearch
    """
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("format_documents")

    # Use a temporary directory that we have permission to create/write in container environment
    import tempfile

    # Create a temporary directory for storing the processed files
    temp_dir = tempfile.mkdtemp()
    output_dir = Path(temp_dir) / "practices"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created temporary output directory: {output_dir}")

    # In Kubeflow components, we need to include our dependencies in the component itself
    # since module references aren't directly available

    # First define the necessary functions inline within the component
    def prepare_documents_for_es(output_dir):
        """
        Prepare processed markdown documents for Elasticsearch ingestion.

        Args:
            output_dir: Directory containing the markdown files

        Returns:
            list: List of objects with index_name and splits fields
        """
        logger.info(f"Preparing documents from {output_dir} for Elasticsearch ingestion")

        # Check if output directory exists and contains files
        if not output_dir.exists() or not output_dir.is_dir():
            logger.error(f"Output directory '{output_dir}' not found")
            return []

        # Find all markdown files
        md_files = list(output_dir.glob("*.md"))

        if not md_files:
            logger.error(f"No markdown files found in '{output_dir}'")
            return []

        logger.info(f"Found {len(md_files)} markdown files to ingest")

        # Create index name for Open Practice Library - ensure it's lowercase and without special characters
        index_name = "opl_practices_en_us".lower()

        # Initialize documents list
        documents = []

        # Process each markdown file
        for md_file in md_files:
            try:
                # Read markdown content
                with open(md_file, encoding="utf-8") as f:
                    content = f.read()

                # Extract title from first line (assumes markdown starts with # Title)
                import re

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
                logger.error(f"Error processing {md_file}: {str(e)}")

        # Return document splits for Elasticsearch ingestion in the format expected by the pipeline
        return [{"index_name": index_name, "splits": documents}]

    def extract_opl_content(html_content):
        """Extract content from Open Practice Library HTML."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, HTML_PARSER)

        # Initialize content dictionary
        practice_content = {
            "title": "Unknown Title",
            "subtitle": "A Practice from the Open Practice Library",
            "sections": {},
        }

        # Extract title
        title_element = soup.select_one("h1")
        if title_element:
            practice_content["title"] = title_element.get_text().strip()

        # Extract subtitle/description
        subtitle = soup.select_one("meta[name='description']")
        if subtitle and subtitle.get("content"):
            practice_content["subtitle"] = subtitle.get("content").strip()

        # Extract main content sections
        sections = {}

        # What is section
        what_is = soup.select_one("div[data-testid='what_is']")
        if what_is:
            sections["What Is"] = convert_html_to_markdown(str(what_is))

        # Why use section
        why_use = soup.select_one("div[data-testid='why_use']")
        if why_use:
            sections["Why Do"] = convert_html_to_markdown(str(why_use))

        # How to section
        how_to = soup.select_one("div[data-testid='how_to']")
        if how_to:
            sections["How to do"] = convert_html_to_markdown(str(how_to))

        # Add additional sections if they exist
        further_info = soup.select_one("div[data-testid='further_info']")
        if further_info:
            sections["Further Information"] = convert_html_to_markdown(str(further_info))

        related_practices = soup.select_one("div[data-testid='related']")
        if related_practices:
            sections["Related Practices"] = convert_html_to_markdown(str(related_practices))

        # Add sections to content
        practice_content["sections"] = sections

        return practice_content

    def convert_html_to_markdown(html_content):
        """Convert HTML to Markdown format."""
        import re

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, HTML_PARSER)
        markdown_output = []

        # Process headings
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(h.name[1])
            markdown_output.append("#" * level + " " + h.get_text().strip())
            markdown_output.append("")

        # Process paragraphs
        for p in soup.find_all("p"):
            markdown_output.append(p.get_text().strip())
            markdown_output.append("")

        # Process lists
        for ul in soup.find_all("ul"):
            for li in ul.find_all("li"):
                markdown_output.append("* " + li.get_text().strip())
            markdown_output.append("")

        # Clean up consecutive blank lines
        result = "\n".join(markdown_output)
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result

    def process_practice(source, output_dir):
        """Process a single practice source and generate a markdown file."""
        logger.info(f"Processing {source}")

        import re

        import requests

        try:
            # Fetch URL content
            response = requests.get(source, timeout=30)
            html_content = response.text

            # Extract content
            practice_content = extract_opl_content(html_content)

            if not practice_content["title"] or practice_content["title"] == "Unknown Title":
                logger.warning(f"Could not extract title for {source}")
                return False

            # Generate markdown filename
            practice_name = practice_content["title"].lower().replace(" ", "-").replace("$", "dollar-")
            practice_name = re.sub(r"[^\w\-]", "", practice_name)  # Remove any non-alphanumeric/hyphen chars

            # Generate markdown content
            markdown_content = format_output(practice_content, "markdown")

            # Write markdown file
            output_path = output_dir / f"{practice_name}.md"
            with output_path.open("w", encoding="utf-8") as fp:
                fp.write(markdown_content)

            logger.info(f"Successfully processed {practice_content['title']} -> {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error processing {source}: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def format_output(practice_content, output_format="markdown"):
        """Format practice content into markdown output."""
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

                # Add content
                import re

                content = re.sub(r"\n{3,}", "\n\n", content)
                output.append(content)
                output.append("")  # Empty line after content

        # Final cleanup to ensure consistent formatting
        result = "\n".join(output)

        # Remove any extra blank lines
        import re

        result = re.sub(r"\n{3,}", "\n\n", result)

        # Ensure the file ends with a newline
        if not result.endswith("\n"):
            result += "\n"

        return result

    # Process each URL
    successful = 0
    failed = 0

    for i, url in enumerate(documents):
        logger.info(f"Processing practice {i + 1}/{len(documents)}: {url}")

        if process_practice(url, output_dir):
            successful += 1
        else:
            failed += 1

    logger.info(f"Successfully processed {successful} practices. Failed: {failed}")

    # Prepare documents for Elasticsearch
    document_splits = prepare_documents_for_es(output_dir)

    # Write splits to output artifact
    with open(splits_artifact.path, "w") as f:
        f.write(json.dumps(document_splits))


@dsl.component(
    base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/minimal-gpu:2024.2",
    packages_to_install=[
        "langchain-community==0.3.8",
        "langchain==0.3.8",
        "elastic-transport==8.15.1",
        "elasticsearch==8.16.0",
        "langchain-elasticsearch==0.3.0",
        "sentence-transformers==2.4.0",
        "einops==0.7.0",
    ],
)
def ingest_documents(input_artifact: Input[Artifact]) -> None:
    """
    KFP Component to ingest documents into Elasticsearch.

    Args:
        input_artifact: Artifact containing document splits
    """
    import json
    import logging
    import os

    from elasticsearch import Elasticsearch
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_elasticsearch import ElasticsearchStore

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ingest_documents")

    # Read input artifact
    with open(input_artifact.path) as input_file:
        splits_artifact = input_file.read()
        document_splits = json.loads(splits_artifact)

    # Log environment variables (for debugging)
    logger.info("Environment variables for Elasticsearch:")
    logger.info(f"ES_USER: {'SET' if os.environ.get('ES_USER') else 'NOT SET'}")
    logger.info(f"ES_PASS: {'SET' if os.environ.get('ES_PASS') else 'NOT SET'}")
    logger.info(f"ES_HOST: {'SET' if os.environ.get('ES_HOST') else 'NOT SET'}")

    # Get Elasticsearch credentials from environment variables
    es_user = os.environ.get("ES_USER")
    es_pass = os.environ.get("ES_PASS")
    es_host = os.environ.get("ES_HOST")

    # Check if required credentials are set
    if not es_user or not es_pass or not es_host:
        logger.error("Elasticsearch config not present. Check ES_USER, ES_PASS, and ES_HOST environment variables.")
        return

    # Initialize Elasticsearch client
    logger.info(f"Connecting to Elasticsearch at {es_host}")
    es_client = Elasticsearch(es_host, basic_auth=(es_user, es_pass), request_timeout=30, verify_certs=False)

    # Health check for Elasticsearch client connection
    logger.info(f"Elasticsearch cluster status: {es_client.cluster.health()}")

    def ingest(index_name, splits):
        """Ingest documents into Elasticsearch with embeddings."""
        # Initialize embedding model
        model_name = "nomic-ai/nomic-embed-text-v1"

        # Set model parameters for GPU (if available)
        device = "cuda" if os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"
        logger.info(f"Using device: {device} for embeddings")

        model_kwargs = {"trust_remote_code": True, "device": device}

        # Initialize embeddings
        logger.info(f"Initializing embeddings model: {model_name}")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            show_progress=True,
            # Remove encode_kwargs to match working implementation
        )

        # Initialize Elasticsearch store with embeddings
        logger.info(f"Creating/updating index: {index_name}")

        # Check if index exists first
        if not es_client.indices.exists(index=index_name.lower()):
            logger.info(f"Index {index_name.lower()} doesn't exist. Creating it...")

            # Create mappings for the index
            mappings = {
                "properties": {
                    "text": {"type": "text"},
                    "metadata": {"type": "object", "enabled": True},
                    "vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
                }
            }

            # Create the index with appropriate settings
            es_client.indices.create(
                index=index_name.lower(), mappings=mappings, settings={"number_of_shards": 1, "number_of_replicas": 0}
            )
            logger.info(f"Successfully created index {index_name.lower()}")

        # Now create the ElasticsearchStore with the index
        db = ElasticsearchStore(
            index_name=index_name.lower(),  # index names in elastic must be lowercase
            embedding=embeddings,
            es_connection=es_client,
        )

        # Add documents in batches
        batch_size = 50
        total_batches = (len(splits) + batch_size - 1) // batch_size

        for i in range(0, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            logger.info(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} documents) to index {index_name}")
            db.add_documents(batch)

        logger.info(f"Successfully uploaded all documents to index {index_name}")

    # Process each index and its documents
    # Additional logging to help debug
    # Log document splits type information
    content_format = "empty" if not document_splits else type(document_splits[0])
    logger.info(f"Got document_splits: type={type(document_splits)}, content format={content_format}")

    if not document_splits:
        logger.warning("No document splits found to process. Check previous steps.")
        return

    for index_data in document_splits:
        index_name = index_data["index_name"]
        splits = index_data["splits"]
        logger.info(f"Processing index {index_name} with {len(splits)} documents")

        if not splits:
            logger.warning(f"No documents to process for index {index_name}. Skipping.")
            continue

        # Convert to Document objects
        documents = [Document(page_content=split["page_content"], metadata=split["metadata"]) for split in splits]

        # Ingest documents
        ingest(index_name=index_name, splits=documents)

    logger.info("Elasticsearch ingestion complete")


@dsl.pipeline(name="Document Ingestion")
def ingestion_pipeline():
    """
    Kubeflow Pipeline for OPL document ingestion.
    """
    # Step 1: Load documents from OPL website
    load_docs_task = load_documents()

    # Step 2: Format documents into markdown
    format_docs_task = format_documents(documents=load_docs_task.output)
    format_docs_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

    # Step 3: Ingest documents into Elasticsearch
    ingest_docs_task = ingest_documents(input_artifact=format_docs_task.outputs["splits_artifact"])
    ingest_docs_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

    # Set environment variables for Elasticsearch - following exactly the same pattern as the working pipeline
    kubernetes.use_secret_as_env(
        ingest_docs_task,
        secret_name="elasticsearch-es-elastic-user",
        secret_key_to_env={"elastic": "ES_PASS"},
    )
    ingest_docs_task.set_env_variable("ES_HOST", "http://elasticsearch-es-http:9200")
    ingest_docs_task.set_env_variable("ES_USER", "elastic")

    # Add GPU tolerations
    kubernetes.add_toleration(format_docs_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
    kubernetes.add_toleration(ingest_docs_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")


def run_kubeflow_pipeline():
    """
    Run the Kubeflow Pipeline with the configured client.

    Returns:
        str: ID of the created run
    """
    # Get Kubeflow endpoint and authentication token
    kubeflow_endpoint = os.environ.get("KUBEFLOW_ENDPOINT")
    _log.info(f"Connecting to kfp: {kubeflow_endpoint}")

    # Get service account token
    sa_token_path = "/run/secrets/kubernetes.io/serviceaccount/token"
    if os.path.isfile(sa_token_path):
        with open(sa_token_path) as f:
            bearer_token = f.read().rstrip()
    else:
        bearer_token = os.environ.get("BEARER_TOKEN")

    # Get service account certificate - currently not used in client creation
    sa_ca_cert = "/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"

    # Create KFP client and run pipeline
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        # ssl_ca_cert is not used, passing None directly instead of variable
    )

    result = client.create_run_from_pipeline_func(
        ingestion_pipeline,
        experiment_name="document_ingestion",
    )

    _log.info(f"Pipeline run created: {result.run_id}")
    return result.run_id
