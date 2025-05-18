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
        "requests==2.31.0",
    ],
)
def extract_practice_content(url: str) -> dict:
    """Extract content from a practice URL."""
    import logging

    import requests
    from bs4 import BeautifulSoup

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("extract_practice_content")

    def convert_html_to_markdown(html_content):
        """Convert HTML to Markdown format."""
        soup = BeautifulSoup(html_content, HTML_PARSER)
        markdown_output = []

        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(h.name[1])
            markdown_output.extend(["#" * level + " " + h.get_text().strip(), ""])

        for p in soup.find_all("p"):
            markdown_output.extend([p.get_text().strip(), ""])

        for ul in soup.find_all("ul"):
            for li in ul.find_all("li"):
                markdown_output.extend(["* " + li.get_text().strip(), ""])

        return "\n".join(markdown_output)

    try:
        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.text, HTML_PARSER)
        practice_content = {
            "url": url,
            "title": "Unknown Title",
            "subtitle": "A Practice from the Open Practice Library",
            "sections": {},
        }

        title_element = soup.select_one("h1")
        if title_element:
            practice_content["title"] = title_element.get_text().strip()

        subtitle = soup.select_one("meta[name='description']")
        if subtitle and subtitle.get("content"):
            practice_content["subtitle"] = subtitle.get("content").strip()

        for section_id, section_name in [
            ("what_is", "What Is"),
            ("why_use", "Why Do"),
            ("how_to", "How to do"),
            ("further_info", "Further Information"),
            ("related", "Related Practices"),
        ]:
            section = soup.select_one(f"div[data-testid='{section_id}']")
            if section:
                practice_content["sections"][section_name] = convert_html_to_markdown(str(section))

        logger.info(f"Successfully extracted content from {url}")
        return practice_content
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {str(e)}")
        raise


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[],
)
def format_practice_markdown(practice_content: dict) -> dict:
    """Format practice content into markdown."""
    import logging
    import re

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("format_practice_markdown")

    def format_section(heading, content, title):
        """Format a section with proper heading."""
        if "What Is" in heading:
            clean_heading = f"### What Is {title}"
        elif "Why Do" in heading:
            clean_heading = f"### Why Do {title}"
        elif "How to do" in heading:
            clean_heading = f"### How to do {title}"
        else:
            clean_heading = f"### {heading}"
        return f"{clean_heading}\n\n{content}\n"

    try:
        output = [
            f"# {practice_content['title']}",
            f"## {practice_content['subtitle']}",
            "",
        ]

        for heading, content in practice_content["sections"].items():
            output.append(format_section(heading, content, practice_content["title"]))

        markdown_content = "\n".join(output)
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        if not markdown_content.endswith("\n"):
            markdown_content += "\n"

        practice_name = practice_content["title"].lower().replace(" ", "-").replace("$", "dollar-")
        practice_name = re.sub(r"[^\w\-]", "", practice_name)

        result = {
            "filename": f"{practice_name}.md",
            "content": markdown_content,
            "metadata": {
                "source": "open_practice_library",
                "source_full_name": "Open Practice Library",
                "title": practice_content["title"],
                "version": "2024",
                "language": "en-US",
                "url": practice_content["url"],
            },
        }

        logger.info(f"Successfully formatted markdown for {practice_content['title']}")
        return result
    except Exception as e:
        logger.error(f"Error formatting markdown: {str(e)}")
        raise


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[],
)
def prepare_elasticsearch_documents(practices: list, splits_artifact: Output[Artifact]):
    """Prepare documents for Elasticsearch ingestion."""
    import json
    import logging
    import tempfile
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("prepare_elasticsearch_documents")

    try:
        # Create temporary directory for markdown files
        temp_dir = Path(tempfile.mkdtemp()) / "practices"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Write markdown files
        for practice in practices:
            output_path = temp_dir / practice["filename"]
            with output_path.open("w", encoding="utf-8") as f:
                f.write(practice["content"])

        # Prepare document splits
        index_name = "opl_practices_en_us".lower()
        documents = [{"page_content": practice["content"], "metadata": practice["metadata"]} for practice in practices]

        document_splits = [{"index_name": index_name, "splits": documents}]

        # Write to output artifact
        with open(splits_artifact.path, "w") as f:
            f.write(json.dumps(document_splits))

        logger.info(f"Successfully prepared {len(practices)} documents for Elasticsearch")
    except Exception as e:
        logger.error(f"Error preparing documents: {str(e)}")
        raise


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
    """Kubeflow Pipeline for OPL document ingestion."""
    # Step 1: Load documents from OPL website
    load_docs_task = load_documents()

    # Step 2: Extract content from each practice
    extract_tasks = []
    for url in load_docs_task.output:
        extract_task = extract_practice_content(url=url)
        extract_tasks.append(extract_task)

    # Step 3: Format each practice into markdown
    format_tasks = []
    for extract_task in extract_tasks:
        format_task = format_practice_markdown(practice_content=extract_task.output)
        format_tasks.append(format_task)

    # Step 4: Prepare documents for Elasticsearch
    prepare_docs_task = prepare_elasticsearch_documents(practices=[task.output for task in format_tasks])

    # Step 5: Ingest documents into Elasticsearch
    ingest_docs_task = ingest_documents(input_artifact=prepare_docs_task.outputs["splits_artifact"])
    ingest_docs_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

    # Set environment variables for Elasticsearch
    kubernetes.use_secret_as_env(
        ingest_docs_task,
        secret_name="elasticsearch-es-elastic-user",
        secret_key_to_env={"elastic": "ES_PASS"},
    )
    ingest_docs_task.set_env_variable("ES_HOST", "http://elasticsearch-es-http:9200")
    ingest_docs_task.set_env_variable("ES_USER", "elastic")

    # Add GPU tolerations
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
