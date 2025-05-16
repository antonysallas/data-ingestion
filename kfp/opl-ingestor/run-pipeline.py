#!/usr/bin/env python3
"""
Script to run the OPL ingestion pipeline directly in OpenShift.

This script directly uses the pipeline definition from the opl module
to ensure proper execution in the OpenShift environment.
"""

import os
import sys
import time
from pathlib import Path

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the pipeline definition
from kubeflow_components import ingestion_pipeline

# Import kfp modules
import kfp


def main():
    """Run the OPL ingestion pipeline."""
    # Get Kubeflow endpoint
    KUBEFLOW_ENDPOINT = os.environ.get("KUBEFLOW_ENDPOINT")
    if not KUBEFLOW_ENDPOINT:
        print("KUBEFLOW_ENDPOINT environment variable not set.")
        return 1

    print(f"Connecting to Kubeflow at: {KUBEFLOW_ENDPOINT}")

    # Get authentication token
    sa_token_path = "/run/secrets/kubernetes.io/serviceaccount/token"
    if os.path.isfile(sa_token_path):
        with open(sa_token_path) as f:
            BEARER_TOKEN = f.read().rstrip()
    else:
        BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
        if not BEARER_TOKEN:
            print("BEARER_TOKEN environment variable not set and service account token not found.")
            return 1

    # Get service account certificate
    sa_ca_cert = "/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"
    ssl_ca_cert = sa_ca_cert if os.path.isfile(sa_ca_cert) else None

    # Create KFP client
    client = kfp.Client(
        host=KUBEFLOW_ENDPOINT,
        existing_token=BEARER_TOKEN,
        ssl_ca_cert=None,  # Using None here as we don't always need certificate verification
    )

    # Create and run the pipeline
    result = client.create_run_from_pipeline_func(
        ingestion_pipeline,
        experiment_name="opl_document_ingestion",
        # Use a unique run name with timestamp
        run_name=f"opl-ingestion-{int(time.time())}",
        # Disable caching to ensure a fresh run
        enable_caching=False,
    )

    print(f"Pipeline run created: {result.run_id}")
    print(f"You can view this run in the Kubeflow Pipelines UI at: {KUBEFLOW_ENDPOINT}")

    return 0


if __name__ == "__main__":
    import time

    sys.exit(main())
