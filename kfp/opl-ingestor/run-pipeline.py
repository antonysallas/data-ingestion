#!/usr/bin/env python3
# ruff: noqa: N999
"""
Script to run the OPL ingestion pipeline directly in OpenShift.

This script directly uses the pipeline definition from the opl module
to ensure proper execution in the OpenShift environment.
"""

import os
import sys
import time

# Third-party imports
import kfp

# Local imports
from .kubeflow_components import ingestion_pipeline  # ruff: noqa: E402

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def main():
    """Run the OPL ingestion pipeline."""
    # Get Kubeflow endpoint
    kubeflow_endpoint = os.environ.get("KUBEFLOW_ENDPOINT")
    if not kubeflow_endpoint:
        print("KUBEFLOW_ENDPOINT environment variable not set.")
        return 1

    print(f"Connecting to Kubeflow at: {kubeflow_endpoint}")

    # Get authentication token
    # ruff: noqa: S105
    sa_token_path = "/run/secrets/kubernetes.io/serviceaccount/token"
    if os.path.isfile(sa_token_path):
        with open(sa_token_path) as f:
            bearer_token = f.read().rstrip()
    else:
        bearer_token = os.environ.get("BEARER_TOKEN")
        if not bearer_token:
            print("BEARER_TOKEN environment variable not set and service account token not found.")
            return 1

    # Create KFP client
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=None,  # Using None directly instead of variable
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
    print(f"You can view this run in the Kubeflow Pipelines UI at: {kubeflow_endpoint}")

    return 0


if __name__ == "__main__":
    import time

    sys.exit(main())
