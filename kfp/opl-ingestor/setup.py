"""
Setup script for the opl_ingestor package.
"""

from setuptools import find_packages, setup

setup(
    name="opl_ingestor",
    version="0.1.0",
    description="Open Practice Library content ingestion pipeline",
    author="Antony Sallas",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "beautifulsoup4>=4.12.2",
        "requests>=2.28.0",
        "langchain>=0.3.8",
        "langchain-community>=0.3.8",
        "langchain-elasticsearch>=0.3.0",
        "elasticsearch>=8.16.0",
        "kfp>=2.0.0",
    ],
)