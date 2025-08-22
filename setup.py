#!/usr/bin/env python3
"""
Setup script for Turkish Address Normalization Tool
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = (
    readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
)

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text(encoding="utf-8").strip().split("\n")
    requirements = [
        req.strip() for req in requirements if req.strip() and not req.startswith("#")
    ]

setup(
    name="addrnorm",
    version="2.0.0",
    description="Turkish Address Normalization Tool with ML and Rule-based Methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Meren Ertugrul",
    author_email="merenert@example.com",
    url="https://github.com/merenert/PreprocessingV2.0",
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    # Dependencies
    install_requires=requirements,
    # Entry points - CLI commands
    entry_points={
        "console_scripts": [
            "addrnorm=addrnorm.cli:main",
        ],
    },
    # Package data
    package_data={
        "addrnorm": [
            "data/*.json",
            "data/*.csv",
            "data/*.txt",
            "models/**/*",
        ],
    },
    include_package_data=True,
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="address normalization, turkish, nlp, named entity recognition",
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/merenert/PreprocessingV2.0/issues",
        "Source": "https://github.com/merenert/PreprocessingV2.0",
        "Documentation": "https://github.com/merenert/PreprocessingV2.0#readme",
    },
)
