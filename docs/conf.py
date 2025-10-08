import os
import sys

# Add project root to sys.path so Sphinx can import the package
sys.path.insert(0, os.path.abspath(".."))

project = "BEVFusion"
author = "BEVFusion Contributors"
copyright = "2025, BEVFusion Contributors"

# General configuration
extensions = [
	"sphinx.ext.autodoc",
	"sphinx.ext.autosummary",
	"sphinx.ext.napoleon",
	"sphinx.ext.viewcode",
	"myst_parser",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Mock heavy optional dependencies to allow API docs to build without full env
autodoc_mock_imports = [
	"torch",
	"mmcv",
	"mmdet",
	"torchpack",
	"spconv",
	"torchsparse",
	"PIL",
	"tqdm",
	"nuscenes",
	"cv2",
]

templates_path = ["_templates"]
exclude_patterns = [
	"_build",
	"Thumbs.db",
	".DS_Store",
]

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

