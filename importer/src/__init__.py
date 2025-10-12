"""Primary source package for the knowledge-builder importer.

This makes the 'src' directory a proper Python package so that test imports
like 'importer.src.processing.agent_orchestrator' succeed. All intra-package
imports should use relative form (e.g. 'from ..config import Config').
"""

__all__ = []
