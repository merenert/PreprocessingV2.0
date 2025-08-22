"""
Explanation parsing module for Turkish address explanations.

This module provides functionality to parse Turkish explanations like:
- "Amorium Hotel karşısı" → {"type": "landmark", "name": "amorium hotel",
                              "relation": "karşısı"}
- "marketin yanı" → {"type": "landmark", "name": "market", "relation": "yanı"}
"""

from .rules import ExplanationResult, parse_explanation

__all__ = ["parse_explanation", "ExplanationResult"]
