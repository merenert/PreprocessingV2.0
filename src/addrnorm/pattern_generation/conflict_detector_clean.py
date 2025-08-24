"""
Simplified Pattern Conflict Detector

Simplified version for demo purposes.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

from .models import PatternSuggestion, PatternConflict, ConflictType, ConflictResolution, ConflictSeverity, ResolutionStrategy

logger = logging.getLogger(__name__)


class PatternConflictDetector:
    """Simplified pattern conflict detector"""

    def __init__(self):
        pass

    def detect_conflicts(
        self, existing_patterns: List[Dict[str, Any]], new_patterns: List[PatternSuggestion]
    ) -> List[PatternConflict]:
        """Detect conflicts between existing and new patterns"""

        conflicts = []

        try:
            for new_pattern in new_patterns:
                for existing in existing_patterns:
                    conflict = self._check_regex_overlap(existing, new_pattern)
                    if conflict:
                        conflicts.append(conflict)

        except Exception as e:
            logger.error(f"Error in conflict detection: {e}")

        return conflicts

    def _check_regex_overlap(self, existing: Dict[str, Any], new: PatternSuggestion) -> Optional[PatternConflict]:
        """Check for regex pattern overlap"""

        try:
            existing_pattern = existing.get("pattern", "")
            new_pattern = new.regex_pattern

            # Simple overlap detection - check if patterns are similar
            if self._patterns_overlap(existing_pattern, new_pattern):
                return PatternConflict(
                    conflict_id=f"overlap_{new.pattern_id}_{existing.get('id', 'unknown')}",
                    pattern1_id=existing.get("id", "unknown"),
                    pattern2_id=new.pattern_id,
                    conflict_type=ConflictType.REGEX_OVERLAP,
                    severity=ConflictSeverity.MEDIUM,
                    description=f"Pattern overlap detected between {existing.get('id')} and {new.pattern_id}",
                    detection_details={"overlap_type": "regex"},
                    suggested_resolutions=self._get_simple_resolutions(),
                )

        except Exception as e:
            logger.error(f"Error checking regex overlap: {e}")

        return None

    def _patterns_overlap(self, pattern1: str, pattern2: str) -> bool:
        """Simple pattern overlap check"""

        # Basic heuristic - check for common keywords
        keywords1 = set(re.findall(r"\w+", pattern1.lower()))
        keywords2 = set(re.findall(r"\w+", pattern2.lower()))

        common = keywords1.intersection(keywords2)
        if len(common) > 0:
            return True

        return False

    def _get_simple_resolutions(self) -> List[ConflictResolution]:
        """Get simple resolution suggestions"""

        return [
            ConflictResolution(
                strategy=ResolutionStrategy.MERGE_PATTERNS,
                confidence=0.7,
                explanation="Consider merging similar patterns",
                auto_applied=False,
            ),
            ConflictResolution(
                strategy=ResolutionStrategy.ADD_SPECIFICITY,
                confidence=0.6,
                explanation="Add more specificity to distinguish patterns",
                auto_applied=False,
            ),
        ]
