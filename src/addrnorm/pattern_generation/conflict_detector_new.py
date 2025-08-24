"""
Pattern Conflict Detector

Detects and analyzes conflicts between existing and suggested patterns.
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict, Counter
from datetime import datetime
import hashlib

from .models import PatternSuggestion, PatternConflict, ConflictType, ConflictResolution, ConflictSeverity, ResolutionStrategy

logger = logging.getLogger(__name__)


class PatternConflictDetector:
    """
    Advanced pattern conflict detection and resolution system
    """

    def __init__(self):
        self.conflict_cache = {}
        self.resolution_history = defaultdict(list)

    def detect_conflicts(
        self, existing_patterns: List[Dict[str, Any]], new_patterns: List[PatternSuggestion]
    ) -> List[PatternConflict]:
        """
        Detect conflicts between existing and new patterns

        Args:
            existing_patterns: Current patterns in the system
            new_patterns: Newly suggested patterns

        Returns:
            List of detected conflicts
        """

        logger.info(f"Analyzing conflicts between {len(existing_patterns)} existing and {len(new_patterns)} new patterns")

        conflicts = []

        for new_pattern in new_patterns:
            for existing_pattern in existing_patterns:
                conflict = self._analyze_pattern_pair(existing_pattern, new_pattern)
                if conflict:
                    conflicts.append(conflict)

        # Cross-analyze new patterns for internal conflicts
        internal_conflicts = self._detect_internal_conflicts(new_patterns)
        conflicts.extend(internal_conflicts)

        # Sort by severity
        conflicts.sort(key=lambda x: x.severity.value, reverse=True)

        logger.info(f"Detected {len(conflicts)} conflicts")

        return conflicts

    def _analyze_pattern_pair(self, existing: Dict[str, Any], new: PatternSuggestion) -> Optional[PatternConflict]:
        """Analyze conflict between existing and new pattern"""

        # Create cache key
        cache_key = self._create_cache_key(existing.get("pattern", ""), new.regex_pattern)

        if cache_key in self.conflict_cache:
            return self.conflict_cache[cache_key]

        conflicts_found = []

        # 1. Regex overlap detection
        overlap_conflict = self._detect_regex_overlap(existing, new)
        if overlap_conflict:
            conflicts_found.append(overlap_conflict)

        # 2. Field mapping conflicts
        field_conflict = self._detect_field_conflicts(existing, new)
        if field_conflict:
            conflicts_found.append(field_conflict)

        # 3. Performance conflicts
        performance_conflict = self._detect_performance_conflicts(existing, new)
        if performance_conflict:
            conflicts_found.append(performance_conflict)

        # 4. Semantic conflicts
        semantic_conflict = self._detect_semantic_conflicts(existing, new)
        if semantic_conflict:
            conflicts_found.append(semantic_conflict)

        # Choose most severe conflict
        if conflicts_found:
            primary_conflict = max(conflicts_found, key=lambda x: x.severity.value)
            self.conflict_cache[cache_key] = primary_conflict
            return primary_conflict

        return None

    def _detect_regex_overlap(self, existing: Dict[str, Any], new: PatternSuggestion) -> Optional[PatternConflict]:
        """Detect regex pattern overlaps"""

        existing_pattern = existing.get("pattern", "")
        new_pattern = new.regex_pattern

        if not existing_pattern or not new_pattern:
            return None

        # Test sample strings against both patterns
        test_strings = self._generate_test_strings(existing, new)

        overlapping_matches = []
        conflicting_matches = []

        for test_string in test_strings:
            existing_match = self._safe_regex_match(existing_pattern, test_string)
            new_match = self._safe_regex_match(new_pattern, test_string)

            if existing_match and new_match:
                # Both patterns match - check if they extract different data
                if self._are_extractions_different(existing_match, new_match):
                    conflicting_matches.append(test_string)
                else:
                    overlapping_matches.append(test_string)

        # Determine conflict type and severity
        if conflicting_matches:
            severity = ConflictSeverity.HIGH if len(conflicting_matches) > 2 else ConflictSeverity.MEDIUM
            conflict_type = ConflictType.REGEX_OVERLAP

            conflict_details = {
                "overlapping_strings": overlapping_matches[:5],
                "conflicting_strings": conflicting_matches[:5],
                "overlap_ratio": len(overlapping_matches) / len(test_strings),
                "conflict_ratio": len(conflicting_matches) / len(test_strings),
            }

            return self._create_conflict(existing, new, conflict_type, severity, conflict_details)

        elif overlapping_matches and len(overlapping_matches) > len(test_strings) * 0.5:
            # High overlap but consistent extractions - medium severity
            severity = ConflictSeverity.MEDIUM
            conflict_type = ConflictType.PATTERN_REDUNDANCY

            conflict_details = {
                "overlapping_strings": overlapping_matches[:5],
                "overlap_ratio": len(overlapping_matches) / len(test_strings),
            }

            return self._create_conflict(existing, new, conflict_type, severity, conflict_details)

        return None

    def _detect_field_conflicts(self, existing: Dict[str, Any], new: PatternSuggestion) -> Optional[PatternConflict]:
        """Detect field mapping conflicts"""

        existing_fields = set(existing.get("fields", []))
        new_fields = set(new.expected_fields)

        if not existing_fields or not new_fields:
            return None

        # Check for field name conflicts
        field_overlap = existing_fields & new_fields

        if field_overlap:
            # Same field names but potentially different meanings
            severity = ConflictSeverity.MEDIUM
            conflict_type = ConflictType.FIELD_MAPPING

            conflict_details = {
                "overlapping_fields": list(field_overlap),
                "existing_fields": list(existing_fields),
                "new_fields": list(new_fields),
                "field_overlap_ratio": len(field_overlap) / len(existing_fields | new_fields),
            }

            return self._create_conflict(existing, new, conflict_type, severity, conflict_details)

        return None

    def _detect_performance_conflicts(self, existing: Dict[str, Any], new: PatternSuggestion) -> Optional[PatternConflict]:
        """Detect performance-related conflicts"""

        # Check regex complexity
        existing_complexity = self._calculate_regex_complexity(existing.get("pattern", ""))
        new_complexity = self._calculate_regex_complexity(new.regex_pattern)

        # Check if new pattern is significantly more complex
        if new_complexity > existing_complexity * 2 and new_complexity > 100:
            severity = ConflictSeverity.LOW
            conflict_type = ConflictType.PERFORMANCE

            conflict_details = {
                "existing_complexity": existing_complexity,
                "new_complexity": new_complexity,
                "complexity_ratio": new_complexity / existing_complexity if existing_complexity > 0 else float("inf"),
            }

            return self._create_conflict(existing, new, conflict_type, severity, conflict_details)

        return None

    def _detect_semantic_conflicts(self, existing: Dict[str, Any], new: PatternSuggestion) -> Optional[PatternConflict]:
        """Detect semantic conflicts"""

        # Check for similar descriptions or purposes
        existing_desc = existing.get("description", "").lower()
        new_desc = new.description.lower()

        if existing_desc and new_desc:
            # Simple keyword overlap check
            existing_words = set(existing_desc.split())
            new_words = set(new_desc.split())

            word_overlap = existing_words & new_words

            if len(word_overlap) > 2 and len(word_overlap) / len(existing_words | new_words) > 0.3:
                severity = ConflictSeverity.LOW
                conflict_type = ConflictType.SEMANTIC_OVERLAP

                conflict_details = {
                    "existing_description": existing_desc,
                    "new_description": new_desc,
                    "common_words": list(word_overlap),
                    "semantic_similarity": len(word_overlap) / len(existing_words | new_words),
                }

                return self._create_conflict(existing, new, conflict_type, severity, conflict_details)

        return None

    def _detect_internal_conflicts(self, patterns: List[PatternSuggestion]) -> List[PatternConflict]:
        """Detect conflicts among new patterns themselves"""

        internal_conflicts = []

        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i + 1 :], i + 1):
                conflict = self._analyze_new_pattern_pair(pattern1, pattern2)
                if conflict:
                    internal_conflicts.append(conflict)

        return internal_conflicts

    def _analyze_new_pattern_pair(self, pattern1: PatternSuggestion, pattern2: PatternSuggestion) -> Optional[PatternConflict]:
        """Analyze conflict between two new patterns"""

        # Convert PatternSuggestion to dict format for consistency
        pattern1_dict = {
            "pattern": pattern1.regex_pattern,
            "fields": pattern1.expected_fields,
            "description": pattern1.description,
            "confidence": pattern1.confidence_score,
        }

        return self._analyze_pattern_pair(pattern1_dict, pattern2)

    def _generate_test_strings(self, existing: Dict[str, Any], new: PatternSuggestion) -> List[str]:
        """Generate test strings for pattern analysis"""

        test_strings = []

        # Use sample matches from new pattern
        if new.sample_matches:
            test_strings.extend(new.sample_matches)

        # Use source cluster addresses if available
        if hasattr(new, "source_cluster") and new.source_cluster:
            test_strings.extend(new.source_cluster.addresses[:5])

        # Generate synthetic test strings
        synthetic_strings = self._generate_synthetic_addresses()
        test_strings.extend(synthetic_strings)

        return list(set(test_strings))  # Remove duplicates

    def _generate_synthetic_addresses(self) -> List[str]:
        """Generate synthetic address strings for testing"""

        templates = [
            "Atatürk Mahallesi, Cumhuriyet Caddesi No:25",
            "Merkez Mah. İstiklal Sok. 15/3",
            "Yenişehir Mahallesi Gazi Bulvarı 42",
            "Kültür Mah, Barış Sk, No:8",
            "Bahçelievler Mahallesi 15. Sokak No:23 Daire:4",
        ]

        return templates

    def _safe_regex_match(self, pattern: str, text: str) -> Optional[re.Match]:
        """Safely attempt regex matching"""

        try:
            return re.search(pattern, text, re.IGNORECASE)
        except re.error:
            return None

    def _are_extractions_different(self, match1: re.Match, match2: re.Match) -> bool:
        """Check if two regex matches extract different data"""

        # Compare group dictionaries
        groups1 = match1.groupdict() if hasattr(match1, "groupdict") else {}
        groups2 = match2.groupdict() if hasattr(match2, "groupdict") else {}

        # If different number of groups, they're different
        if len(groups1) != len(groups2):
            return True

        # Check for value differences in common keys
        common_keys = set(groups1.keys()) & set(groups2.keys())

        for key in common_keys:
            if groups1[key] != groups2[key]:
                return True

        return False

    def _calculate_regex_complexity(self, pattern: str) -> int:
        """Calculate regex complexity score"""

        if not pattern:
            return 0

        complexity = 0

        # Basic length
        complexity += len(pattern)

        # Special character penalties
        special_chars = r"*+?{}[]()^$|\.\\/"
        complexity += sum(pattern.count(char) * 2 for char in special_chars)

        # Group penalties
        complexity += pattern.count("(") * 3

        # Lookahead/lookbehind penalties
        complexity += pattern.count("?=") * 5
        complexity += pattern.count("?!") * 5
        complexity += pattern.count("?<=") * 5
        complexity += pattern.count("?<!") * 5

        return complexity

    def _create_conflict(
        self,
        existing: Dict[str, Any],
        new: PatternSuggestion,
        conflict_type: ConflictType,
        severity: ConflictSeverity,
        details: Dict[str, Any],
    ) -> PatternConflict:
        """Create PatternConflict object"""

        conflict_id = self._generate_conflict_id(existing, new, conflict_type)

        # Generate resolution suggestions
        resolutions = self._suggest_resolutions(existing, new, conflict_type, details)

        return PatternConflict(
            conflict_id=conflict_id,
            existing_pattern_id=existing.get("id", "unknown"),
            new_pattern_id=new.pattern_id,
            conflict_type=conflict_type,
            severity=severity,
            description=self._generate_conflict_description(conflict_type, severity, details),
            affected_samples=details.get("conflicting_strings", []),
            resolution_suggestions=resolutions,
            conflict_details=details,
            detected_at=datetime.now(),
        )

    def _suggest_resolutions(
        self, existing: Dict[str, Any], new: PatternSuggestion, conflict_type: ConflictType, details: Dict[str, Any]
    ) -> List[ConflictResolution]:
        """Suggest resolution strategies for conflict"""

        resolutions = []

        if conflict_type == ConflictType.REGEX_OVERLAP:
            resolutions.extend(self._suggest_overlap_resolutions(existing, new, details))

        elif conflict_type == ConflictType.FIELD_MAPPING:
            resolutions.extend(self._suggest_field_resolutions(existing, new, details))

        elif conflict_type == ConflictType.PATTERN_REDUNDANCY:
            resolutions.extend(self._suggest_redundancy_resolutions(existing, new, details))

        elif conflict_type == ConflictType.PERFORMANCE:
            resolutions.extend(self._suggest_performance_resolutions(existing, new, details))

        elif conflict_type == ConflictType.SEMANTIC_OVERLAP:
            resolutions.extend(self._suggest_semantic_resolutions(existing, new, details))

        # Add generic resolutions
        resolutions.extend(self._suggest_generic_resolutions(existing, new))

        return resolutions

    def _suggest_overlap_resolutions(
        self, existing: Dict[str, Any], new: PatternSuggestion, details: Dict[str, Any]
    ) -> List[ConflictResolution]:
        """Suggest resolutions for regex overlap conflicts"""

        resolutions = []

        # Resolution 1: Merge patterns
        if details.get("overlap_ratio", 0) > 0.7:
            resolutions.append(
                ConflictResolution(
                    strategy=ResolutionStrategy.MERGE_PATTERNS,
                    description="Merge overlapping patterns into a single comprehensive pattern",
                    confidence_score=0.8,
                    implementation_notes="Combine regex patterns using alternation (|) operator",
                    impact_assessment="Reduces pattern count, may increase complexity",
                )
            )

        # Resolution 2: Add specificity
        resolutions.append(
            ConflictResolution(
                strategy=ResolutionStrategy.ADD_SPECIFICITY,
                description="Add more specific conditions to distinguish patterns",
                confidence_score=0.7,
                implementation_notes="Add lookahead/lookbehind assertions or more specific character classes",
                impact_assessment="Increases pattern precision, may reduce coverage",
            )
        )

        # Resolution 3: Priority-based selection
        resolutions.append(
            ConflictResolution(
                strategy=ResolutionStrategy.PRIORITY_BASED,
                description="Use confidence scores to determine pattern priority",
                confidence_score=0.6,
                implementation_notes="Apply patterns in order of confidence score",
                impact_assessment="Maintains all patterns but adds execution overhead",
            )
        )

        return resolutions

    def _suggest_field_resolutions(
        self, existing: Dict[str, Any], new: PatternSuggestion, details: Dict[str, Any]
    ) -> List[ConflictResolution]:
        """Suggest resolutions for field mapping conflicts"""

        resolutions = []

        # Resolution 1: Rename fields
        resolutions.append(
            ConflictResolution(
                strategy=ResolutionStrategy.RENAME_FIELDS,
                description="Rename conflicting field names to avoid ambiguity",
                confidence_score=0.9,
                implementation_notes="Add prefixes or suffixes to distinguish field purposes",
                impact_assessment="Requires updating field mapping configuration",
            )
        )

        # Resolution 2: Merge field mappings
        if len(details.get("overlapping_fields", [])) > 1:
            resolutions.append(
                ConflictResolution(
                    strategy=ResolutionStrategy.MERGE_PATTERNS,
                    description="Merge patterns with compatible field mappings",
                    confidence_score=0.7,
                    implementation_notes="Combine patterns and create unified field mapping",
                    impact_assessment="Simplifies configuration but may reduce specificity",
                )
            )

        return resolutions

    def _suggest_redundancy_resolutions(
        self, existing: Dict[str, Any], new: PatternSuggestion, details: Dict[str, Any]
    ) -> List[ConflictResolution]:
        """Suggest resolutions for pattern redundancy"""

        resolutions = []

        # Choose better pattern based on confidence
        existing_confidence = existing.get("confidence", 0.5)
        new_confidence = new.confidence_score

        if new_confidence > existing_confidence + 0.1:
            resolutions.append(
                ConflictResolution(
                    strategy=ResolutionStrategy.REPLACE_PATTERN,
                    description="Replace existing pattern with higher-confidence new pattern",
                    confidence_score=0.8,
                    implementation_notes="Remove existing pattern and activate new pattern",
                    impact_assessment="Improves accuracy, may affect existing integrations",
                )
            )
        else:
            resolutions.append(
                ConflictResolution(
                    strategy=ResolutionStrategy.KEEP_EXISTING,
                    description="Keep existing pattern, discard redundant new pattern",
                    confidence_score=0.7,
                    implementation_notes="No changes required to existing configuration",
                    impact_assessment="Maintains stability, may miss improvement opportunities",
                )
            )

        return resolutions

    def _suggest_performance_resolutions(
        self, existing: Dict[str, Any], new: PatternSuggestion, details: Dict[str, Any]
    ) -> List[ConflictResolution]:
        """Suggest resolutions for performance conflicts"""

        resolutions = []

        # Resolution 1: Optimize pattern
        resolutions.append(
            ConflictResolution(
                strategy=ResolutionStrategy.OPTIMIZE_PATTERN,
                description="Simplify regex pattern to improve performance",
                confidence_score=0.8,
                implementation_notes="Remove unnecessary groups, use non-capturing groups, optimize quantifiers",
                impact_assessment="Improves performance, may require testing for accuracy",
            )
        )

        # Resolution 2: Add timeout
        resolutions.append(
            ConflictResolution(
                strategy=ResolutionStrategy.ADD_CONSTRAINTS,
                description="Add regex timeout to prevent performance issues",
                confidence_score=0.6,
                implementation_notes="Set regex engine timeout to limit execution time",
                impact_assessment="Prevents hangs but may miss some matches",
            )
        )

        return resolutions

    def _suggest_semantic_resolutions(
        self, existing: Dict[str, Any], new: PatternSuggestion, details: Dict[str, Any]
    ) -> List[ConflictResolution]:
        """Suggest resolutions for semantic conflicts"""

        resolutions = []

        # Resolution: Update descriptions
        resolutions.append(
            ConflictResolution(
                strategy=ResolutionStrategy.UPDATE_METADATA,
                description="Update pattern descriptions to clarify differences",
                confidence_score=0.9,
                implementation_notes="Add more specific descriptions and usage examples",
                impact_assessment="Improves maintainability, no functional impact",
            )
        )

        return resolutions

    def _suggest_generic_resolutions(self, existing: Dict[str, Any], new: PatternSuggestion) -> List[ConflictResolution]:
        """Suggest generic resolution strategies"""

        resolutions = []

        # Manual review resolution
        resolutions.append(
            ConflictResolution(
                strategy=ResolutionStrategy.MANUAL_REVIEW,
                description="Require manual review and decision",
                confidence_score=0.5,
                implementation_notes="Flag for human review with detailed conflict analysis",
                impact_assessment="Ensures accuracy but requires human intervention",
            )
        )

        return resolutions

    def _generate_conflict_description(
        self, conflict_type: ConflictType, severity: ConflictSeverity, details: Dict[str, Any]
    ) -> str:
        """Generate human-readable conflict description"""

        if conflict_type == ConflictType.REGEX_OVERLAP:
            overlap_ratio = details.get("overlap_ratio", 0)
            return f"Regex patterns overlap with {overlap_ratio:.1%} of test cases matching both patterns"

        elif conflict_type == ConflictType.FIELD_MAPPING:
            overlapping_fields = details.get("overlapping_fields", [])
            return f"Field mapping conflict: overlapping fields {', '.join(overlapping_fields)}"

        elif conflict_type == ConflictType.PATTERN_REDUNDANCY:
            overlap_ratio = details.get("overlap_ratio", 0)
            return f"Pattern redundancy detected with {overlap_ratio:.1%} overlap"

        elif conflict_type == ConflictType.PERFORMANCE:
            complexity_ratio = details.get("complexity_ratio", 1)
            return f"Performance concern: new pattern is {complexity_ratio:.1f}x more complex"

        elif conflict_type == ConflictType.SEMANTIC_OVERLAP:
            similarity = details.get("semantic_similarity", 0)
            return f"Semantic similarity detected ({similarity:.1%} description overlap)"

        else:
            return f"{conflict_type.value} conflict with {severity.value} severity"

    def _generate_conflict_id(self, existing: Dict[str, Any], new: PatternSuggestion, conflict_type: ConflictType) -> str:
        """Generate unique conflict ID"""

        content = f"{existing.get('id', 'unknown')}_{new.pattern_id}_{conflict_type.value}"
        return f"conflict_{hashlib.md5(content.encode()).hexdigest()[:8]}"

    def _create_cache_key(self, pattern1: str, pattern2: str) -> str:
        """Create cache key for conflict analysis"""

        # Sort patterns to ensure consistent cache keys
        patterns = sorted([pattern1, pattern2])
        content = "_".join(patterns)
        return hashlib.md5(content.encode()).hexdigest()[:16]
