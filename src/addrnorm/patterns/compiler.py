"""
Pattern compiler for Turkish address patterns.

Converts DSL patterns from YAML to compiled regex patterns with slot information.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class CompiledSlot:
    """Information about a compiled slot in a pattern."""

    name: str
    is_optional: bool
    slot_type: str  # 'named', 'number', 'text'
    group_index: int  # Regex group index
    weight: float = 1.0


@dataclass
class CompiledPattern:
    """A compiled pattern with regex and slot information."""

    id: str
    priority: int
    pattern_text: str
    regex: re.Pattern
    slots: List[CompiledSlot]
    description: str
    examples: List[str]

    def __post_init__(self):
        """Calculate total weight for normalization."""
        self.total_weight = sum(
            slot.weight for slot in self.slots if not slot.is_optional
        )


class PatternCompiler:
    """Compiles DSL patterns into regex patterns with slot metadata."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize compiler with pattern configuration."""
        if config_path is None:
            # Look for patterns config relative to project root
            current_file = Path(__file__)
            project_root = (
                current_file.parent.parent.parent.parent
            )  # Go up to project root
            config_path = project_root / "data" / "patterns" / "tr.yml"

        self.config_path = Path(config_path)
        self.patterns_config = self._load_config()
        self.compiled_patterns: List[CompiledPattern] = []
        self.slot_weights = self.patterns_config.get("scoring", {}).get(
            "slot_weights", {}
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load pattern configuration from YAML file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Pattern config file not found: {self.config_path}"
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in pattern config: {e}")

    def compile_all_patterns(self) -> List[CompiledPattern]:
        """Compile all patterns from the configuration."""
        patterns_data = self.patterns_config.get("patterns", [])
        compiled = []

        for pattern_data in patterns_data:
            try:
                compiled_pattern = self._compile_single_pattern(pattern_data)
                compiled.append(compiled_pattern)
            except Exception as e:
                print(
                    f"Warning: Failed to compile pattern "
                    f"{pattern_data.get('id', 'unknown')}: {e}"
                )
                continue

        # Sort by priority (higher first)
        compiled.sort(key=lambda p: p.priority, reverse=True)
        self.compiled_patterns = compiled
        return compiled

    def _compile_single_pattern(self, pattern_data: Dict[str, Any]) -> CompiledPattern:
        """Compile a single pattern from configuration data."""
        pattern_id = pattern_data["id"]
        priority = pattern_data["priority"]
        pattern_text = pattern_data["pattern"]
        description = pattern_data.get("description", "")
        examples = pattern_data.get("examples", [])

        # Parse pattern and generate regex
        regex_pattern, slots = self._parse_pattern(pattern_text)
        compiled_regex = re.compile(regex_pattern, re.IGNORECASE)

        return CompiledPattern(
            id=pattern_id,
            priority=priority,
            pattern_text=pattern_text,
            regex=compiled_regex,
            slots=slots,
            description=description,
            examples=examples,
        )

    def _parse_pattern(self, pattern_text: str) -> Tuple[str, List[CompiledSlot]]:
        """Parse DSL pattern into regex and slot information."""
        slots = []
        regex_parts = []
        group_index = 1  # Start from 1 (group 0 is full match)

        # Split pattern into tokens while preserving spaces
        tokens = re.findall(r"<[^>]+>|[^\s<]+|\s+", pattern_text)

        for token in tokens:
            if token.startswith("<") and token.endswith(">"):
                # This is a slot
                slot_def = token[1:-1]  # Remove < >

                # Check if optional
                is_optional = slot_def.endswith("?")
                if is_optional:
                    slot_def = slot_def[:-1]

                # Determine slot type and regex pattern
                if slot_def == "n" or slot_def.startswith("n"):
                    # Number slot
                    slot_type = "number"
                    slot_regex = (
                        r"(\d+(?:[a-z])?)"  # Numbers with optional letter suffix
                    )
                elif slot_def == "text":
                    # Generic text slot
                    slot_type = "text"
                    slot_regex = r"(\S+(?:\s+\S+)*?)"  # Non-greedy text capture
                else:
                    # Named slot (location, street, etc.)
                    slot_type = "named"
                    slot_regex = (
                        r"([a-zA-ZçğıöşüÇĞIÖŞÜ]+(?:\s+[a-zA-ZçğıöşüÇĞIÖŞÜ]+)*?)"
                    )

                # Get weight for this slot type
                weight = self.slot_weights.get(slot_def, 1.0)

                # Create slot info
                slot = CompiledSlot(
                    name=slot_def,
                    is_optional=is_optional,
                    slot_type=slot_type,
                    group_index=group_index,
                    weight=weight,
                )
                slots.append(slot)

                # Add to regex with optional wrapper if needed
                if is_optional:
                    regex_parts.append(f"(?:{slot_regex})?")
                else:
                    regex_parts.append(slot_regex)

                group_index += 1

            elif token.isspace():
                # Whitespace - make flexible
                regex_parts.append(r"\s+")
            else:
                # Literal text
                escaped_token = re.escape(token)
                regex_parts.append(escaped_token)

        # Join all parts and wrap in word boundaries
        full_regex = r"\b" + "".join(regex_parts) + r"\b"

        return full_regex, slots

    def get_keywords(self) -> Dict[str, List[str]]:
        """Get keyword lists for pattern matching enhancement."""
        return self.patterns_config.get("keywords", {})

    def get_scoring_config(self) -> Dict[str, Any]:
        """Get scoring configuration."""
        return self.patterns_config.get("scoring", {})

    def validate_pattern(self, pattern_text: str) -> bool:
        """Validate a pattern text for correct DSL syntax."""
        try:
            regex_pattern, slots = self._parse_pattern(pattern_text)
            re.compile(regex_pattern)
            return True
        except Exception:
            return False

    def get_pattern_by_id(self, pattern_id: str) -> Optional[CompiledPattern]:
        """Get a compiled pattern by its ID."""
        for pattern in self.compiled_patterns:
            if pattern.id == pattern_id:
                return pattern
        return None


def main():
    """Demo function to show pattern compilation."""
    compiler = PatternCompiler()
    patterns = compiler.compile_all_patterns()

    print(f"Compiled {len(patterns)} patterns:")
    print()

    for pattern in patterns[:3]:  # Show first 3 patterns
        print(f"ID: {pattern.id}")
        print(f"Priority: {pattern.priority}")
        print(f"Pattern: {pattern.pattern_text}")
        print(f"Regex: {pattern.regex.pattern}")
        print(f"Slots: {len(pattern.slots)}")
        for slot in pattern.slots:
            optional_text = "[optional]" if slot.is_optional else "[required]"
            print(
                f"  - {slot.name} ({slot.slot_type}) {optional_text} "
                f"weight={slot.weight}"
            )
        print(f"Examples: {pattern.examples}")
        print("-" * 60)


if __name__ == "__main__":
    main()
