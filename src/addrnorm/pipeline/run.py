"""
End-to-end pipeline for Turkish address normalization.
Integrates preprocessing, pattern matching, ML NER, fallback, validation.
"""

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..fallback import normalize_address_fallback
from ..ml.infer import MLAddressNormalizer
from ..patterns.matcher import PatternMatcher
from ..patterns.thresholds import ThresholdManager

# Import pipeline components
from ..preprocess.core import clean_address, normalize_case, normalize_unicode
from ..utils.contracts import AddressOut, ExplanationParsed, MethodEnum
from ..validate import create_geo_validator


@dataclass
class PipelineConfig:
    """Configuration for the address normalization pipeline."""

    # Pattern matching thresholds
    pattern_threshold_high: float = 0.8
    pattern_threshold_medium: float = 0.6
    pattern_threshold_low: float = 0.4

    # ML model settings
    ml_model_path: str = "models/turkish_address_ner_improved"
    ml_confidence_threshold: float = 0.7

    # Processing settings
    enable_multiprocessing: bool = False  # Disabled due to spaCy threading issues
    max_workers: Optional[int] = None
    batch_size: int = 100

    # Validation settings
    enable_validation: bool = True
    geo_data_dir: Optional[str] = None

    # Determinism
    random_seed: int = 42

    # Logging
    log_level: str = "INFO"


@dataclass
class ProcessingResult:
    """Result of processing a single address."""

    raw_input: str
    success: bool
    address_out: Optional[AddressOut]
    error: Optional[str]
    processing_method: str
    confidence: float
    processing_time_ms: float


class AddressNormalizationPipeline:
    """
    End-to-end pipeline for Turkish address normalization.

    Pipeline stages:
    1. Preprocess - Clean and normalize text
    2. Pattern Match - Try pattern-based extraction with dynamic thresholds
    3. ML NER - Use ML model if pattern confidence is low
    4. Fallback - Use rule-based fallback if ML fails
    5. Explanation Parse - Parse and format explanation
    6. Validate - Geographic validation and consistency checks
    7. Format - Create final AddressOut object
    """

    def __init__(self, config: PipelineConfig = None):
        """Initialize the pipeline with given configuration."""

        self.config = config or PipelineConfig()

        # Set up deterministic behavior
        self._setup_determinism()

        # Set up logging
        self._setup_logging()

        # Initialize components
        self.logger.info("Initializing pipeline components...")
        self._initialize_components()

        self.logger.info("Pipeline initialization complete")

    def _setup_determinism(self):
        """Set up deterministic behavior for reproducible results."""

        random.seed(self.config.random_seed)
        # Note: For full determinism, would also need to set numpy and torch seeds
        # if using ML components that depend on them

    def _setup_logging(self):
        """Set up logging configuration."""

        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_components(self):
        """Initialize all pipeline components."""

        try:
            # Pattern matcher
            self.logger.info("Loading pattern matcher...")
            self.threshold_manager = ThresholdManager()
            self.pattern_matcher = PatternMatcher(
                threshold_manager=self.threshold_manager
            )

            # ML NER inference
            self.logger.info(f"Loading ML model from {self.config.ml_model_path}...")
            self.ml_inference = None
            if os.path.exists(self.config.ml_model_path):
                try:
                    self.ml_inference = MLAddressNormalizer(
                        model_path=self.config.ml_model_path
                    )
                    self.logger.info("ML model loaded successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to load ML model: {e}")
            else:
                self.logger.warning(
                    f"ML model path not found: {self.config.ml_model_path}"
                )

            # Geographic validator
            if self.config.enable_validation:
                self.logger.info("Loading geographic validator...")
                self.geo_validator = create_geo_validator(self.config.geo_data_dir)
            else:
                self.geo_validator = None

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    def process_single(self, raw_address: str) -> ProcessingResult:
        """
        Process a single address through the complete pipeline.

        Args:
            raw_address: Raw address string to process

        Returns:
            ProcessingResult with normalized address or error
        """

        import time

        start_time = time.time()

        try:
            # Stage 1: Preprocess
            self.logger.debug(f"Processing: {raw_address}")
            cleaned = self._preprocess(raw_address)

            # Stage 2: Pattern Match
            pattern_result = self._pattern_match(cleaned)

            # Determine if pattern confidence is sufficient
            if (
                pattern_result
                and pattern_result.get("confidence", 0)
                >= self.config.pattern_threshold_high
            ):
                # High confidence pattern match
                address_out = self._create_address_out_from_pattern(
                    pattern_result, raw_address, MethodEnum.PATTERN
                )
                method = "pattern_high"
                confidence = pattern_result["confidence"]

            elif (
                pattern_result
                and pattern_result.get("confidence", 0)
                >= self.config.pattern_threshold_medium
            ):
                # Medium confidence - validate with ML if available
                if self.ml_inference:
                    ml_result = self._ml_extract(cleaned)
                    if (
                        ml_result
                        and ml_result.get("confidence", 0)
                        >= self.config.ml_confidence_threshold
                    ):
                        # Use ML result
                        address_out = self._create_address_out_from_ml(
                            ml_result, raw_address, MethodEnum.ML
                        )
                        method = "ml_validation"
                        confidence = ml_result["confidence"]
                    else:
                        # Fall back to pattern
                        address_out = self._create_address_out_from_pattern(
                            pattern_result, raw_address, MethodEnum.PATTERN
                        )
                        method = "pattern_medium"
                        confidence = pattern_result["confidence"]
                else:
                    # No ML available, use pattern
                    address_out = self._create_address_out_from_pattern(
                        pattern_result, raw_address, MethodEnum.PATTERN
                    )
                    method = "pattern_medium"
                    confidence = pattern_result["confidence"]

            else:
                # Low or no pattern confidence - try ML
                if self.ml_inference:
                    ml_result = self._ml_extract(cleaned)
                    if (
                        ml_result
                        and ml_result.get("confidence", 0)
                        >= self.config.ml_confidence_threshold
                    ):
                        address_out = self._create_address_out_from_ml(
                            ml_result, raw_address, MethodEnum.ML
                        )
                        method = "ml_primary"
                        confidence = ml_result["confidence"]
                    else:
                        # Fall back to rules
                        address_out = self._fallback_extract(raw_address)
                        method = "fallback"
                        confidence = address_out.explanation_parsed.confidence
                else:
                    # No ML, use fallback
                    address_out = self._fallback_extract(raw_address)
                    method = "fallback"
                    confidence = address_out.explanation_parsed.confidence

            # Stage 6: Validate
            if self.config.enable_validation and self.geo_validator:
                address_out = self._validate_address(address_out)

            processing_time = (time.time() - start_time) * 1000

            return ProcessingResult(
                raw_input=raw_address,
                success=True,
                address_out=address_out,
                error=None,
                processing_method=method,
                confidence=confidence,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"Processing failed: {str(e)}"
            self.logger.error(error_msg)

            return ProcessingResult(
                raw_input=raw_address,
                success=False,
                address_out=None,
                error=error_msg,
                processing_method="error",
                confidence=0.0,
                processing_time_ms=processing_time,
            )

    def _preprocess(self, text: str) -> str:
        """Stage 1: Preprocess the input text."""

        # Clean and normalize
        cleaned = clean_address(text)
        normalized = normalize_unicode(normalize_case(cleaned))

        return normalized

    def _pattern_match(self, text: str) -> Optional[Dict[str, Any]]:
        """Stage 2: Try pattern-based matching."""

        try:
            matches = self.pattern_matcher.match_text(text)

            if matches:
                best_match = matches[0]  # Highest confidence match

                return {
                    "confidence": best_match.confidence,
                    "pattern_id": best_match.pattern_id,
                    "slots": best_match.slots,
                    "method": "pattern",
                }

        except Exception as e:
            self.logger.warning(f"Pattern matching failed: {e}")

        return None

    def _ml_extract(self, text: str) -> Optional[Dict[str, Any]]:
        """Stage 3: Try ML-based extraction."""

        if not self.ml_inference:
            return None

        try:
            result = self.ml_inference.extract_entities_ml(text)

            if result and result.get("entities"):
                # Convert entities to address components
                entities = result["entities"]
                components = self._entities_to_components(entities)
                confidence = self._calculate_ml_confidence(entities)

                return {
                    "confidence": confidence,
                    "components": components,
                    "entities": entities,
                    "method": "ml",
                }

        except Exception as e:
            self.logger.warning(f"ML extraction failed: {e}")

        return None

    def _fallback_extract(self, text: str) -> AddressOut:
        """Stage 4: Use rule-based fallback extraction."""

        try:
            # Use the fallback normalizer
            return normalize_address_fallback({}, text)

        except Exception as e:
            self.logger.warning(f"Fallback extraction failed: {e}")

            # Create minimal AddressOut
            return AddressOut(
                explanation_raw=text,
                explanation_parsed=ExplanationParsed(
                    confidence=0.0,
                    method=MethodEnum.FALLBACK,
                    warnings=[f"Fallback failed: {str(e)}"],
                ),
                normalized_address="Processing failed",
            )

    def _create_address_out_from_pattern(
        self, pattern_result: Dict, raw_input: str, method: MethodEnum
    ) -> AddressOut:
        """Create AddressOut from pattern matching result."""

        # Map pattern slots to AddressOut fields
        field_mapping = {
            "city": "city",
            "district": "district",
            "neighborhood": "neighborhood",
            "street": "street",
            "building": "building",
            "block": "block",
            "number": "number",
            "entrance": "entrance",
            "floor": "floor",
            "apartment": "apartment",
            "postcode": "postcode",
        }

        address_kwargs = {
            "explanation_raw": raw_input,
            "explanation_parsed": ExplanationParsed(
                confidence=pattern_result["confidence"], method=method, warnings=[]
            ),
            "normalized_address": self._build_normalized_address(
                pattern_result["slots"]
            ),
        }

        # Add components from slots
        for slot_name, value in pattern_result["slots"].items():
            if slot_name in field_mapping and value:
                field_name = field_mapping[slot_name]
                address_kwargs[field_name] = str(value).strip()

        return AddressOut(**address_kwargs)

    def _create_address_out_from_ml(
        self, ml_result: Dict, raw_input: str, method: MethodEnum
    ) -> AddressOut:
        """Create AddressOut from ML extraction result."""

        address_kwargs = {
            "explanation_raw": raw_input,
            "explanation_parsed": ExplanationParsed(
                confidence=ml_result["confidence"], method=method, warnings=[]
            ),
            "normalized_address": self._build_normalized_address(
                ml_result["components"]
            ),
        }

        # Add components
        for field_name, value in ml_result["components"].items():
            if value:
                address_kwargs[field_name] = str(value).strip()

        return AddressOut(**address_kwargs)

    def _entities_to_components(self, entities: List[Dict]) -> Dict[str, str]:
        """Convert ML entities to address components."""

        # Map entity labels to AddressOut fields
        entity_mapping = {
            "IL": "city",
            "ILCE": "district",
            "MAH": "neighborhood",
            "SOKAK": "street",
            "CADDE": "street",
            "BULVAR": "street",
            "NO": "number",
            "KAT": "floor",
            "DAIRE": "apartment",
            "BLOK": "block",
        }

        components = {}

        for entity in entities:
            label = entity.get("label", "")
            text = entity.get("text", "").strip()

            if label in entity_mapping and text:
                field = entity_mapping[label]

                # Handle multiple street entities (combine them)
                if field == "street" and field in components:
                    components[field] = f"{components[field]} {text}"
                else:
                    components[field] = text

        return components

    def _calculate_ml_confidence(self, entities: List[Dict]) -> float:
        """Calculate overall confidence from ML entities."""

        if not entities:
            return 0.0

        # Simple average of entity confidences
        confidences = []
        for entity in entities:
            conf = entity.get("confidence", entity.get("score", 0.5))
            confidences.append(float(conf))

        return sum(confidences) / len(confidences)

    def _build_normalized_address(self, components: Dict[str, str]) -> str:
        """Build normalized address string from components."""

        # Order components appropriately
        order = [
            "neighborhood",
            "street",
            "building",
            "block",
            "number",
            "entrance",
            "floor",
            "apartment",
            "district",
            "city",
            "postcode",
        ]

        parts = []
        for field in order:
            if field in components and components[field]:
                value = components[field]

                # Add appropriate Turkish suffixes if missing
                if field == "neighborhood" and not any(
                    suffix in value.lower() for suffix in ["mahalle", "mah"]
                ):
                    value = f"{value} Mahallesi"
                elif field == "number" and not value.lower().startswith("no"):
                    value = f"No: {value}"
                elif field == "floor" and not value.lower().startswith("kat"):
                    value = f"Kat: {value}"
                elif field == "apartment" and not value.lower().startswith("daire"):
                    value = f"Daire: {value}"

                parts.append(value)

        return ", ".join(parts) if parts else "Adres bilgisi eksik"

    def _validate_address(self, address_out: AddressOut) -> AddressOut:
        """Stage 6: Validate address components."""

        if not self.geo_validator:
            return address_out

        warnings = list(address_out.explanation_parsed.warnings)

        # Validate city
        if address_out.city:
            city_result = self.geo_validator.validate_city(address_out.city)
            if (
                city_result.standardized_value
                and city_result.standardized_value != address_out.city
            ):
                # Update city with standardized value
                address_out.city = city_result.standardized_value
                warnings.extend(city_result.warnings)

        # Validate district
        if address_out.district:
            district_result = self.geo_validator.validate_district(
                address_out.district, address_out.city
            )
            if (
                district_result.standardized_value
                and district_result.standardized_value != address_out.district
            ):
                address_out.district = district_result.standardized_value
                warnings.extend(district_result.warnings)

        # Validate city-district consistency
        if address_out.city and address_out.district:
            consistency = self.geo_validator.validate_city_district_consistency(
                address_out.city, address_out.district
            )
            if not consistency.is_consistent:
                warnings.extend(consistency.warnings)

        # Validate postcode
        if address_out.postcode:
            postcode_result = self.geo_validator.validate_postcode(address_out.postcode)
            if (
                postcode_result.standardized_value
                and postcode_result.standardized_value != address_out.postcode
            ):
                address_out.postcode = postcode_result.standardized_value
                warnings.extend(postcode_result.warnings)

        # Update warnings
        address_out.explanation_parsed.warnings = warnings

        return address_out

    def process_batch(self, addresses: List[str]) -> List[ProcessingResult]:
        """
        Process a batch of addresses.

        Args:
            addresses: List of raw address strings

        Returns:
            List of ProcessingResult objects
        """

        # Always process sequentially to avoid pickle issues with spaCy models
        return [self.process_single(addr) for addr in addresses]

    def _process_single_worker(self, address: str) -> ProcessingResult:
        """Worker function for multiprocessing."""
        # Note: In practice, each worker would need to reinitialize the pipeline
        # For now, this is a placeholder
        return self.process_single(address)

    def process_file(
        self, input_file: Union[str, Path], output_file: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Process addresses from a file and write results to output file.

        Args:
            input_file: Path to input file (one address per line)
            output_file: Path to output file (JSON lines format)

        Returns:
            Processing statistics
        """

        input_path = Path(input_file)
        output_path = Path(output_file)

        # Read input addresses
        with open(input_path, "r", encoding="utf-8") as f:
            addresses = [line.strip() for line in f if line.strip()]

        self.logger.info(f"Processing {len(addresses)} addresses from {input_path}")

        # Process in batches
        all_results = []
        batch_size = self.config.batch_size

        for i in range(0, len(addresses), batch_size):
            batch = addresses[i : i + batch_size]
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)

            self.logger.info(
                f"Processed batch {i//batch_size + 1}/"
                f"{(len(addresses) + batch_size - 1)//batch_size}"
            )

        # Write results
        with open(output_path, "w", encoding="utf-8") as f:
            for result in all_results:
                if result.success and result.address_out:
                    # Write as JSON line
                    json_line = result.address_out.to_json()
                    f.write(json_line + "\n")
                else:
                    # Write error information
                    error_record = {
                        "explanation_raw": result.raw_input,
                        "error": result.error,
                        "processing_method": result.processing_method,
                    }
                    f.write(json.dumps(error_record, ensure_ascii=False) + "\n")

        # Calculate statistics
        successful = sum(1 for r in all_results if r.success)
        failed = len(all_results) - successful
        avg_confidence = sum(r.confidence for r in all_results if r.success) / max(
            successful, 1
        )
        avg_time = sum(r.processing_time_ms for r in all_results) / len(all_results)

        methods = {}
        for r in all_results:
            methods[r.processing_method] = methods.get(r.processing_method, 0) + 1

        stats = {
            "total_processed": len(all_results),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(all_results),
            "average_confidence": avg_confidence,
            "average_time_ms": avg_time,
            "methods_used": methods,
            "output_file": str(output_path),
        }

        self.logger.info(f"Processing complete. Stats: {stats}")
        return stats


def create_pipeline(config: PipelineConfig = None) -> AddressNormalizationPipeline:
    """Factory function to create a pipeline instance."""
    return AddressNormalizationPipeline(config)
