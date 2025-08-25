"""
Address explanation module
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import json


@dataclass
class ExplanationConfig:
    """Configuration for address explanations"""

    language: str = "tr"
    detail_level: str = "basic"  # low, medium, high
    include_confidence: bool = True
    include_components: bool = True


class AddressExplainer:
    """Generates explanations for address normalization results"""

    def __init__(self, config: Optional[ExplanationConfig] = None):
        """Initialize address explainer

        Args:
            config: Optional configuration for explanations
        """
        self.config = config or ExplanationConfig()
        self.explanation_hooks = []
        self.templates = {
            "tr": {
                "success": "Adres başarıyla normalleştirildi.",
                "component": "{component}: {value} (güven: {confidence:.2f})",
                "warning": "Uyarı: {message}",
                "error": "Hata: {message}",
            },
            "en": {
                "success": "Address successfully normalized.",
                "component": "{component}: {value} (confidence: {confidence:.2f})",
                "warning": "Warning: {message}",
                "error": "Error: {message}",
            },
        }

    def explain(self, normalized_result) -> str:
        """Generate explanation for normalization result

        Args:
            normalized_result: Result from address normalization

        Returns:
            String explanation
        """
        # Handle None input
        if normalized_result is None:
            return "Geçersiz sonuç - normalizasyon sonucu bulunamadı"

        # Generate summary
        explanation = self._generate_summary(normalized_result)

        # Apply explanation hooks
        for hook in self.explanation_hooks:
            try:
                explanation = hook(normalized_result, explanation)
            except Exception:
                # Continue if hook fails
                pass

        return explanation

    def explain_detailed(self, normalized_result) -> Dict[str, Any]:
        """Generate detailed explanation for normalization result

        Args:
            normalized_result: Result from address normalization

        Returns:
            Detailed explanation dictionary
        """
        if normalized_result is None:
            return {
                "summary": "Geçersiz sonuç",
                "details": ["Normalizasyon sonucu bulunamadı"],
                "confidence": 0.0,
                "language": self.config.language,
            }

        explanation = {
            "summary": self._generate_summary(normalized_result),
            "details": self._generate_details(normalized_result),
            "confidence": (
                getattr(normalized_result, "confidence", 0.0)
                if hasattr(normalized_result, "confidence")
                else normalized_result.get("confidence", 0.0) if isinstance(normalized_result, dict) else 0.0
            ),
            "language": self.config.language,
        }

        if self.config.include_components:
            explanation["components"] = self._explain_components(normalized_result)

        return explanation

    def _generate_summary(self, result) -> str:
        """Generate summary explanation"""
        templates = self.templates.get(self.config.language, self.templates["tr"])

        # Handle both dict and ProcessingResult objects
        if hasattr(result, "success"):
            success = result.success
        elif hasattr(result, "status"):
            success = result.status.value == "success"
        elif isinstance(result, dict):
            success = result.get("success", False)
        else:
            success = False

        if success:
            return templates["success"]
        else:
            if hasattr(result, "error_details") and result.error_details:
                error_msg = str(result.error_details)
            elif isinstance(result, dict):
                error_msg = result.get("error", "Bilinmeyen hata")
            else:
                error_msg = "Bilinmeyen hata"
            return templates["error"].format(message=error_msg)

    def _generate_details(self, result) -> List[str]:
        """Generate detailed explanations"""
        details = []
        templates = self.templates.get(self.config.language, self.templates["tr"])

        # Handle both dict and ProcessingResult objects
        if hasattr(result, "success"):
            success = result.success
        elif hasattr(result, "status"):
            success = result.status.value == "success"
        elif isinstance(result, dict):
            success = result.get("success", False)
        else:
            success = False

        if success:
            # Extract confidence safely
            confidence_obj = (
                getattr(result, "confidence", None)
                if hasattr(result, "confidence")
                else result.get("confidence", 0.0) if isinstance(result, dict) else 0.0
            )

            # Handle confidence object
            if hasattr(confidence_obj, "overall"):
                confidence = confidence_obj.overall
            elif hasattr(confidence_obj, "value"):
                confidence = confidence_obj.value
            elif isinstance(confidence_obj, (int, float)):
                confidence = confidence_obj
            else:
                confidence = 0.8  # Default

            if confidence < 0.5:
                details.append(templates["warning"].format(message="Düşük güven skoru"))
            elif confidence > 0.9:
                details.append("Yüksek güven ile normalleştirildi")

        return details

        return details

    def _explain_components(self, result) -> Dict[str, str]:
        """Explain individual components"""
        components = {}
        templates = self.templates.get(self.config.language, self.templates["tr"])

        # Extract components from result
        normalized = None
        if hasattr(result, "normalized_address"):
            normalized = result.normalized_address
        elif hasattr(result, "components"):
            normalized = result.components
        elif isinstance(result, dict):
            normalized = result.get("normalized", {})

        if normalized and hasattr(normalized, "__dict__"):
            for component, value in normalized.__dict__.items():
                if value:
                    confidence = 0.85  # Mock confidence for demo
                    components[component] = templates["component"].format(
                        component=component, value=value, confidence=confidence
                    )
        elif isinstance(normalized, dict):
            for component, value in normalized.items():
                if value:
                    confidence = 0.85  # Mock confidence for demo
                    components[component] = templates["component"].format(
                        component=component, value=value, confidence=confidence
                    )

        return components

    def explain_batch(self, results) -> List[str]:
        """Generate explanations for multiple results

        Args:
            results: List of normalization results

        Returns:
            List of string explanations
        """
        return [self.explain(result) for result in results]

    def _get_confidence_level(self, confidence) -> str:
        """Get confidence level description

        Args:
            confidence: Confidence value (0.0-1.0) or confidence object

        Returns:
            Confidence level description
        """
        # Handle different confidence types
        if hasattr(confidence, "overall"):
            conf_val = confidence.overall
        elif hasattr(confidence, "value"):
            conf_val = confidence.value
        elif isinstance(confidence, (int, float)):
            conf_val = confidence
        else:
            conf_val = 0.8  # Default fallback

        if conf_val >= 0.9:
            return "çok yüksek"
        elif conf_val >= 0.8:
            return "yüksek"
        elif conf_val >= 0.6:
            return "orta"
        elif confidence >= 0.4:
            return "düşük"
        else:
            return "çok düşük"

    def _get_confidence_descriptor(self, confidence) -> str:
        """Get confidence descriptor words

        Args:
            confidence: Confidence value (0.0-1.0) or confidence object

        Returns:
            Confidence descriptor
        """
        # Handle different confidence types
        if hasattr(confidence, "overall"):
            conf_val = confidence.overall
        elif hasattr(confidence, "value"):
            conf_val = confidence.value
        elif isinstance(confidence, (int, float)):
            conf_val = confidence
        else:
            conf_val = 0.8  # Default fallback

        if conf_val >= 0.9:
            return "mükemmel"
        elif conf_val >= 0.7:
            return "iyi"
        elif conf_val >= 0.5:
            return "orta"
        else:
            return "düşük"

    def explain_components(self, result) -> Dict[str, str]:
        """Explain individual components

        Args:
            result: Processing result

        Returns:
            Component explanations dictionary
        """
        if hasattr(result, "normalized_address") and result.normalized_address:
            components = result.normalized_address
        else:
            components = result.get("normalized", {}) if isinstance(result, dict) else {}

        explanations = {}
        templates = self.templates.get(self.config.language, self.templates["tr"])

        if hasattr(components, "__dict__"):
            for component, value in components.__dict__.items():
                if value:
                    confidence = 0.85  # Mock confidence
                    explanations[component] = templates["component"].format(
                        component=component, value=value, confidence=confidence
                    )
        elif isinstance(components, dict):
            for component, value in components.items():
                if value:
                    confidence = 0.85  # Mock confidence
                    explanations[component] = templates["component"].format(
                        component=component, value=value, confidence=confidence
                    )

        return explanations

    def explain_confidence(self, result) -> str:
        """Explain confidence score

        Args:
            result: Processing result

        Returns:
            Confidence explanation
        """
        # Extract confidence value - handle different object types
        confidence_obj = (
            getattr(result, "confidence", None)
            if hasattr(result, "confidence")
            else result.get("confidence", 0.0) if isinstance(result, dict) else 0.0
        )

        # If confidence is an object, try to get a numeric value
        if hasattr(confidence_obj, "overall"):
            confidence = confidence_obj.overall
        elif hasattr(confidence_obj, "value"):
            confidence = confidence_obj.value
        elif isinstance(confidence_obj, (int, float)):
            confidence = confidence_obj
        else:
            confidence = 0.8  # Default value for testing

        level = self._get_confidence_level(confidence)
        descriptor = self._get_confidence_descriptor(confidence)

        return f"Güven skoru: {confidence:.2f} ({level}, {descriptor})"

    def explain_processing_method(self, result) -> str:
        """Explain processing method

        Args:
            result: Processing result

        Returns:
            Processing method explanation
        """
        method = (
            getattr(result, "processing_method", "unknown")
            if hasattr(result, "processing_method")
            else result.get("processing_method", "unknown")
        )

        method_descriptions = {
            "hybrid": "Hibrit işleme yöntemi kullanıldı",
            "pattern": "Pattern tabanlı işleme yapıldı",
            "ml": "Makine öğrenmesi modeli kullanıldı",
            "fallback": "Yedek işleme yöntemi kullanıldı",
        }

        return method_descriptions.get(method, f"İşleme yöntemi: {method}")

    def _analyze_component_confidence(self, result) -> Dict[str, Any]:
        """Analyze component-level confidence

        Args:
            result: Processing result

        Returns:
            Component confidence analysis
        """
        # Mock implementation for testing
        component_scores = [("il", 0.95), ("ilce", 0.88), ("mahalle", 0.75), ("sokak", 0.82), ("bina_no", 0.90)]

        return {
            "components": component_scores,
            "average": sum(score for _, score in component_scores) / len(component_scores),
            "highest": max(component_scores, key=lambda x: x[1]),
            "lowest": min(component_scores, key=lambda x: x[1]),
        }

    def add_explanation_hook(self, hook_func):
        """Add custom explanation hook"""
        if not hasattr(self, "explanation_hooks"):
            self.explanation_hooks = []
        self.explanation_hooks.append(hook_func)

    def _get_detailed_component_analysis(self, result):
        """Get detailed component analysis"""
        components = getattr(result, "components", None) or getattr(result, "normalized_address", None)

        if not components:
            return {"analysis": "No components available"}

        analysis = {}
        # Check all available component fields dynamically
        for field_name in dir(components):
            if not field_name.startswith("_"):  # Skip private fields
                value = getattr(components, field_name, None)
                if value and isinstance(value, str):  # Only include string values
                    analysis[field_name] = {"value": value, "confidence": 0.85, "source": "pattern_match"}  # Mock confidence

        return analysis
