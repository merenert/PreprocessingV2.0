"""
Explanation templates module
"""

from typing import Dict, Any


class ExplanationTemplates:
    """Template manager for address explanations"""

    def __init__(self):
        """Initialize templates"""
        self.templates = {
            "tr": {
                "success": "Adres başarıyla normalleştirildi.",
                "component": "{component}: {value} (güven: {confidence:.2f})",
                "warning": "Uyarı: {message}",
                "error": "Hata: {message}",
                "low_confidence": "Düşük güven skoru",
                "high_confidence": "Yüksek güven ile normalleştirildi",
            },
            "en": {
                "success": "Address successfully normalized.",
                "component": "{component}: {value} (confidence: {confidence:.2f})",
                "warning": "Warning: {message}",
                "error": "Error: {message}",
                "low_confidence": "Low confidence score",
                "high_confidence": "High confidence normalization",
            },
        }

    def get_template(self, template_name: str, language: str, detail_level: str = "basic") -> str:
        """Get template string

        Args:
            template_name: Template name
            language: Language code
            detail_level: Detail level (basic, detailed, verbose)

        Returns:
            Template string
        """
        lang_templates = self.templates.get(language, self.templates["tr"])

        # Include detail level in template key if needed
        template_key = f"{template_name}_{detail_level}" if detail_level != "basic" else template_name
        template = lang_templates.get(template_key, lang_templates.get(template_name, f"Varsayılan {template_name} mesajı"))

        return template

    def format_template(self, template: str, **kwargs) -> str:
        """Format template with parameters

        Args:
            template: Template string
            **kwargs: Template parameters

        Returns:
            Formatted string
        """
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError):
            return template

    def validate_template(self, template: str) -> bool:
        """Validate template syntax

        Args:
            template: Template string to validate

        Returns:
            True if template is valid
        """
        try:
            # Check for valid string format syntax - allow unknown variables
            import string

            formatter = string.Formatter()
            parsed = list(formatter.parse(template))
            return True
        except (ValueError, AttributeError):
            return False
