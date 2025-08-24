"""
Enhanced Output System Demo

Comprehensive demonstration of the enhanced output formatting system
with confidence scoring, quality metrics, and multiple output formats.
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from addrnorm.scoring.confidence import ConfidenceCalculator, ProcessingMethod
from addrnorm.scoring.quality import QualityAssessment
from addrnorm.output.enhanced_formatter import EnhancedFormatter, OutputFormat, format_address_result


def create_sample_normalization_result():
    """Create sample normalization result for testing"""
    return {
        "normalized_address": "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15",
        "extracted_fields": {
            "il": "İstanbul",
            "ilce": "Kadıköy",
            "mahalle": "Moda",
            "sokak": "Bahariye Caddesi",
            "bina_no": "15",
            "posta_kodu": "34710",
        },
    }


def create_sample_processing_context():
    """Create sample processing context"""
    return {
        "method_used": "pattern_primary",
        "pattern_match_score": 0.95,
        "pattern_quality": 0.88,
        "pattern_coverage": 0.92,
        "pattern_specificity": 0.85,
        "ml_model_confidence": 0.87,
        "prediction_entropy": 0.15,
        "feature_quality": 0.9,
        "training_similarity": 0.82,
        "processing_steps": [
            "Input validation",
            "Turkish address pattern detection",
            "Component extraction",
            "Geographic validation",
            "Quality assessment",
        ],
        "patterns_matched": ["turkish_standard_address"],
        "detected_components": ["il", "ilce", "mahalle", "sokak", "bina_no"],
    }


def demo_basic_enhanced_output():
    """Demo basic enhanced output formatting"""
    print("=" * 60)
    print("🎯 BASIC ENHANCED OUTPUT DEMO")
    print("=" * 60)

    # Create test data
    original_address = "Amorium Hotel karşısı"
    normalization_result = create_sample_normalization_result()
    processing_context = create_sample_processing_context()

    # Create formatter
    formatter = EnhancedFormatter(enable_legacy_compatibility=True, include_debug=True)

    print(f"📄 Original: {original_address}")
    print(f"🎯 Normalized: {normalization_result['normalized_address']}")
    print()

    # Format result
    enhanced_result = formatter.format_result(
        normalization_result, original_address, processing_context, OutputFormat.ENHANCED_JSON
    )

    print("✅ Enhanced Output:")
    print(json.dumps(enhanced_result, indent=2, ensure_ascii=False))
    print()

    return enhanced_result


def demo_confidence_and_quality():
    """Demo confidence scoring and quality assessment"""
    print("=" * 60)
    print("🔍 CONFIDENCE & QUALITY METRICS DEMO")
    print("=" * 60)

    # Create components
    confidence_calc = ConfidenceCalculator(pattern_weight=0.6, ml_weight=0.4, enable_detailed_breakdown=True)

    quality_assessor = QualityAssessment(enable_detailed_breakdown=True)

    # Test data
    normalization_result = create_sample_normalization_result()
    original_address = "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15"
    context = create_sample_processing_context()

    # Calculate confidence
    confidence = confidence_calc.calculate_confidence(normalization_result, context)
    print("📊 Confidence Scores:")
    print(f"  Pattern: {confidence.pattern:.3f}")
    print(f"  ML: {confidence.ml:.3f}")
    print(f"  Overall: {confidence.overall:.3f}")
    print(f"  Level: {confidence_calc.get_confidence_level(confidence.overall)}")
    print()

    # Assess quality
    quality = quality_assessor.assess_quality(normalization_result, original_address, context)
    print("🏆 Quality Metrics:")
    print(f"  Completeness: {quality.completeness:.3f}")
    print(f"  Consistency: {quality.consistency:.3f}")
    print(f"  Accuracy: {quality.accuracy:.3f}")
    print(f"  Usability: {quality.usability:.3f}")
    print(f"  Overall: {quality.overall_score:.3f}")
    print(f"  Level: {quality.quality_level.value}")
    print()

    if quality.quality_issues:
        print("⚠️ Quality Issues:")
        for issue in quality.quality_issues:
            print(f"  • {issue}")
        print()

    if quality.recommendations:
        print("💡 Recommendations:")
        for rec in quality.recommendations:
            print(f"  • {rec}")
        print()


def demo_multiple_output_formats():
    """Demo multiple output formats"""
    print("=" * 60)
    print("📋 MULTIPLE OUTPUT FORMATS DEMO")
    print("=" * 60)

    # Test data
    normalization_result = create_sample_normalization_result()
    original_address = "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15"
    context = create_sample_processing_context()

    formatter = EnhancedFormatter()

    # Test each format
    formats_to_test = [
        (OutputFormat.ENHANCED_JSON, "Enhanced JSON"),
        (OutputFormat.LEGACY_JSON, "Legacy JSON"),
        (OutputFormat.CSV, "CSV"),
        (OutputFormat.XML, "XML"),
    ]

    for output_format, format_name in formats_to_test:
        print(f"📄 {format_name} Format:")
        print("-" * 40)

        try:
            result = formatter.format_result(normalization_result, original_address, context, output_format)

            if isinstance(result, dict):
                if output_format == OutputFormat.CSV:
                    # Show CSV structure
                    print("CSV fields:")
                    for key, value in result.items():
                        print(f"  {key}: {value}")
                else:
                    print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(str(result))

        except Exception as e:
            print(f"❌ Error: {e}")

        print()


def demo_landmark_processing():
    """Demo landmark address processing"""
    print("=" * 60)
    print("🏛️ LANDMARK ADDRESS PROCESSING DEMO")
    print("=" * 60)

    # Landmark test cases
    landmark_cases = ["Amorium Hotel karşısı", "McDonald's yanı", "Migros arkası", "Metro istasyonu önü"]

    formatter = EnhancedFormatter()

    for original in landmark_cases:
        print(f"📍 Original: {original}")

        # Create context with landmark detection
        context = create_sample_processing_context()
        context["input_type"] = "landmark"

        # Mock normalization result
        result = {
            "normalized_address": "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15",
            "extracted_fields": {
                "il": "İstanbul",
                "ilce": "Kadıköy",
                "mahalle": "Moda",
                "sokak": "Bahariye Caddesi",
                "bina_no": "15",
            },
        }

        enhanced = formatter.format_result(result, original, context)

        print("🔍 Explanation:")
        explanation = enhanced["explanation"]
        print(f"  Raw: {explanation['raw']}")
        if "parsed" in explanation and "type" in explanation["parsed"]:
            parsed = explanation["parsed"]
            print(f"  Type: {parsed.get('type', 'unknown')}")
            if "name" in parsed:
                print(f"  Landmark: {parsed['name']}")
                print(f"  Relation: {parsed['relation']}")
        print()


def demo_batch_processing():
    """Demo batch processing and export"""
    print("=" * 60)
    print("📦 BATCH PROCESSING DEMO")
    print("=" * 60)

    # Create batch test data
    batch_data = [
        {
            "original": "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15",
            "result": create_sample_normalization_result(),
            "context": create_sample_processing_context(),
        },
        {
            "original": "Amorium Hotel karşısı",
            "result": {
                "normalized_address": "İstanbul Beşiktaş Etiler Mahallesi Nispetiye Caddesi No: 23",
                "extracted_fields": {
                    "il": "İstanbul",
                    "ilce": "Beşiktaş",
                    "mahalle": "Etiler",
                    "sokak": "Nispetiye Caddesi",
                    "bina_no": "23",
                },
            },
            "context": create_sample_processing_context(),
        },
        {
            "original": "Ankara Çankaya Kızılay Mahallesi Atatürk Bulvarı No: 89",
            "result": {
                "normalized_address": "Ankara Çankaya Kızılay Mahallesi Atatürk Bulvarı No: 89",
                "extracted_fields": {
                    "il": "Ankara",
                    "ilce": "Çankaya",
                    "mahalle": "Kızılay",
                    "sokak": "Atatürk Bulvarı",
                    "bina_no": "89",
                },
            },
            "context": create_sample_processing_context(),
        },
    ]

    formatter = EnhancedFormatter()

    print(f"📄 Processing {len(batch_data)} addresses...")

    # Test different batch formats
    formats_to_test = [
        (OutputFormat.ENHANCED_JSON, "enhanced_batch.json"),
        (OutputFormat.CSV, "enhanced_batch.csv"),
        (OutputFormat.XML, "enhanced_batch.xml"),
    ]

    output_dir = Path("enhanced_output_demo")
    output_dir.mkdir(exist_ok=True)

    for output_format, filename in formats_to_test:
        print(f"📋 Exporting {output_format.value} format...")

        try:
            # Process batch
            batch_results = formatter.format_batch(batch_data, output_format=output_format, output_file=output_dir / filename)

            print(f"  ✅ Exported to: {output_dir / filename}")

            # Show sample of results
            if isinstance(batch_results, list) and batch_results:
                sample = batch_results[0]
                if isinstance(sample, dict):
                    print(f"  📊 Sample result keys: {list(sample.keys())[:5]}...")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n📁 Output files saved in: {output_dir}")


def demo_schema_and_migration():
    """Demo schema export and migration guide"""
    print("=" * 60)
    print("📚 SCHEMA & MIGRATION DEMO")
    print("=" * 60)

    formatter = EnhancedFormatter()
    output_dir = Path("enhanced_output_demo")
    output_dir.mkdir(exist_ok=True)

    # Export schema
    print("📋 Exporting output schema...")
    schema_files = [("enhanced_schema.json", "json"), ("enhanced_schema.md", "markdown")]

    for filename, format_type in schema_files:
        try:
            formatter.export_schema(output_dir / filename, format_type)
            print(f"  ✅ Schema exported: {filename}")
        except Exception as e:
            print(f"  ❌ Error exporting {filename}: {e}")

    # Create migration guide
    print("\n📖 Creating migration guide...")
    try:
        formatter.create_migration_guide(output_dir / "migration_guide.md")
        print("  ✅ Migration guide created: migration_guide.md")
    except Exception as e:
        print(f"  ❌ Error creating migration guide: {e}")

    print(f"\n📁 Documentation files saved in: {output_dir}")


def demo_legacy_compatibility():
    """Demo legacy format compatibility"""
    print("=" * 60)
    print("🔄 LEGACY COMPATIBILITY DEMO")
    print("=" * 60)

    # Test data
    normalization_result = create_sample_normalization_result()
    original_address = "İstanbul Kadıköy Moda Mahallesi Bahariye Caddesi No: 15"
    context = create_sample_processing_context()

    formatter = EnhancedFormatter(enable_legacy_compatibility=True)

    # Enhanced format
    print("📋 Enhanced Format:")
    enhanced = formatter.format_result(normalization_result, original_address, context, OutputFormat.ENHANCED_JSON)
    print(
        json.dumps(
            {k: v for k, v in enhanced.items() if k in ["normalized_address", "confidence_scores", "extracted_fields"]},
            indent=2,
            ensure_ascii=False,
        )
    )
    print()

    # Legacy format
    print("📋 Legacy Format:")
    legacy = formatter.format_result(normalization_result, original_address, context, OutputFormat.LEGACY_JSON)
    print(json.dumps(legacy, indent=2, ensure_ascii=False))
    print()

    # Compatibility check
    print("🔍 Compatibility Analysis:")
    print(f"  Enhanced confidence: {enhanced['confidence_scores']['overall']}")
    print(f"  Legacy confidence: {legacy['confidence']}")
    print(
        f"  Fields match: {enhanced['extracted_fields'] == {k: v for k, v in legacy.items() if k not in ['normalized_address', 'confidence', 'method']}}"
    )
    print(f"  Method mapping: {enhanced['processing_method']} -> {legacy['method']}")


def demo_error_handling():
    """Demo error handling and fallback mechanisms"""
    print("=" * 60)
    print("🚨 ERROR HANDLING DEMO")
    print("=" * 60)

    formatter = EnhancedFormatter()

    # Test with malformed data
    error_cases = [
        {"name": "Empty result", "result": {}, "original": "Test address", "context": {}},
        {
            "name": "Missing fields",
            "result": {"normalized_address": "Test"},
            "original": "Test address",
            "context": {"method_used": "unknown"},
        },
        {
            "name": "Invalid confidence data",
            "result": {"normalized_address": "Test", "extracted_fields": {}},
            "original": "Test address",
            "context": {"pattern_match_score": "invalid", "ml_model_confidence": None},
        },
    ]

    for test_case in error_cases:
        print(f"🧪 Testing: {test_case['name']}")

        try:
            result = formatter.format_result(test_case["result"], test_case["original"], test_case["context"])

            print(f"  ✅ Handled gracefully")
            print(f"  📊 Overall confidence: {result['confidence_scores']['overall']}")
            print(f"  🎯 Validation status: {result['validation_status']}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        print()


def main():
    """Run all demos"""
    print("🚀 ENHANCED OUTPUT SYSTEM COMPREHENSIVE DEMO")
    print("=" * 80)
    print()

    try:
        # Core demos
        demo_basic_enhanced_output()
        demo_confidence_and_quality()
        demo_multiple_output_formats()
        demo_landmark_processing()

        # Advanced demos
        demo_batch_processing()
        demo_schema_and_migration()
        demo_legacy_compatibility()
        demo_error_handling()

        print("=" * 80)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("📁 Check the 'enhanced_output_demo' directory for exported files:")
        print("  • Schema documentation")
        print("  • Migration guide")
        print("  • Sample output files in multiple formats")
        print()

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
