"""
Advanced ML integration demo for Turkish address normalization.
Shows hybrid approach with adaptive thresholds.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.addrnorm.ml.models import HybridProcessor, ModelTrainer, ModelEvaluator, ProcessingMethod
import json


def demo_hybrid_processing():
    """Demonstrate hybrid processing with adaptive thresholds."""
    print("🚀 Turkish Address Normalization - Hybrid ML Demo")
    print("=" * 60)

    # Initialize hybrid processor
    processor = HybridProcessor()

    # Test cases with varying complexity
    test_cases = [
        {
            "text": "Çankaya Mahallesi Tunali Hilmi Caddesi No:15 Daire:3 Ankara",
            "pattern_result": {
                "confidence": 0.85,
                "components": {
                    "city": "Ankara",
                    "neighborhood": "Çankaya Mahallesi",
                    "street": "Tunali Hilmi Caddesi",
                    "number": "15",
                    "apartment": "3",
                },
            },
            "expected": {"city": "Ankara", "neighborhood": "Çankaya Mahallesi", "street": "Tunali Hilmi Caddesi"},
        },
        {
            "text": "Maltepe Istanbul yakınlarında bir yer",
            "pattern_result": {
                "confidence": 0.45,  # Low confidence - will trigger ML
                "components": {"city": "Istanbul", "district": "Maltepe"},
            },
            "expected": {"city": "Istanbul", "district": "Maltepe"},
        },
        {
            "text": "34000 Bahçelievler/İstanbul Merkez Efendi Mahallesi",
            "pattern_result": {
                "confidence": 0.75,
                "components": {
                    "city": "İstanbul",
                    "district": "Bahçelievler",
                    "neighborhood": "Merkez Efendi Mahallesi",
                    "postal_code": "34000",
                },
            },
            "expected": {"city": "İstanbul", "district": "Bahçelievler", "neighborhood": "Merkez Efendi Mahallesi"},
        },
    ]

    print("\n📊 Processing Test Cases:")
    print("-" * 40)

    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"Text: {case['text']}")
        print(f"Pattern Confidence: {case['pattern_result']['confidence']:.2f}")

        # Process with hybrid approach
        result = processor.process_hybrid(case["text"], case["pattern_result"])

        print(f"✅ Method Used: {result.confidence.method_used.value.upper()}")
        print(f"📈 Overall Confidence: {result.confidence.overall:.3f}")
        print(f"🎯 Adaptive Threshold: {result.confidence.threshold_used:.3f}")
        print(f"⏱️  Processing Time: {result.processing_time:.4f}s")

        if result.components:
            print("🏷️  Extracted Components:")
            for label, component in result.components.items():
                print(f"   {label}: {component.text} (conf: {component.confidence:.2f})")

        results.append(result)

        # Update performance tracking
        expected_components = set(case["expected"].keys())
        found_components = set(result.components.keys())
        success = len(expected_components & found_components) > 0

        processor.update_performance(success, result.confidence.method_used)

    return results


def demo_adaptive_thresholds():
    """Demonstrate adaptive threshold calculation."""
    print("\n\n🎛️  Adaptive Threshold Analysis")
    print("=" * 40)

    processor = HybridProcessor()

    # Simulate different pattern performance scenarios
    scenarios = [
        {
            "pattern_strength": 0.9,
            "context": {"has_postal_code": 1, "word_count": 5},
            "description": "High pattern strength + structured address",
        },
        {
            "pattern_strength": 0.6,
            "context": {"has_postal_code": 0, "word_count": 10},
            "description": "Medium pattern strength + complex address",
        },
        {
            "pattern_strength": 0.4,
            "context": {"has_postal_code": 0, "word_count": 15},
            "description": "Low pattern strength + very complex address",
        },
        {
            "pattern_strength": 0.8,
            "context": {"has_postal_code": 1, "word_count": 3},
            "description": "High pattern strength + simple address",
        },
    ]

    for scenario in scenarios:
        threshold = processor.threshold_calculator.calculate_threshold(scenario["pattern_strength"], scenario["context"])

        print(f"\n📐 Scenario: {scenario['description']}")
        print(f"   Pattern Strength: {scenario['pattern_strength']:.1f}")
        print(f"   Context Features: {scenario['context']}")
        print(f"   🎯 Calculated Threshold: {threshold:.3f}")

        # Determine method selection
        if scenario["pattern_strength"] >= threshold:
            method = "PATTERN"
        else:
            method = "ML"
        print(f"   🔧 Selected Method: {method}")


def demo_training_pipeline():
    """Demonstrate model training pipeline."""
    print("\n\n🏋️  Model Training Pipeline Demo")
    print("=" * 40)

    # Sample training data
    training_data = [
        {
            "text": "Kadıköy İstanbul Fenerbahçe Mahallesi",
            "components": {"city": "İstanbul", "district": "Kadıköy", "neighborhood": "Fenerbahçe Mahallesi"},
        },
        {
            "text": "Ankara Çankaya Kızılay Meydanı",
            "components": {"city": "Ankara", "district": "Çankaya", "neighborhood": "Kızılay"},
        },
    ]

    trainer = ModelTrainer()

    print(f"📚 Training Data: {len(training_data)} examples")

    # Prepare training data
    prepared_data = trainer.prepare_training_data(training_data)
    print(f"🔧 Prepared {len(prepared_data)} training examples")

    # Train model (simulation)
    model = trainer.train_sequence_model(prepared_data)

    # Evaluate model
    metrics = trainer.evaluate_model(model, prepared_data[:1])  # Using subset for demo

    print("\n📊 Model Performance:")
    for metric, value in metrics.items():
        print(f"   {metric.replace('_', ' ').title()}: {value:.3f}")


def demo_batch_evaluation():
    """Demonstrate batch evaluation."""
    print("\n\n📈 Batch Evaluation Demo")
    print("=" * 40)

    processor = HybridProcessor()
    evaluator = ModelEvaluator()

    # Extended test cases for evaluation
    test_cases = [
        {
            "text": "Beyoğlu İstanbul Galatasaray Mahallesi",
            "pattern_result": {"confidence": 0.8, "components": {"city": "İstanbul", "district": "Beyoğlu"}},
            "expected": {"city": "İstanbul", "district": "Beyoğlu"},
        },
        {
            "text": "Antalya merkez bölgesi yakınları",
            "pattern_result": {"confidence": 0.3, "components": {"city": "Antalya"}},
            "expected": {"city": "Antalya"},
        },
        {
            "text": "06100 Ankara Çankaya Kızılay",
            "pattern_result": {
                "confidence": 0.9,
                "components": {"city": "Ankara", "district": "Çankaya", "postal_code": "06100"},
            },
            "expected": {"city": "Ankara", "district": "Çankaya"},
        },
    ]

    # Run batch evaluation
    evaluation_results = evaluator.evaluate_batch(processor, test_cases)

    print(f"🎯 Evaluation Results ({evaluation_results['total_cases']} cases):")
    print(f"   Precision: {evaluation_results['precision']:.3f}")
    print(f"   Recall: {evaluation_results['recall']:.3f}")
    print(f"   F1-Score: {evaluation_results['f1_score']:.3f}")
    print(f"   Avg Processing Time: {evaluation_results['average_processing_time']:.4f}s")

    print(f"\n🔧 Method Distribution:")
    for method, count in evaluation_results["method_distribution"].items():
        percentage = (count / evaluation_results["total_cases"]) * 100
        print(f"   {method.upper()}: {count} cases ({percentage:.1f}%)")


def main():
    """Run all demos."""
    try:
        # Main hybrid processing demo
        results = demo_hybrid_processing()

        # Adaptive threshold analysis
        demo_adaptive_thresholds()

        # Training pipeline demo
        demo_training_pipeline()

        # Batch evaluation demo
        demo_batch_evaluation()

        print(f"\n\n🎉 Demo completed successfully!")
        print(f"✨ Processed {len(results)} test cases with hybrid ML approach")
        print("\n📝 Key Features Demonstrated:")
        print("   ✅ Adaptive threshold calculation")
        print("   ✅ Pattern vs ML method selection")
        print("   ✅ Performance tracking and adaptation")
        print("   ✅ Comprehensive evaluation metrics")
        print("   ✅ Training pipeline integration")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
