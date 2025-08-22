"""
Comprehensive end-to-end acceptance test for Turkish address normalization pipeline.
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from addrnorm.pipeline import PipelineConfig, create_pipeline


def test_acceptance_criteria():
    """Test acceptance criteria for the complete pipeline."""

    print("TURKISH ADDRESS NORMALIZATION PIPELINE")
    print("COMPREHENSIVE ACCEPTANCE TEST")
    print("=" * 60)

    # Test configuration
    config = PipelineConfig(
        enable_multiprocessing=False,
        enable_validation=True,
        ml_model_path="models/turkish_address_ner_improved",
        random_seed=42,
    )

    # Create pipeline
    pipeline = create_pipeline(config)

    # 10 diverse test addresses covering different patterns and complexities
    test_cases = [
        {
            "input": "Atatürk Mahallesi Cumhuriyet Caddesi No:15 Çankaya/Ankara",
            "expected_components": [
                "neighborhood",
                "street",
                "number",
                "district",
                "city",
            ],
            "min_confidence": 0.7,
        },
        {
            "input": "Beşiktaş Akaretler Sıraselviler Caddesi 10/3 İstanbul",
            "expected_components": ["street", "number", "city"],
            "min_confidence": 0.6,
        },
        {
            "input": "Kızılay Meydanı Çankaya Ankara",
            "expected_components": ["district", "city"],
            "min_confidence": 0.5,
        },
        {
            "input": "İzmir Konak Alsancak Mahallesi",
            "expected_components": ["city", "district", "neighborhood"],
            "min_confidence": 0.6,
        },
        {
            "input": "Antalya Muratpaşa Fener Mahallesi Dumlupınar Bulvarı",
            "expected_components": ["city", "district", "neighborhood", "street"],
            "min_confidence": 0.6,
        },
        {
            "input": "Kadıköy Moda Caddesi 23/5 İstanbul",
            "expected_components": ["district", "street", "number", "city"],
            "min_confidence": 0.7,
        },
        {
            "input": "Trabzon Ortahisar Güzelcehisar Mahallesi Atatürk Alanı Cd. No:10",
            "expected_components": [
                "city",
                "district",
                "neighborhood",
                "street",
                "number",
            ],
            "min_confidence": 0.7,
        },
        {
            "input": "Eskişehir Tepebaşı Arifiye Mahallesi Eski Bağlar Caddesi 45",
            "expected_components": [
                "city",
                "district",
                "neighborhood",
                "street",
                "number",
            ],
            "min_confidence": 0.7,
        },
        {
            "input": "Bursa Nilüfer Görükle Mahallesi Uludağ Üniversitesi Kampüsü",
            "expected_components": ["city", "district", "neighborhood"],
            "min_confidence": 0.6,
        },
        {
            "input": "Gaziantep Şahinbey 100. Yıl Mahallesi Adnan Menderes Bulvarı 200",
            "expected_components": [
                "city",
                "district",
                "neighborhood",
                "street",
                "number",
            ],
            "min_confidence": 0.7,
        },
    ]

    print(f"Testing {len(test_cases)} diverse address patterns...")
    print()

    results = []
    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"{i:2d}. Testing: {test_case['input']}")

        # Process address
        result = pipeline.process_single(test_case["input"])

        if not result.success:
            print(f"    FAILED: {result.error}")
            failed += 1
            continue

        addr = result.address_out

        # Check confidence
        confidence_ok = result.confidence >= test_case["min_confidence"]

        # Check components
        found_components = []
        if addr.city:
            found_components.append("city")
        if addr.district:
            found_components.append("district")
        if addr.neighborhood:
            found_components.append("neighborhood")
        if addr.street:
            found_components.append("street")
        if addr.number:
            found_components.append("number")
        if addr.building:
            found_components.append("building")
        if addr.floor:
            found_components.append("floor")
        if addr.apartment:
            found_components.append("apartment")

        # Check if we found the expected key components
        expected_found = sum(
            1 for comp in test_case["expected_components"] if comp in found_components
        )
        coverage = expected_found / len(test_case["expected_components"])

        # Test passes if confidence meets threshold and we found most components
        test_passed = confidence_ok and coverage >= 0.6  # 60% component coverage

        if test_passed:
            print("    PASSED")
            passed += 1
        else:
            print("    FAILED")
            failed += 1

        print(f"       Method: {result.processing_method}")
        print(
            f"       Confidence: {result.confidence:.3f} "
            f"({'OK' if confidence_ok else 'LOW'} >= "
            f"{test_case['min_confidence']})"
        )
        print(
            f"       Components: {len(found_components)} found "
            f"({coverage:.1%} coverage)"
        )
        print(f"       Time: {result.processing_time_ms:.1f}ms")

        # Show key extracted components
        key_parts = []
        if addr.city:
            key_parts.append(f"City: {addr.city}")
        if addr.district:
            key_parts.append(f"District: {addr.district}")
        if addr.neighborhood:
            key_parts.append(f"Neighborhood: {addr.neighborhood}")
        if addr.street:
            key_parts.append(f"Street: {addr.street}")
        if addr.number:
            key_parts.append(f"Number: {addr.number}")

        if key_parts:
            print(f"       Extracted: {', '.join(key_parts)}")

        print()

        # Store result for summary
        results.append(
            {
                "input": test_case["input"],
                "success": test_passed,
                "confidence": result.confidence,
                "method": result.processing_method,
                "components": len(found_components),
                "time_ms": result.processing_time_ms,
            }
        )

    # Summary
    print("=" * 60)
    print("ACCEPTANCE TEST RESULTS")
    print("=" * 60)

    success_rate = passed / len(test_cases)
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    avg_time = sum(r["time_ms"] for r in results) / len(results)

    print(f"Passed: {passed}/{len(test_cases)} ({success_rate:.1%})")
    print(f"Failed: {failed}/{len(test_cases)}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Average Processing Time: {avg_time:.1f}ms")
    print()

    # Method distribution
    methods = {}
    for r in results:
        methods[r["method"]] = methods.get(r["method"], 0) + 1

    print("Processing Methods Used:")
    for method, count in methods.items():
        print(f"   {method}: {count} addresses ({count/len(results):.1%})")
    print()

    # Performance criteria
    print("Performance Criteria:")
    print(
        f"   Success Rate: {success_rate:.1%} "
        f"{'PASS' if success_rate >= 0.8 else 'FAIL'} (>= 80%)"
    )
    print(
        f"   Avg Confidence: {avg_confidence:.3f} "
        f"{'PASS' if avg_confidence >= 0.6 else 'FAIL'} (>= 0.6)"
    )
    print(
        f"   Avg Time: {avg_time:.1f}ms "
        f"{'PASS' if avg_time <= 50 else 'FAIL'} (<= 50ms)"
    )
    print()

    # Overall result
    criteria_met = success_rate >= 0.8 and avg_confidence >= 0.6 and avg_time <= 50

    if criteria_met:
        print("ACCEPTANCE TEST PASSED!")
        print("   Pipeline meets all acceptance criteria.")
    else:
        print("ACCEPTANCE TEST FAILED!")
        print("   Pipeline does not meet all acceptance criteria.")

    print()
    print("=" * 60)
    print("End-to-end Turkish address normalization pipeline test complete.")
    print("=" * 60)

    return criteria_met


if __name__ == "__main__":
    test_acceptance_criteria()
