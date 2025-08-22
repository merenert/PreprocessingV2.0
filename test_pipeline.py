"""
Basic test for the end-to-end pipeline.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from addrnorm.pipeline import PipelineConfig, create_pipeline


def test_pipeline_basic():
    """Test basic pipeline functionality."""

    # Create pipeline with test configuration
    config = PipelineConfig(
        enable_multiprocessing=False,  # Disable for testing
        enable_validation=False,  # Disable if no geo data available
        ml_model_path="test_model_path",  # Will fail to load, that's OK
    )

    pipeline = create_pipeline(config)

    # Test addresses
    test_addresses = [
        "Atatürk Mahallesi Cumhuriyet Caddesi No:15 Çankaya/Ankara",
        "Beşiktaş Akaretler Sıraselviler Caddesi 10/3 İstanbul",
        "Kızılay Meydanı Çankaya Ankara",
        "İzmir Konak Alsancak Mahallesi",
        "Antalya Muratpaşa Fener Mahallesi Dumlupınar Bulvarı",
    ]

    print("Testing single address processing:")
    for i, addr in enumerate(test_addresses[:3]):  # Test first 3
        print(f"\n{i+1}. Input: {addr}")

        result = pipeline.process_single(addr)

        if result.success:
            print(f"   Method: {result.processing_method}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Time: {result.processing_time_ms:.1f}ms")
            print(f"   Normalized: {result.address_out.normalized_address}")

            # Show key components
            components = []
            if result.address_out.city:
                components.append(f"City: {result.address_out.city}")
            if result.address_out.district:
                components.append(f"District: {result.address_out.district}")
            if result.address_out.neighborhood:
                components.append(f"Neighborhood: {result.address_out.neighborhood}")
            if result.address_out.street:
                components.append(f"Street: {result.address_out.street}")

            if components:
                print(f"   Components: {', '.join(components)}")
        else:
            print(f"   Error: {result.error}")

    print("\n" + "=" * 60)
    print("Testing batch processing:")

    # Test batch processing
    batch_results = pipeline.process_batch(test_addresses)

    successful = sum(1 for r in batch_results if r.success)
    total_time = sum(r.processing_time_ms for r in batch_results)

    print(f"Processed {len(batch_results)} addresses")
    print(f"Successful: {successful}/{len(batch_results)}")
    print(f"Total time: {total_time:.1f}ms")
    print(f"Average time: {total_time/len(batch_results):.1f}ms per address")

    # Show method distribution
    methods = {}
    for result in batch_results:
        methods[result.processing_method] = methods.get(result.processing_method, 0) + 1

    print("Methods used:")
    for method, count in methods.items():
        print(f"  {method}: {count}")

    print("\n" + "=" * 60)
    print("Testing file processing:")

    # Test file processing
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for addr in test_addresses:
            f.write(addr + "\n")
        input_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        output_file = f.name

    try:
        stats = pipeline.process_file(input_file, output_file)

        print("File processing stats:")
        print(f"  Total: {stats['total_processed']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Average confidence: {stats['average_confidence']:.3f}")
        print(f"  Average time: {stats['average_time_ms']:.1f}ms")

        # Read and show first few results
        print(f"\nFirst few results from {output_file}:")
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:2]):
                print(f"  {i+1}. {line.strip()[:100]}...")

    finally:
        # Clean up temp files
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("Pipeline test completed successfully!")


if __name__ == "__main__":
    test_pipeline_basic()
