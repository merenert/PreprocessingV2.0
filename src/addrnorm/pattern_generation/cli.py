"""
ML Pattern Generation CLI

Command-line interface for pattern suggestion, review, and conflict analysis.
"""

import click
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .ml_suggester import MLPatternSuggester
from .conflict_detector import PatternConflictDetector
from .pattern_analyzer import PatternAnalyzer
from .review_interface import PatternReviewInterface, ReviewCLI
from .models import PatternGenerationConfig, ResolutionStrategy


class MLPatternCLI:
    """
    Main CLI interface for ML pattern generation system
    """

    def __init__(self):
        self.config = PatternGenerationConfig()
        self.suggester = MLPatternSuggester(self.config)
        self.conflict_detector = PatternConflictDetector()
        self.pattern_analyzer = PatternAnalyzer()
        self.review_interface = PatternReviewInterface()
        self.review_cli = ReviewCLI(self.review_interface)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def cli(ctx):
    """ML Pattern Generation System CLI"""
    ctx.ensure_object(dict)
    ctx.obj["cli"] = MLPatternCLI()


@cli.command()
@click.option("--input-file", "-i", required=True, help="Input file containing address examples")
@click.option("--output-file", "-o", default="pattern_suggestions.json", help="Output file for suggestions")
@click.option("--max-patterns", "-m", default=10, help="Maximum number of patterns to generate")
@click.option("--min-confidence", "-c", default=0.6, type=float, help="Minimum confidence threshold")
@click.pass_context
def suggest(ctx, input_file, output_file, max_patterns, min_confidence):
    """Generate pattern suggestions from address examples"""

    cli_obj = ctx.obj["cli"]

    # Load addresses
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            if input_file.endswith(".json"):
                data = json.load(f)
                if isinstance(data, list):
                    addresses = data
                else:
                    addresses = data.get("addresses", [])
            else:
                addresses = [line.strip() for line in f if line.strip()]

        click.echo(f"Loaded {len(addresses)} addresses from {input_file}")

    except Exception as e:
        click.echo(f"Error loading addresses: {e}", err=True)
        return

    # Update config
    cli_obj.config.min_pattern_confidence = min_confidence

    # Generate suggestions
    click.echo("Generating pattern suggestions...")

    try:
        suggestions = cli_obj.suggester.suggest_patterns(addresses, max_patterns)

        click.echo(f"Generated {len(suggestions)} pattern suggestions")

        # Serialize suggestions
        serialized_suggestions = []
        for suggestion in suggestions:
            serialized_suggestions.append(
                {
                    "pattern_id": suggestion.pattern_id,
                    "regex_pattern": suggestion.regex_pattern,
                    "description": suggestion.description,
                    "confidence_score": suggestion.confidence_score,
                    "expected_fields": suggestion.expected_fields,
                    "sample_matches": suggestion.sample_matches,
                    "quality_metrics": suggestion.quality_metrics,
                    "generated_at": suggestion.generated_at.isoformat(),
                }
            )

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "suggestions": serialized_suggestions,
                    "generation_config": {
                        "max_patterns": max_patterns,
                        "min_confidence": min_confidence,
                        "source_file": input_file,
                        "address_count": len(addresses),
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        click.echo(f"Suggestions saved to {output_file}")

        # Show summary
        if suggestions:
            avg_confidence = sum(s.confidence_score for s in suggestions) / len(suggestions)
            high_confidence = sum(1 for s in suggestions if s.confidence_score > 0.8)

            click.echo(f"\n📊 Summary:")
            click.echo(f"  Average confidence: {avg_confidence:.3f}")
            click.echo(f"  High confidence patterns: {high_confidence}")
            click.echo(f"  Pattern types: {len(set(s.description.split()[0] for s in suggestions))}")

    except Exception as e:
        click.echo(f"Error generating suggestions: {e}", err=True)


@cli.command()
@click.option("--existing-patterns", "-e", required=True, help="JSON file with existing patterns")
@click.option("--new-patterns", "-n", required=True, help="JSON file with new pattern suggestions")
@click.option("--output-file", "-o", default="conflict_analysis.json", help="Output file for conflict analysis")
@click.pass_context
def analyze_conflicts(ctx, existing_patterns, new_patterns, output_file):
    """Analyze conflicts between existing and new patterns"""

    cli_obj = ctx.obj["cli"]

    try:
        # Load existing patterns
        with open(existing_patterns, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_list = existing_data
            else:
                existing_list = existing_data.get("patterns", [])

        # Load new patterns
        with open(new_patterns, "r", encoding="utf-8") as f:
            new_data = json.load(f)
            if isinstance(new_data, list):
                new_list = new_data
            else:
                new_list = new_data.get("suggestions", [])

        click.echo(f"Loaded {len(existing_list)} existing patterns and {len(new_list)} new patterns")

        # Convert new patterns to PatternSuggestion objects (simplified)
        from datetime import datetime
        from .models import PatternSuggestion

        pattern_suggestions = []
        for item in new_list:
            suggestion = PatternSuggestion(
                pattern_id=item.get("pattern_id", f"pattern_{len(pattern_suggestions)}"),
                regex_pattern=item.get("regex_pattern", ""),
                description=item.get("description", ""),
                source_cluster=None,
                confidence_score=item.get("confidence_score", 0.5),
                expected_fields=item.get("expected_fields", []),
                sample_matches=item.get("sample_matches", []),
                quality_metrics=item.get("quality_metrics", {}),
                generated_at=datetime.now(),
            )
            pattern_suggestions.append(suggestion)

        # Detect conflicts
        click.echo("Analyzing conflicts...")
        conflicts = cli_obj.conflict_detector.detect_conflicts(existing_list, pattern_suggestions)

        click.echo(f"Detected {len(conflicts)} conflicts")

        # Serialize conflicts
        serialized_conflicts = []
        for conflict in conflicts:
            serialized_conflicts.append(
                {
                    "conflict_id": conflict.conflict_id,
                    "existing_pattern_id": conflict.existing_pattern_id,
                    "new_pattern_id": conflict.new_pattern_id,
                    "conflict_type": conflict.conflict_type.value,
                    "severity": conflict.severity.value,
                    "description": conflict.description,
                    "affected_samples": conflict.affected_samples,
                    "resolution_suggestions": [
                        {
                            "strategy": res.strategy.value,
                            "description": res.description,
                            "confidence_score": res.confidence_score,
                            "implementation_notes": res.implementation_notes,
                            "impact_assessment": res.impact_assessment,
                        }
                        for res in conflict.resolution_suggestions
                    ],
                    "conflict_details": conflict.conflict_details,
                    "detected_at": conflict.detected_at.isoformat(),
                }
            )

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "conflicts": serialized_conflicts,
                    "analysis_summary": {
                        "existing_patterns": len(existing_list),
                        "new_patterns": len(new_list),
                        "total_conflicts": len(conflicts),
                        "high_severity_conflicts": sum(1 for c in conflicts if c.severity.value in ["HIGH", "CRITICAL"]),
                        "conflict_types": list(set(c.conflict_type.value for c in conflicts)),
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        click.echo(f"Conflict analysis saved to {output_file}")

        # Show summary
        if conflicts:
            severity_counts = {}
            for conflict in conflicts:
                severity = conflict.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            click.echo(f"\n⚠️  Conflict Summary:")
            for severity, count in severity_counts.items():
                click.echo(f"  {severity}: {count}")

    except Exception as e:
        click.echo(f"Error analyzing conflicts: {e}", err=True)


@cli.command()
@click.option("--suggestions-file", "-s", required=True, help="JSON file with pattern suggestions")
@click.option("--conflicts-file", "-c", help="JSON file with conflicts (optional)")
@click.option("--reviewer-id", "-r", default="cli_user", help="Reviewer identifier")
@click.option(
    "--priority", "-p", default="normal", type=click.Choice(["low", "normal", "high", "critical"]), help="Review priority"
)
@click.pass_context
def submit_review(ctx, suggestions_file, conflicts_file, reviewer_id, priority):
    """Submit patterns for human review"""

    cli_obj = ctx.obj["cli"]

    try:
        # Load suggestions
        with open(suggestions_file, "r", encoding="utf-8") as f:
            suggestions_data = json.load(f)
            if isinstance(suggestions_data, list):
                suggestions_list = suggestions_data
            else:
                suggestions_list = suggestions_data.get("suggestions", [])

        # Convert to PatternSuggestion objects
        from datetime import datetime
        from .models import PatternSuggestion

        pattern_suggestions = []
        for item in suggestions_list:
            suggestion = PatternSuggestion(
                pattern_id=item.get("pattern_id", f"pattern_{len(pattern_suggestions)}"),
                regex_pattern=item.get("regex_pattern", ""),
                description=item.get("description", ""),
                source_cluster=None,
                confidence_score=item.get("confidence_score", 0.5),
                expected_fields=item.get("expected_fields", []),
                sample_matches=item.get("sample_matches", []),
                quality_metrics=item.get("quality_metrics", {}),
                generated_at=datetime.now(),
            )
            pattern_suggestions.append(suggestion)

        # Load conflicts if provided
        conflicts = []
        if conflicts_file:
            with open(conflicts_file, "r", encoding="utf-8") as f:
                conflicts_data = json.load(f)
                # Simplified conflict loading - would need full deserialization in practice
                conflicts = conflicts_data.get("conflicts", [])

        # Submit for review
        review_id = cli_obj.review_interface.submit_for_review(
            pattern_suggestions, conflicts, reviewer_id, priority  # This would need proper PatternConflict objects
        )

        click.echo(f"✅ Submitted review session: {review_id}")
        click.echo(f"   Patterns: {len(pattern_suggestions)}")
        click.echo(f"   Conflicts: {len(conflicts)}")
        click.echo(f"   Priority: {priority}")

    except Exception as e:
        click.echo(f"Error submitting review: {e}", err=True)


@cli.command()
@click.option("--reviewer-id", "-r", help="Filter by reviewer ID")
@click.pass_context
def list_reviews(ctx, reviewer_id):
    """List pending review sessions"""

    cli_obj = ctx.obj["cli"]
    cli_obj.review_cli.show_pending_reviews()


@cli.command()
@click.argument("review_id")
@click.pass_context
def show_review(ctx, review_id):
    """Show detailed review session information"""

    cli_obj = ctx.obj["cli"]
    cli_obj.review_cli.show_review_session(review_id)


@cli.command()
@click.pass_context
def review_stats(ctx):
    """Show review statistics"""

    cli_obj = ctx.obj["cli"]
    cli_obj.review_cli.show_statistics()


@cli.command()
@click.option("--pattern-file", "-p", required=True, help="JSON file with patterns to analyze")
@click.option("--output-file", "-o", default="pattern_analysis.json", help="Output file for analysis")
@click.pass_context
def analyze_patterns(ctx, pattern_file, output_file):
    """Analyze pattern quality and characteristics"""

    cli_obj = ctx.obj["cli"]

    try:
        # Load patterns
        with open(pattern_file, "r", encoding="utf-8") as f:
            pattern_data = json.load(f)
            if isinstance(pattern_data, list):
                patterns = pattern_data
            else:
                patterns = pattern_data.get("patterns", pattern_data.get("suggestions", []))

        click.echo(f"Analyzing {len(patterns)} patterns...")

        # Analyze each pattern
        analysis_results = []

        for pattern in patterns:
            pattern_info = {
                "pattern_id": pattern.get("pattern_id", "unknown"),
                "regex_pattern": pattern.get("regex_pattern", ""),
                "confidence_score": pattern.get("confidence_score", 0.0),
            }

            try:
                quality_analysis = cli_obj.pattern_analyzer.analyze_pattern_quality(pattern_info)

                analysis_results.append(
                    {
                        "pattern_id": pattern_info["pattern_id"],
                        "quality_analysis": quality_analysis,
                        "original_pattern": pattern_info,
                    }
                )

            except Exception as e:
                click.echo(f"Error analyzing pattern {pattern_info['pattern_id']}: {e}")
                continue

        # Save analysis results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "analysis_results": analysis_results,
                    "summary": {
                        "total_patterns": len(patterns),
                        "analyzed_patterns": len(analysis_results),
                        "analysis_timestamp": datetime.now().isoformat(),
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        click.echo(f"Pattern analysis saved to {output_file}")

        # Show summary
        if analysis_results:
            avg_quality = sum(result["quality_analysis"]["overall_quality_score"] for result in analysis_results) / len(
                analysis_results
            )

            high_quality = sum(1 for result in analysis_results if result["quality_analysis"]["overall_quality_score"] > 0.8)

            click.echo(f"\n📊 Analysis Summary:")
            click.echo(f"  Average quality score: {avg_quality:.3f}")
            click.echo(f"  High quality patterns: {high_quality}")

    except Exception as e:
        click.echo(f"Error analyzing patterns: {e}", err=True)


@cli.command()
@click.option("--config-file", "-c", help="Configuration file path")
@click.pass_context
def show_config(ctx, config_file):
    """Show current configuration"""

    cli_obj = ctx.obj["cli"]

    if config_file:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            click.echo(f"Configuration from {config_file}:")
            click.echo(json.dumps(config_data, indent=2))
        except Exception as e:
            click.echo(f"Error loading config file: {e}", err=True)
    else:
        click.echo("Current Configuration:")
        click.echo(f"  Min Pattern Confidence: {cli_obj.config.min_pattern_confidence}")
        click.echo(f"  Max Clusters: {cli_obj.config.max_clusters}")
        click.echo(f"  Min Cluster Size: {cli_obj.config.min_cluster_size}")


if __name__ == "__main__":
    cli()
