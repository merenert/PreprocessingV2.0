#!/usr/bin/env python3
"""
Enhanced Turkish Address Normalization CLI Tool

A modern command-line interface with rich formatting, interactive mode,
and comprehensive functionality for normalizing Turkish addresses.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Rich imports for enhanced UI
try:
    import click
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich.syntax import Syntax
    from rich import box
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import InMemoryHistory

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    click = None

# Legacy argparse for fallback
import argparse

# Try to import from installed package first
try:
    from addrnorm.pipeline import PipelineConfig, create_pipeline
    from addrnorm.explanation.explainer import AddressExplainer, ExplanationConfig
    from addrnorm.monitoring.metrics_collector import MetricsCollector
    from addrnorm.pattern_generation.system import PatternGenerationSystem
except ImportError:
    # Fallback for development
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent
    sys.path.insert(0, str(src_dir))
    from addrnorm.pipeline import PipelineConfig, create_pipeline
    from addrnorm.explanation.explainer import AddressExplainer, ExplanationConfig
    from addrnorm.monitoring.metrics_collector import MetricsCollector
    from addrnorm.pattern_generation.system import PatternGenerationSystem

# Initialize console (fallback to print if rich not available)
if RICH_AVAILABLE:
    console = Console()
else:

    class FallbackConsole:
        def print(self, *args, **kwargs):
            print(*args)

    console = FallbackConsole()


class AddrnormCLI:
    """Enhanced CLI for Turkish Address Normalization"""

    def __init__(self):
        self.console = console
        self.pipeline = None
        self.explainer = None
        self.metrics_collector = MetricsCollector() if RICH_AVAILABLE else None
        self.pattern_system = PatternGenerationSystem() if RICH_AVAILABLE else None

    def initialize_pipeline(self, config: Optional[Dict] = None):
        """Initialize the normalization pipeline"""
        if not self.pipeline:
            pipeline_config = PipelineConfig(**(config or {}))
            self.pipeline = create_pipeline(pipeline_config)
            if RICH_AVAILABLE:
                self.explainer = AddressExplainer(ExplanationConfig(language="tr"))

    def display_banner(self):
        """Display the application banner"""
        if RICH_AVAILABLE:
            banner = Panel.fit(
                Text.from_markup(
                    """
[bold blue]🏠 ADDRNORM - Turkish Address Normalization[/bold blue]
[dim]Advanced CLI for processing Turkish addresses[/dim]

[yellow]✨ Features:[/yellow]
• Interactive processing mode
• Rich visual feedback
• Pattern management
• Real-time monitoring
• Analytics & reporting
                """
                ),
                border_style="blue",
                padding=(1, 2),
            )
            self.console.print(banner)
        else:
            print("🏠 ADDRNORM - Turkish Address Normalization")
            print("Advanced CLI for processing Turkish addresses")


# Enhanced CLI with Rich (if available)
if RICH_AVAILABLE and click:

    @click.group(invoke_without_command=True)
    @click.pass_context
    def cli(ctx):
        """🏠 Turkish Address Normalization - Enhanced CLI Tool"""
        if ctx.invoked_subcommand is None:
            # If no subcommand, show interactive mode
            cli_app = AddrnormCLI()
            cli_app.display_banner()
            interactive_mode()

    @cli.command()
    @click.option("--input", "-i", type=click.Path(exists=True), help="Input file path")
    @click.option("--output", "-o", type=click.Path(), help="Output file path")
    @click.option("--format", "-f", type=click.Choice(["json", "csv", "xlsx"]), default="json", help="Output format")
    @click.option("--batch-size", "-b", type=int, default=100, help="Batch processing size")
    @click.option("--jobs", "-j", type=int, default=1, help="Number of parallel jobs")
    @click.option("--threshold", "-t", type=float, default=0.7, help="Confidence threshold")
    @click.option("--address", "-a", type=str, help="Single address to normalize")
    @click.option("--explain", is_flag=True, help="Include explanations in output")
    @click.option("--verbose", "-v", is_flag=True, help="Verbose output")
    @click.option("--progress", is_flag=True, default=True, help="Show progress bar")
    def normalize(input, output, format, batch_size, jobs, threshold, address, explain, verbose, progress):
        """🔄 Normalize Turkish addresses with enhanced processing"""

        cli_app = AddrnormCLI()
        cli_app.initialize_pipeline({"confidence_threshold": threshold})

        if address:
            # Single address mode
            _process_single_address(cli_app, address, explain, verbose)
        elif input:
            # File processing mode
            _process_file(cli_app, input, output, format, batch_size, jobs, explain, verbose, progress)
        else:
            console.print("[red]Error:[/red] Either --input or --address is required")
            raise click.Abort()

    @cli.command()
    @click.option("--address", "-a", type=str, required=True, help="Address to explain")
    @click.option(
        "--detail", "-d", type=click.Choice(["basic", "detailed", "verbose"]), default="detailed", help="Detail level"
    )
    @click.option("--language", "-l", type=click.Choice(["tr", "en"]), default="tr", help="Output language")
    def explain(address, detail, language):
        """💡 Generate detailed explanations for address normalization"""

        cli_app = AddrnormCLI()
        cli_app.initialize_pipeline()

        with console.status("[blue]Processing address..."):
            # Process the address
            result = cli_app.pipeline.process(address)

            # Generate explanation
            explainer = AddressExplainer(ExplanationConfig(detail_level=detail, language=language))
            explanation = explainer.explain(result)

        # Display results
        _display_explanation_result(result, explanation, detail)

    @cli.command()
    def interactive():
        """🎮 Interactive address normalization mode"""
        interactive_mode()

    def interactive_mode():
        """Enhanced interactive mode with rich UI"""

        cli_app = AddrnormCLI()
        cli_app.initialize_pipeline()

        # Create autocomplete history
        history = InMemoryHistory()

        # Common Turkish address components for autocomplete
        address_completer = WordCompleter(
            [
                "mahalle",
                "mahallesi",
                "mah",
                "sokak",
                "sokağı",
                "sok",
                "cadde",
                "caddesi",
                "cad",
                "bulvar",
                "bulvarı",
                "blv",
                "apartman",
                "apart",
                "apt",
                "site",
                "plaza",
                "ankara",
                "istanbul",
                "izmir",
                "antalya",
                "bursa",
                "adana",
                "gaziantep",
            ]
        )

        console.clear()
        cli_app.display_banner()

        # Statistics tracking
        stats = {"processed": 0, "successful": 0, "avg_confidence": 0.0, "start_time": time.time()}

        console.print("\n[bold green]🎮 Interactive Mode Started[/bold green]")
        console.print("[dim]Type 'help' for commands, 'quit' to exit[/dim]\n")

        while True:
            try:
                # Get user input with autocomplete
                user_input = prompt(
                    "📍 Enter address: ",
                    history=history,
                    completer=address_completer,
                ).strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ["quit", "q", "exit"]:
                    _display_session_stats(stats)
                    console.print("[yellow]👋 Goodbye![/yellow]")
                    break
                elif user_input.lower() == "help":
                    _display_interactive_help()
                    continue
                elif user_input.lower() == "stats":
                    _display_current_stats(stats)
                    continue
                elif user_input.lower() == "clear":
                    console.clear()
                    cli_app.display_banner()
                    continue

                # Process the address
                with console.status("[blue]🔄 Processing address..."):
                    try:
                        result = cli_app.pipeline.process(user_input)
                        explanation = cli_app.explainer.explain(result)

                        # Update statistics
                        stats["processed"] += 1
                        if hasattr(result, "success") and result.success:
                            stats["successful"] += 1

                        # Calculate average confidence
                        if hasattr(result, "confidence"):
                            confidence = (
                                getattr(result.confidence, "overall", 0.0) if hasattr(result.confidence, "overall") else 0.0
                            )
                            stats["avg_confidence"] = (
                                stats["avg_confidence"] * (stats["processed"] - 1) + confidence
                            ) / stats["processed"]

                    except Exception as e:
                        console.print(f"[red]❌ Error processing address: {e}[/red]")
                        continue

                # Display results in a beautiful panel
                _display_interactive_result(user_input, result, explanation)

                # Ask for next action
                action = Prompt.ask(
                    "\n[cyan]Next action[/cyan]",
                    choices=["continue", "explain", "stats", "quit"],
                    default="continue",
                    show_choices=False,
                )

                if action == "explain":
                    _display_detailed_explanation(result, explanation)
                elif action == "stats":
                    _display_current_stats(stats)
                elif action == "quit":
                    _display_session_stats(stats)
                    console.print("[yellow]👋 Goodbye![/yellow]")
                    break

            except KeyboardInterrupt:
                _display_session_stats(stats)
                console.print("\n[yellow]👋 Session interrupted. Goodbye![/yellow]")
                break
            except EOFError:
                _display_session_stats(stats)
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                break

    def _process_single_address(cli_app: AddrnormCLI, address: str, explain: bool, verbose: bool):
        """Process a single address with rich output"""

        with console.status("[blue]Processing address..."):
            result = cli_app.pipeline.process(address)

            if explain:
                explanation = cli_app.explainer.explain(result)
            else:
                explanation = None

        # Display result
        _display_single_result(address, result, explanation, verbose)

    def _process_file(
        cli_app: AddrnormCLI,
        input_file: str,
        output_file: str,
        format: str,
        batch_size: int,
        jobs: int,
        explain: bool,
        verbose: bool,
        show_progress: bool,
    ):
        """Process file with progress tracking"""

        input_path = Path(input_file)

        # Count total lines for progress
        with open(input_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        console.print(f"[blue]📂 Processing {total_lines:,} addresses from {input_file}[/blue]")

        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            task = progress.add_task("[green]Processing addresses...", total=total_lines)

            with open(input_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    address = line.strip()
                    if not address:
                        continue

                    try:
                        result = cli_app.pipeline.process(address)

                        result_data = {
                            "line_number": line_num,
                            "original_address": address,
                            "normalized_address": (
                                result.normalized_address.__dict__ if hasattr(result, "normalized_address") else {}
                            ),
                            "confidence": getattr(result.confidence, "overall", 0.0) if hasattr(result, "confidence") else 0.0,
                            "processing_time": getattr(result, "processing_time", 0.0),
                            "success": getattr(result, "success", False),
                        }

                        if explain:
                            result_data["explanation"] = cli_app.explainer.explain(result)

                        results.append(result_data)

                    except Exception as e:
                        if verbose:
                            console.print(f"[red]Error on line {line_num}: {e}[/red]")
                        results.append(
                            {"line_number": line_num, "original_address": address, "error": str(e), "success": False}
                        )

                    progress.update(task, advance=1)

        # Save results
        if output_file:
            _save_results(results, output_file, format)
            console.print(f"[green]✅ Results saved to {output_file}[/green]")

        # Display summary
        _display_processing_summary(results)

    # Helper functions for rich display
    def _display_interactive_result(original: str, result: Any, explanation: str):
        """Display address processing result in interactive mode"""

        # Create result panel
        if hasattr(result, "normalized_address") and result.normalized_address:
            components = result.normalized_address.__dict__

            result_text = ""
            for key, value in components.items():
                if value:
                    key_display = key.replace("_", " ").title()
                    result_text += f"• [cyan]{key_display}:[/cyan] {value}\n"

            # Get confidence info
            confidence = 0.0
            if hasattr(result, "confidence"):
                confidence = getattr(result.confidence, "overall", 0.0) if hasattr(result.confidence, "overall") else 0.0

            # Determine status color
            if confidence >= 0.8:
                status_color = "green"
                status_icon = "✅"
            elif confidence >= 0.6:
                status_color = "yellow"
                status_icon = "⚠️"
            else:
                status_color = "red"
                status_icon = "❌"

            panel_content = f"""[bold]Original:[/bold] {original}

[bold green]Normalized Components:[/bold green]
{result_text}
[bold]Confidence:[/bold] [{status_color}]{confidence:.2f}[/{status_color}] {status_icon}

[dim]{explanation}[/dim]"""

        else:
            panel_content = f"""[bold]Original:[/bold] {original}

[red]❌ Processing failed[/red]
{explanation or 'No explanation available'}"""

        panel = Panel(panel_content, title="🏠 Address Processing Result", border_style="blue", padding=(1, 2))

        console.print(panel)

    def _display_single_result(original: str, result: Any, explanation: Optional[str], verbose: bool):
        """Display single address result"""

        table = Table(title="🏠 Address Normalization Result")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Original Address", original)

        if hasattr(result, "normalized_address") and result.normalized_address:
            components = result.normalized_address.__dict__
            for key, value in components.items():
                if value:
                    key_display = key.replace("_", " ").title()
                    table.add_row(key_display, str(value))

        if hasattr(result, "confidence"):
            confidence = getattr(result.confidence, "overall", 0.0)
            table.add_row("Confidence", f"{confidence:.3f}")

        if hasattr(result, "processing_time"):
            table.add_row("Processing Time", f"{result.processing_time:.3f}s")

        console.print(table)

        if explanation:
            console.print(Panel(explanation, title="💡 Explanation", border_style="yellow"))

    def _display_processing_summary(results: List[Dict]):
        """Display processing summary statistics"""

        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        failed = total - successful

        avg_confidence = 0.0
        if successful > 0:
            confidences = [r.get("confidence", 0.0) for r in results if r.get("success", False)]
            avg_confidence = sum(confidences) / len(confidences)

        avg_time = 0.0
        processing_times = [r.get("processing_time", 0.0) for r in results if "processing_time" in r]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)

        summary_table = Table(title="📊 Processing Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Addresses", f"{total:,}")
        summary_table.add_row("Successful", f"{successful:,} ({successful/total*100:.1f}%)")
        summary_table.add_row("Failed", f"{failed:,} ({failed/total*100:.1f}%)")
        summary_table.add_row("Average Confidence", f"{avg_confidence:.3f}")
        summary_table.add_row("Average Processing Time", f"{avg_time:.3f}s")

        console.print(summary_table)

    def _save_results(results: List[Dict], output_file: str, format: str):
        """Save results to file in specified format"""

        output_path = Path(output_file)

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                for result in results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write("\n")
        elif format == "csv":
            try:
                import pandas as pd

                df = pd.json_normalize(results)
                df.to_csv(output_path, index=False, encoding="utf-8")
            except ImportError:
                # Fallback to basic CSV
                import csv

                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    if results:
                        writer = csv.DictWriter(f, fieldnames=results[0].keys())
                        writer.writeheader()
                        writer.writerows(results)

    def _display_interactive_help():
        """Display help for interactive mode"""

        help_panel = Panel.fit(
            """[bold yellow]🎮 Interactive Mode Commands[/bold yellow]

[cyan]Address Input:[/cyan]
• Type any Turkish address to normalize it
• Use Tab for auto-completion

[cyan]Special Commands:[/cyan]
• [bold]help[/bold]  - Show this help message
• [bold]stats[/bold] - Show current session statistics
• [bold]clear[/bold] - Clear the screen
• [bold]quit[/bold]  - Exit interactive mode

[cyan]Navigation:[/cyan]
• Enter - Process address
• Ctrl+C - Exit
• Up/Down arrows - Browse history
            """,
            title="💡 Help",
            border_style="yellow",
        )
        console.print(help_panel)

    def _display_current_stats(stats: Dict):
        """Display current session statistics"""

        elapsed = time.time() - stats["start_time"]
        rate = stats["processed"] / elapsed if elapsed > 0 else 0

        stats_table = Table(title="📊 Session Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Addresses Processed", str(stats["processed"]))
        stats_table.add_row("Successful", str(stats["successful"]))
        stats_table.add_row("Success Rate", f"{stats['successful']/max(stats['processed'],1)*100:.1f}%")
        stats_table.add_row("Average Confidence", f"{stats['avg_confidence']:.3f}")
        stats_table.add_row("Session Time", f"{elapsed:.1f}s")
        stats_table.add_row("Processing Rate", f"{rate:.2f} addr/sec")

        console.print(stats_table)

    def _display_session_stats(stats: Dict):
        """Display final session statistics"""

        console.print("\n[bold cyan]📊 Session Summary[/bold cyan]")
        _display_current_stats(stats)

    def _display_explanation_result(result: Any, explanation: str, detail: str):
        """Display explanation result"""

        # Show the normalized result first
        if hasattr(result, "normalized_address") and result.normalized_address:
            components = result.normalized_address.__dict__

            result_table = Table(title="🏠 Normalization Result")
            result_table.add_column("Component", style="cyan")
            result_table.add_column("Value", style="green")

            for key, value in components.items():
                if value:
                    key_display = key.replace("_", " ").title()
                    result_table.add_row(key_display, str(value))

            console.print(result_table)

        # Show the explanation
        explanation_panel = Panel(explanation, title=f"💡 Explanation ({detail} level)", border_style="yellow", padding=(1, 2))
        console.print(explanation_panel)

    def _display_detailed_explanation(result: Any, explanation: str):
        """Display detailed explanation in interactive mode"""

        detailed_panel = Panel.fit(
            f"""[bold yellow]💡 Detailed Explanation[/bold yellow]

{explanation}

[dim]Processing method, confidence calculations, and validation steps[/dim]
            """,
            title="Detailed Analysis",
            border_style="yellow",
        )
        console.print(detailed_panel)

else:
    # Fallback to basic functionality if rich/click not available
    cli = None
    interactive_mode = None


# Legacy argparse-based CLI for compatibility
def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""

    parser = argparse.ArgumentParser(
        prog="addrnorm",
        description="Turkish Address Normalization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic normalization to JSONL
  addrnorm normalize --in addresses.txt --out results.jsonl

  # Export to CSV format
  addrnorm normalize --in data.txt --out output.csv --format csv

  # Single address processing
  addrnorm normalize --address "Atatürk Mahallesi Cumhuriyet Caddesi No:15"

For more information and examples, visit: https://github.com/merenert/PreprocessingV2.0
        """,
    )

    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Normalize command
    normalize_parser = subparsers.add_parser("normalize", help="Normalize Turkish addresses")

    # Input options
    input_group = normalize_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--in",
        "--input",
        dest="input_file",
        type=str,
        help="Input file containing addresses (one per line)",
    )
    input_group.add_argument("--address", "-a", type=str, help="Single address to normalize")

    # Output options
    normalize_parser.add_argument(
        "--out",
        "--output",
        dest="output_file",
        type=str,
        help="Output file path",
    )
    normalize_parser.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )

    # Processing options
    normalize_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )
    normalize_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for normalization (default: 0.7)",
    )
    normalize_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Add interactive command if rich is available
    if RICH_AVAILABLE:
        interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")

    # Add version command
    subparsers.add_parser("version", help="Show version information")

    return parser


def run_legacy_normalize(args):
    """Run normalization using legacy interface"""

    cli_app = AddrnormCLI()
    config = {"confidence_threshold": args.confidence_threshold}
    cli_app.initialize_pipeline(config)

    if args.address:
        # Single address processing
        try:
            result = cli_app.pipeline.process(args.address)

            if hasattr(result, "normalized_address") and result.normalized_address:
                result_dict = result.normalized_address.__dict__

                if args.output_file:
                    with open(args.output_file, "w", encoding="utf-8") as f:
                        json.dump(result_dict, f, ensure_ascii=False, indent=2)
                    print(f"Result written to: {args.output_file}")
                else:
                    print(json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                print("Error: Could not normalize address", file=sys.stderr)
                sys.exit(1)

        except Exception as e:
            print(f"Error processing address: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.input_file:
        # File processing
        if not args.output_file:
            print("Error: Output file required for file processing", file=sys.stderr)
            sys.exit(1)

        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                addresses = [line.strip() for line in f if line.strip()]

            print(f"Processing {len(addresses)} addresses...")

            results = []
            for i, address in enumerate(addresses):
                try:
                    result = cli_app.pipeline.process(address)
                    if hasattr(result, "normalized_address") and result.normalized_address:
                        results.append(result.normalized_address.__dict__)
                    else:
                        results.append({"error": "Could not normalize", "original": address})

                    if args.verbose and (i + 1) % 100 == 0:
                        print(f"Processed {i + 1}/{len(addresses)} addresses")

                except Exception as e:
                    results.append({"error": str(e), "original": address})

            # Save results
            with open(args.output_file, "w", encoding="utf-8") as f:
                if args.format == "jsonl":
                    for result in results:
                        json.dump(result, f, ensure_ascii=False)
                        f.write("\n")
                else:  # CSV
                    try:
                        import pandas as pd

                        df = pd.json_normalize(results)
                        df.to_csv(args.output_file, index=False, encoding="utf-8")
                    except ImportError:
                        print("Error: pandas required for CSV output", file=sys.stderr)
                        sys.exit(1)

            print(f"Results written to: {args.output_file}")
            print(f"Processed {len(addresses)} addresses")

        except Exception as e:
            print(f"Error processing file: {e}", file=sys.stderr)
            sys.exit(1)


def show_version():
    """Show version information."""
    print("Turkish Address Normalization Tool")
    print("Version: 2.0.0 Enhanced")
    print("Author: Meren Ertugrul")
    print("Repository: https://github.com/merenert/PreprocessingV2.0")


def main():
    """Main CLI entry point."""

    # If rich and click are available, use enhanced CLI
    if RICH_AVAILABLE and cli:
        try:
            cli()
        except Exception as e:
            # Fallback to legacy mode if enhanced CLI fails
            print(f"Enhanced CLI error: {e}", file=sys.stderr)
            print("Falling back to legacy mode...", file=sys.stderr)
            main_legacy()
    else:
        # Use legacy argparse-based CLI
        main_legacy()


def main_legacy():
    """Legacy main function using argparse"""

    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        if RICH_AVAILABLE:
            # Show interactive mode if available
            try:
                cli_app = AddrnormCLI()
                cli_app.display_banner()
                interactive_mode()
                return
            except:
                pass

        parser.print_help()
        sys.exit(1)

    if args.command == "normalize":
        run_legacy_normalize(args)
    elif args.command == "interactive" and RICH_AVAILABLE:
        cli_app = AddrnormCLI()
        cli_app.display_banner()
        interactive_mode()
    elif args.command == "version":
        show_version()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
