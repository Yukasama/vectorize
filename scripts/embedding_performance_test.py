#!/usr/bin/env python3
"""Erweiterte Embedding Performance Test Suite
Testet Model Loading, Caching, Concurrent Requests und verschiedene Szenarien
"""

import asyncio
import json
import statistics
import time
from datetime import datetime

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


class EmbeddingPerformanceTester:
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)
        self.console = Console()
        self.results = {}

    async def close(self):
        """Schließt den HTTP Client"""
        await self.client.aclose()

    async def load_huggingface_model(
        self, model_tag: str, revision: str = "main"
    ) -> str | None:
        """Lädt ein HuggingFace-Modell und gibt den internen model_tag zurück"""
        self.console.print(f"[bold blue]Lade Modell:[/bold blue] {model_tag}")

        upload_data = {"model_tag": model_tag, "revision": revision}

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Model wird hochgeladen...", total=None)

                response = await self.client.post(
                    f"{self.base_url}/uploads/huggingface", json=upload_data
                )

                progress.update(task, description="Upload abgeschlossen")

            if response.status_code == 201:
                self.console.print("[green]✓[/green] Upload erfolgreich")
                return await self._find_actual_model_tag(model_tag)

            if response.status_code == 409:
                self.console.print("[yellow]⚠[/yellow] Modell bereits vorhanden")
                actual_tag = await self._find_actual_model_tag(model_tag)

                if actual_tag and await self._test_model_ready(actual_tag):
                    self.console.print("[green]✓[/green] Modell ist bereit")
                    return actual_tag

                self.console.print("[yellow]⏳[/yellow] Warte auf Modell...")
                await asyncio.sleep(5)
                return actual_tag
            self.console.print(
                f"[red]✗[/red] Upload fehlgeschlagen: {response.status_code}"
            )
            return None

        except Exception as e:
            self.console.print(f"[red]✗[/red] Fehler beim Upload: {e}")
            return None

    async def _find_actual_model_tag(self, original_model_tag: str) -> str | None:
        """Findet den internen model_tag"""
        try:
            response = await self.client.get(f"{self.base_url}/models?size=50")

            if response.status_code != 200:
                return None

            models_data = response.json()
            for model in models_data.get("items", []):
                if model.get("name") == original_model_tag:
                    return model.get("model_tag")

            return None

        except Exception as e:
            self.console.print(f"[red]Fehler beim Suchen des Modells:[/red] {e}")
            return None

    async def _test_model_ready(self, model_tag: str) -> bool:
        """Testet ob das Modell bereit ist"""
        try:
            warmup_data = {"model": model_tag, "input": "warmup test"}
            response = await self.client.post(
                f"{self.base_url}/embeddings", json=warmup_data
            )
            return response.status_code == 200
        except:
            return False

    async def benchmark_cold_vs_warm(self, model_tag: str) -> dict:
        """Vergleicht Cold Start vs. Warm Cache Performance"""
        self.console.print("\n[bold cyan]🧪 Cold vs Warm Benchmark[/bold cyan]")

        # Cold Start simulieren (verschiedene Inputs für Cache-Miss)
        cold_times = []
        self.console.print("Testing Cold Start (verschiedene Inputs)...")

        for i in range(3):
            cold_input = f"cold start test sentence number {i} with unique content"
            start_time = time.perf_counter()

            response = await self.client.post(
                f"{self.base_url}/embeddings",
                json={"model": model_tag, "input": cold_input},
            )

            elapsed = time.perf_counter() - start_time

            if response.status_code == 200:
                cold_times.append(elapsed)
                self.console.print(f"  Cold {i + 1}: {elapsed:.3f}s")
            else:
                self.console.print(
                    f"  Cold {i + 1}: [red]Error {response.status_code}[/red]"
                )

        # Warm Cache Tests (gleicher Input)
        warm_times = []
        warm_input = "this is a repeated test sentence for warm cache testing"

        self.console.print("Testing Warm Cache (wiederholte Inputs)...")

        for i in range(7):  # Mehr Tests für bessere Statistik
            start_time = time.perf_counter()

            response = await self.client.post(
                f"{self.base_url}/embeddings",
                json={"model": model_tag, "input": warm_input},
            )

            elapsed = time.perf_counter() - start_time

            if response.status_code == 200:
                warm_times.append(elapsed)
                self.console.print(f"  Warm {i + 1}: {elapsed:.3f}s")
            else:
                self.console.print(
                    f"  Warm {i + 1}: [red]Error {response.status_code}[/red]"
                )

        # Statistiken berechnen
        cold_avg = statistics.mean(cold_times) if cold_times else 0
        warm_avg = statistics.mean(warm_times) if warm_times else 0
        cold_std = statistics.stdev(cold_times) if len(cold_times) > 1 else 0
        warm_std = statistics.stdev(warm_times) if len(warm_times) > 1 else 0

        improvement = ((cold_avg - warm_avg) / cold_avg * 100) if cold_avg > 0 else 0
        speedup = cold_avg / warm_avg if warm_avg > 0 else 0

        result = {
            "cold_times": cold_times,
            "warm_times": warm_times,
            "cold_avg": cold_avg,
            "warm_avg": warm_avg,
            "cold_std": cold_std,
            "warm_std": warm_std,
            "improvement_percent": improvement,
            "speedup_factor": speedup,
        }

        self.results["cold_vs_warm"] = result
        return result

    async def benchmark_batch_sizes(self, model_tag: str) -> dict:
        """Testet verschiedene Batch-Größen"""
        self.console.print("\n[bold cyan]📊 Batch Size Benchmark[/bold cyan]")

        batch_configs = {
            "Single": ["Single text for embedding generation test."],
            "Batch_3": [
                "First sentence in small batch.",
                "Second sentence for comparison.",
                "Third and final sentence.",
            ],
            "Batch_5": [
                "First sentence in medium batch test.",
                "Second sentence with different content.",
                "Third sentence for batch processing.",
                "Fourth sentence in this group.",
                "Fifth and final sentence here.",
            ],
            "Batch_10": [
                f"Batch sentence number {i} for performance testing."
                for i in range(1, 11)
            ],
            "Long_Text": [
                "This is a significantly longer text that contains multiple sentences and various topics including machine learning, natural language processing, deep learning architectures, transformer models, attention mechanisms, and their applications in modern AI systems. The purpose is to test how the embedding generation performs with longer input sequences that might require more computational resources and time to process effectively."
            ],
        }

        batch_results = {}

        for batch_name, texts in batch_configs.items():
            self.console.print(f"Testing {batch_name} ({len(texts)} texts)...")

            times = []
            for run in range(5):  # 5 Durchläufe pro Batch-Größe
                start_time = time.perf_counter()

                response = await self.client.post(
                    f"{self.base_url}/embeddings",
                    json={"model": model_tag, "input": texts},
                )

                elapsed = time.perf_counter() - start_time

                if response.status_code == 200:
                    times.append(elapsed)
                    self.console.print(f"  Run {run + 1}: {elapsed:.3f}s")
                else:
                    self.console.print(
                        f"  Run {run + 1}: [red]Error {response.status_code}[/red]"
                    )

                await asyncio.sleep(0.2)  # Kurze Pause

            if times:
                avg_time = statistics.mean(times)
                std_time = statistics.stdev(times) if len(times) > 1 else 0
                min_time = min(times)
                max_time = max(times)

                # Tokens pro Sekunde schätzen (grob)
                total_chars = sum(len(text) for text in texts)
                estimated_tokens = total_chars / 4  # Grobe Schätzung
                tokens_per_second = estimated_tokens / avg_time

                batch_results[batch_name] = {
                    "times": times,
                    "avg_time": avg_time,
                    "std_time": std_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "text_count": len(texts),
                    "estimated_tokens": estimated_tokens,
                    "tokens_per_second": tokens_per_second,
                }

        self.results["batch_sizes"] = batch_results
        return batch_results

    async def benchmark_concurrent_requests(self, model_tag: str) -> dict:
        """Testet gleichzeitige Requests"""
        self.console.print("\n[bold cyan]⚡ Concurrent Requests Benchmark[/bold cyan]")

        test_text = "Concurrent request test sentence for performance measurement."
        concurrency_levels = [1, 2, 5, 10]

        concurrent_results = {}

        for concurrency in concurrency_levels:
            self.console.print(f"Testing {concurrency} concurrent requests...")

            async def single_request():
                start_time = time.perf_counter()
                response = await self.client.post(
                    f"{self.base_url}/embeddings",
                    json={"model": model_tag, "input": test_text},
                )
                elapsed = time.perf_counter() - start_time
                return elapsed, response.status_code == 200

            # Führe mehrere Batches aus für bessere Statistik
            all_times = []
            success_count = 0

            for batch in range(3):  # 3 Batches pro Concurrency-Level
                start_batch = time.perf_counter()

                # Starte concurrent requests
                tasks = [single_request() for _ in range(concurrency)]
                results = await asyncio.gather(*tasks)

                batch_time = time.perf_counter() - start_batch

                # Sammle Ergebnisse
                request_times = [r[0] for r in results]
                successes = sum(1 for r in results if r[1])

                all_times.extend(request_times)
                success_count += successes

                throughput = concurrency / batch_time

                self.console.print(
                    f"  Batch {batch + 1}: {batch_time:.3f}s total, "
                    f"{throughput:.1f} req/s, {successes}/{concurrency} success"
                )

            if all_times:
                avg_time = statistics.mean(all_times)
                std_time = statistics.stdev(all_times) if len(all_times) > 1 else 0

                concurrent_results[concurrency] = {
                    "times": all_times,
                    "avg_time": avg_time,
                    "std_time": std_time,
                    "success_rate": success_count / (concurrency * 3),
                    "total_requests": concurrency * 3,
                }

        self.results["concurrent"] = concurrent_results
        return concurrent_results

    async def check_server_health(self) -> bool:
        """Prüft ob der Server erreichbar ist"""
        try:
            response = await self.client.get(
                f"{self.base_url.replace('/v1', '')}/health"
            )
            return response.status_code == 204
        except:
            return False

    def print_results_table(self):
        """Zeigt Ergebnisse in einer schönen Tabelle"""
        if "cold_vs_warm" in self.results:
            table = Table(title="🔥 Cold vs Warm Performance")
            table.add_column("Metric", style="cyan")
            table.add_column("Cold Start", style="red")
            table.add_column("Warm Cache", style="green")
            table.add_column("Improvement", style="bold yellow")

            cold_warm = self.results["cold_vs_warm"]
            table.add_row(
                "Average Time",
                f"{cold_warm['cold_avg']:.3f}s",
                f"{cold_warm['warm_avg']:.3f}s",
                f"{cold_warm['improvement_percent']:.1f}%",
            )
            table.add_row(
                "Std Deviation",
                f"±{cold_warm['cold_std']:.3f}s",
                f"±{cold_warm['warm_std']:.3f}s",
                f"{cold_warm['speedup_factor']:.1f}x faster",
            )

            self.console.print(table)

        if "batch_sizes" in self.results:
            table = Table(title="📊 Batch Size Performance")
            table.add_column("Batch Type", style="cyan")
            table.add_column("Texts", justify="right")
            table.add_column("Avg Time", justify="right")
            table.add_column("Tokens/sec", justify="right", style="green")
            table.add_column("Efficiency", justify="right", style="yellow")

            batch_data = self.results["batch_sizes"]
            for batch_name, data in batch_data.items():
                efficiency = data["text_count"] / data["avg_time"]
                table.add_row(
                    batch_name,
                    str(data["text_count"]),
                    f"{data['avg_time']:.3f}s",
                    f"{data['tokens_per_second']:.0f}",
                    f"{efficiency:.1f} texts/s",
                )

            self.console.print(table)

        if "concurrent" in self.results:
            table = Table(title="⚡ Concurrent Performance")
            table.add_column("Concurrency", justify="right", style="cyan")
            table.add_column("Avg Response", justify="right")
            table.add_column("Success Rate", justify="right", style="green")
            table.add_column("Throughput", justify="right", style="yellow")

            concurrent_data = self.results["concurrent"]
            for concurrency, data in concurrent_data.items():
                throughput = (
                    concurrency / data["avg_time"] if data["avg_time"] > 0 else 0
                )
                table.add_row(
                    str(concurrency),
                    f"{data['avg_time']:.3f}s",
                    f"{data['success_rate']:.1%}",
                    f"{throughput:.1f} req/s",
                )

            self.console.print(table)

    def save_results(self, filepath: str = "embedding_benchmark_results.json"):
        """Speichert Ergebnisse in JSON-Datei"""
        results_with_metadata = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "results": self.results,
        }

        with open(filepath, "w") as f:
            json.dump(results_with_metadata, f, indent=2, default=str)

        self.console.print(f"[green]✓[/green] Ergebnisse gespeichert: {filepath}")


async def main():
    """Hauptfunktion für den erweiterten Performance-Test"""
    tester = EmbeddingPerformanceTester()

    try:
        tester.console.print(
            "[bold blue]🚀 Embedding Performance Test Suite[/bold blue]"
        )
        tester.console.print("=" * 60)

        # 1. Server-Status prüfen
        if not await tester.check_server_health():
            tester.console.print("[red]✗ Server nicht erreichbar![/red]")
            tester.console.print("Starten Sie den Server mit: uv run app")
            return

        tester.console.print("[green]✓[/green] Server ist erreichbar")

        # 2. Modell laden
        model_to_test = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model_tag = await tester.load_huggingface_model(model_to_test)

        if not model_tag:
            tester.console.print("[red]✗ Modell konnte nicht geladen werden[/red]")
            return

        tester.console.print(f"[green]✓[/green] Modell bereit: {model_tag}")

        # 3. Performance-Tests durchführen
        tester.console.print(
            "\n[bold yellow]📈 Starting Performance Benchmarks...[/bold yellow]"
        )

        # Cold vs Warm Test
        await tester.benchmark_cold_vs_warm(model_tag)

        # Batch Size Tests
        await tester.benchmark_batch_sizes(model_tag)

        # Concurrent Request Tests
        await tester.benchmark_concurrent_requests(model_tag)

        # 4. Ergebnisse anzeigen
        tester.console.print("\n[bold green]🎯 BENCHMARK RESULTS[/bold green]")
        tester.console.print("=" * 60)
        tester.print_results_table()

        # 5. Ergebnisse speichern
        tester.save_results()

        # 6. Empfehlungen
        tester.console.print("\n[bold blue]💡 Empfehlungen:[/bold blue]")

        if "cold_vs_warm" in tester.results:
            improvement = tester.results["cold_vs_warm"]["improvement_percent"]
            if improvement > 50:
                tester.console.print("[green]✓[/green] Excellent caching performance!")
            elif improvement > 25:
                tester.console.print(
                    "[yellow]⚠[/yellow] Good caching, could be optimized further"
                )
            else:
                tester.console.print(
                    "[red]⚠[/red] Low caching benefit, check implementation"
                )

        tester.console.print(
            "\n[dim]Test abgeschlossen. Ergebnisse wurden gespeichert.[/dim]"
        )

    except KeyboardInterrupt:
        tester.console.print("\n[yellow]Test abgebrochen[/yellow]")
    except Exception as e:
        tester.console.print(f"\n[red]Unerwarteter Fehler:[/red] {e}")
    finally:
        await tester.close()


if __name__ == "__main__":
    # Dependencies check
    try:
        import rich
    except ImportError:
        print("Installing required dependency: rich")
        import subprocess

        subprocess.check_call(["pip", "install", "rich"])

    asyncio.run(main())
