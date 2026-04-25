from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
from rich.columns import Columns
from rich.text import Text
from .tables import create_summary_table

class CLIPrinter:
    """Advanced CLI Terminal Output System for RL execution."""

    def __init__(self):
        self.console = Console()
        self._live: Optional[Live] = None

    def print_metrics(self, step: int, metrics: Dict[str, Any], title: str = "Metrics"):
        """Print a formatted table of metrics."""
        table = create_summary_table([metrics], title=f"{title} (Step {step})")
        self.console.print(table)

    def print_graph_summary(self, nodes: List[Dict[str, Any]], title: str = "Graph Summary"):
        """Print a collapsible tree summary of the graph."""
        tree = Tree(f"[bold blue]{title}[/bold blue]")
        
        # Group by type
        by_type = {}
        for node in nodes:
            ntype = node.get("node_type", "Unknown")
            if ntype not in by_type:
                by_type[ntype] = []
            by_type[ntype].append(node)
            
        for ntype, nlist in by_type.items():
            branch = tree.add(f"[bold cyan]{ntype}[/bold cyan] ({len(nlist)})")
            for node in nlist:
                branch.add(f"[dim]{node['node_id']}[/dim]")
                
        self.console.print(tree)

    def print_step_performance(self, step: int, node_stats: Dict[str, float]):
        """Print performance stats for a single step's execution."""
        table = Table(title=f"Step {step} Performance", show_header=True, header_style="bold magenta")
        table.add_column("Node", style="dim")
        table.add_column("Duration (ms)", justify="right")
        
        total = sum(node_stats.values())
        for node_id, duration in node_stats.items():
            color = "green" if duration < 1.0 else ("yellow" if duration < 5.0 else "red")
            table.add_row(node_id, f"[{color}]{duration:.2f}[/{color}]")
            
        table.add_section()
        table.add_row("[bold]Total[/bold]", f"[bold white]{total:.2f}[/bold white]")
        
        self.console.print(table)

    def print_panel(self, message: str, title: Optional[str] = None, style: str = "green"):
        """Print a message in a beautiful panel."""
        self.console.print(Panel(message, title=title, border_style=style, padding=(1, 2)))

    def clear(self):
        """Clear the console."""
        self.console.clear()

# Global instance
_GLOBAL_PRINTER = CLIPrinter()

def get_printer() -> CLIPrinter:
    return _GLOBAL_PRINTER
