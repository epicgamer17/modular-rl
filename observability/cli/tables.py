from rich.table import Table
from typing import Dict, Any, List

def create_summary_table(data: List[Dict[str, Any]], title: str = "Summary") -> Table:
    """Create a formatted summary table from a list of dictionaries."""
    if not data:
        return Table(title=title)

    columns = list(data[0].keys())
    table = Table(title=title, show_header=True, header_style="bold cyan")
    
    for col in columns:
        table.add_column(col)
        
    for row in data:
        table.add_row(*[str(row.get(col, "")) for col in columns])
        
    return table
