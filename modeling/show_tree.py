"""
Generate and display a visual directory tree of the modeling folder.
"""
from pathlib import Path
import os
from typing import Optional


def generate_tree(directory: Path, prefix: str = "", is_last: bool = True, 
                 max_depth: int = 4, current_depth: int = 0, 
                 exclude_dirs: Optional[set] = None, exclude_patterns: Optional[set] = None):
    """
    Generate a visual tree structure of a directory.
    
    Args:
        directory: Directory path to visualize
        prefix: Prefix for tree branches
        is_last: Whether this is the last item in its parent
        max_depth: Maximum depth to traverse
        current_depth: Current depth level
        exclude_dirs: Set of directory names to exclude
        exclude_patterns: Set of patterns to exclude files
    """
    if exclude_dirs is None:
        exclude_dirs = {'__pycache__', '.git', '.ipynb_checkpoints', 'node_modules', '.venv', 'venv'}
    
    if exclude_patterns is None:
        exclude_patterns = {'.pyc', '.pyo', '.DS_Store', '.gitignore'}
    
    if current_depth > max_depth:
        return []
    
    lines = []
    
    # Get items in directory
    try:
        items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return lines
    
    # Filter out excluded items
    items = [
        item for item in items 
        if (item.name not in exclude_dirs and 
            not any(item.name.endswith(pattern) for pattern in exclude_patterns))
    ]
    
    for i, item in enumerate(items):
        is_last_item = (i == len(items) - 1)
        
        # Build the tree branch
        if is_last_item:
            branch = "└── "
            extension = "    "
        else:
            branch = "├── "
            extension = "│   "
        
        # Add item name with appropriate icon
        if item.is_dir():
            icon = "📁 "
            name_display = f"{item.name}/"
        else:
            # Different icons for different file types
            if item.suffix == '.py':
                icon = "🐍 "
            elif item.suffix == '.pkl':
                icon = "💾 "
            elif item.suffix in ['.png', '.jpg', '.jpeg']:
                icon = "🖼️  "
            elif item.suffix in ['.json', '.yaml', '.yml']:
                icon = "📄 "
            elif item.suffix == '.csv':
                icon = "📊 "
            elif item.suffix == '.md':
                icon = "📝 "
            else:
                icon = "📄 "
            
            name_display = item.name
            
            # Add file size for non-python files
            if item.suffix != '.py':
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024**2:
                    size_str = f"{size/1024:.1f}KB"
                elif size < 1024**3:
                    size_str = f"{size/(1024**2):.1f}MB"
                else:
                    size_str = f"{size/(1024**3):.1f}GB"
                name_display += f" ({size_str})"
        
        lines.append(f"{prefix}{branch}{icon}{name_display}")
        
        # Recurse into directories
        if item.is_dir():
            new_prefix = prefix + extension
            lines.extend(generate_tree(
                item, new_prefix, is_last_item, max_depth, 
                current_depth + 1, exclude_dirs, exclude_patterns
            ))
    
    return lines


def print_modeling_tree():
    """Print the complete modeling directory tree."""
    modeling_dir = Path(__file__).parent
    
    print("=" * 80)
    print(" MODELING DIRECTORY STRUCTURE ".center(80, "="))
    print("=" * 80)
    print()
    print(f"📁 {modeling_dir.name}/")
    
    tree_lines = generate_tree(modeling_dir, prefix="", max_depth=3)
    for line in tree_lines:
        print(line)
    
    print()
    print("=" * 80)
    
    # Count files by type
    python_files = list(modeling_dir.rglob("*.py"))
    pkl_files = list(modeling_dir.rglob("*.pkl"))
    png_files = list(modeling_dir.rglob("*.png"))
    csv_files = list(modeling_dir.rglob("*.csv"))
    json_files = list(modeling_dir.rglob("*.json"))
    
    print(f"\n📊 Summary:")
    print(f"   Python files:     {len(python_files)}")
    print(f"   Model files:      {len(pkl_files)}")
    print(f"   Plots:            {len(png_files)}")
    print(f"   CSV files:        {len(csv_files)}")
    print(f"   JSON files:       {len(json_files)}")
    print()


def print_output_tree_only():
    """Print only the output directory tree."""
    modeling_dir = Path(__file__).parent
    output_dir = modeling_dir / "output"
    
    if not output_dir.exists():
        print("⚠️  Output directory does not exist yet. Run main.py first.")
        return
    
    print("=" * 80)
    print(" OUTPUT DIRECTORY STRUCTURE ".center(80, "="))
    print("=" * 80)
    print()
    print(f"📁 output/")
    
    tree_lines = generate_tree(output_dir, prefix="", max_depth=2)
    for line in tree_lines:
        print(line)
    
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--output-only":
        print_output_tree_only()
    else:
        print_modeling_tree()
