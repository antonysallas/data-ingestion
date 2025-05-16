#!/usr/bin/env python
"""
Fix script for OPL Ingestion Pipeline.

This script checks for common issues and fixes them.
"""
import sys
from pathlib import Path


def fix_module_imports(module_path, fixes):
    """Fix imports in a module file."""
    if not Path(module_path).exists():
        print(f"Error: Module file {module_path} not found")
        return False

    with open(module_path, "r") as file:
        content = file.read()

    modified = False
    for fix in fixes:
        if "missing_import" in fix:
            import_line = fix["missing_import"]
            # Check if import already exists
            if import_line not in content:
                # Add import after other imports
                import_section_end = content.find("\n\n# Configure")
                if import_section_end == -1:
                    import_section_end = content.find("\n\n_log =")

                if import_section_end != -1:
                    # Insert after imports
                    new_content = (
                        content[:import_section_end] + "\n" + import_line + content[import_section_end:]
                    )
                    content = new_content
                    modified = True
                    print(f"Added missing import: {import_line} to {module_path}")

        if "function_fix" in fix:
            function_name = fix["function_fix"]["name"]
            old_signature = fix["function_fix"]["old_signature"]
            new_signature = fix["function_fix"]["new_signature"]

            if old_signature in content:
                content = content.replace(old_signature, new_signature)
                modified = True
                print(f"Fixed function {function_name} in {module_path}")

    if modified:
        with open(module_path, "w") as file:
            file.write(content)
        return True

    return False


# Specify fixes for modules
fixes = {
    "markdown_processing.py": [
        {
            "function_fix": {
                "name": "process_practice",
                "old_signature": 'def process_practice(source, output_dir):\n    """\n    Process a single practice source (URL or file path) and generate a markdown file.\n\n    Args:\n        source: URL or Path to the HTML file to process\n        output_dir: Directory to save the markdown file\n\n    Returns:\n        bool: Success status\n    """\n    # Import here to avoid circular imports\n    import requests',
                "new_signature": 'def process_practice(source, output_dir):\n    """\n    Process a single practice source (URL or file path) and generate a markdown file.\n\n    Args:\n        source: URL or Path to the HTML file to process\n        output_dir: Directory to save the markdown file\n\n    Returns:\n        bool: Success status\n    """\n    # Import here to avoid circular imports\n    import requests\n    from pathlib import Path',
            }
        }
    ],
    "html_processing.py": [{"missing_import": "from pathlib import Path"}],
}


def main():
    """Run the fix script."""
    # Find module files
    module_dir = Path.cwd()
    fixed_any = False

    for module_name, module_fixes in fixes.items():
        module_path = module_dir / module_name
        if module_path.exists():
            if fix_module_imports(module_path, module_fixes):
                fixed_any = True
        else:
            print(f"Warning: Module file {module_name} not found")

    if fixed_any:
        print("\nFixed issues in module files. Please run your script again.")
    else:
        print("\nNo issues found or fixed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
