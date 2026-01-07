import ast
import sys
from pathlib import Path
from typing import Set, List

PROJECT_PREFIXES = ("gm.",)
STD_IMPORT_PREFIXES = (
    "torch", "numpy", "math", "time", "typing", "dataclasses",
    "transformers", "datasets", "collections", "itertools",
    "functools", "json", "random", "os", "sys"
)


def is_project_import(module: str | None) -> bool:
    if module is None:
        return False
    return module.startswith(PROJECT_PREFIXES)


def is_std_import(module: str | None) -> bool:
    if module is None:
        return False
    return module.startswith(STD_IMPORT_PREFIXES)


def resolve_import(module: str, project_root: Path) -> Path | None:
    rel = module.replace(".", "/") + ".py"
    path = project_root / rel
    return path if path.exists() else None


def resolve_relative_import(level: int, module: str | None, current_file: Path) -> Path | None:
    base = current_file.parent
    for _ in range(level):
        base = base.parent
    if module:
        base = base / module.replace(".", "/")
    return (base.with_suffix(".py") if base.with_suffix(".py").exists() else None)


def inline_file(
        path: Path,
        project_root: Path,
        visited: Set[Path],
        collected_imports: Set[str],
        is_entry: bool = False,
) -> List[str]:
    path = path.resolve()
    if path in visited:
        return []

    visited.add(path)

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    inlined_code: List[str] = []
    deferred_code: List[str] = []

    # --- collect imports ---
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if is_project_import(name):
                    dep = resolve_import(name, project_root)
                    if dep:
                        inlined_code += inline_file(dep, project_root, visited, collected_imports)
                else:
                    collected_imports.add(f"import {name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if node.level > 0:
                dep = resolve_relative_import(node.level, module, path)
                if dep:
                    inlined_code += inline_file(dep, project_root, visited, collected_imports)
            elif is_project_import(module):
                dep = resolve_import(module, project_root)
                if dep:
                    inlined_code += inline_file(dep, project_root, visited, collected_imports)
            else:
                if module:
                    collected_imports.add(
                        f"from {module} import {', '.join(a.name for a in node.names)}"
                    )

    # --- strip imports + optional __main__ ---
    lines = source.splitlines()
    inside_main = False
    main_indent = None

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if stripped.startswith(("import ", "from ")):
            continue

        if stripped.startswith('if __name__ == "__main__"'):
            if not is_entry:
                inside_main = True
                main_indent = indent
                continue
            else:
                deferred_code.append(line)
                continue

        if inside_main:
            if indent <= main_indent and stripped:
                inside_main = False
            else:
                continue

        deferred_code.append(line)

    return inlined_code + deferred_code


def main():
    if len(sys.argv) != 3:
        print("Usage: python inline_project.py <project_root> <entry_file>")
        sys.exit(1)

    project_root = Path(sys.argv[1]).resolve()
    entry_file = Path(sys.argv[2]).resolve()

    if not entry_file.exists():
        raise FileNotFoundError(entry_file)

    visited: Set[Path] = set()
    collected_imports: Set[str] = set()

    body = inline_file(
        entry_file,
        project_root,
        visited,
        collected_imports,
        is_entry=True,
    )

    output_path = "colab_inline.py"

    with open(output_path, "w", encoding="utf-8") as f:
        for imp in sorted(collected_imports):
            f.write(imp + "\n")
        f.write("\n\n")
        for line in body:
            f.write(line + "\n")

    print(f"[+] Inlined file written to: {output_path}")


if __name__ == "__main__":
    main()
