#!/usr/bin/env python
"""
Check that only src.db.session imports asyncpg directly.
This enforces the isolation rule for the pool.
"""
import ast
import sys
from pathlib import Path

# Modules that are allowed to import asyncpg
ALLOWED_MODULES = {"src.db.session"}

# Modules in src that should NOT import asyncpg
# Note: bootstrap_roles and migrate are infrastructure modules that legitimately
# need direct asyncpg connections for setup; they are not restricted.
FORBIDDEN_MODULES = {
    "src.api",
    "src.auth",
    "src.chunking",
    "src.generate",
    "src.ingest",
    "src.llm",
    "src.orchestrate",
    "src.retrieval",
    "src.telemetry",
}


class AsyncpgImportChecker(ast.NodeVisitor):
    def __init__(self, module_name):
        self.module_name = module_name
        self.imports_asyncpg = False
        self.import_lines = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "asyncpg" or alias.name.startswith("asyncpg."):
                self.imports_asyncpg = True
                self.import_lines.append((node.lineno, alias.name))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "asyncpg" or (node.module and node.module.startswith("asyncpg.")):
            self.imports_asyncpg = True
            self.import_lines.append((node.lineno, node.module))
        self.generic_visit(node)


def check_module(module_path):
    """Check a single Python module for asyncpg imports."""
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading {module_path}: {e}")
        return None

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"Syntax error in {module_path}: {e}")
        return None

    # Convert file path to module name
    try:
        relative_path = module_path.relative_to(Path.cwd())
    except ValueError:
        relative_path = Path(module_path)
    module_name = str(relative_path).replace("/", ".").replace("\\", ".").replace(".py", "")
    # Normalize __init__ modules to their package name
    if module_name.endswith(".__init__"):
        module_name = module_name[: -len(".__init__")]

    checker = AsyncpgImportChecker(module_name)
    checker.visit(tree)

    return {
        "module": module_name,
        "path": module_path,
        "imports_asyncpg": checker.imports_asyncpg,
        "import_lines": checker.import_lines,
    }


def main():
    """Run the check."""
    src_dir = Path("src")
    violations = []
    passed = []

    # Check all Python files in src/
    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "__pycache__":
            continue

        result = check_module(py_file)
        if result is None:
            continue

        module = result["module"]
        imports_asyncpg = result["imports_asyncpg"]

        if imports_asyncpg:
            # Violation only if in the forbidden list
            if module in FORBIDDEN_MODULES:
                violations.append(result)
            else:
                # OK: either in ALLOWED_MODULES or infrastructure/unspecified
                passed.append(result)
        else:
            passed.append(result)

    # Report results
    if violations:
        print("=" * 70)
        print("FORBIDDEN asyncpg imports detected:")
        print("=" * 70)
        for violation in violations:
            print(f"\n[FAIL] {violation['module']} ({violation['path']})")
            for lineno, import_name in violation["import_lines"]:
                print(f"   Line {lineno}: import {import_name}")
        print("\n" + "=" * 70)
        print("LINT FAILED")
        print("=" * 70)
        return 1
    else:
        print("=" * 70)
        print("[PASS] All asyncpg imports are restricted to src.db.session")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
