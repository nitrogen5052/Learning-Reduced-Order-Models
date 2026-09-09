#!/usr/bin/env python3
"""Extract deterministic architecture evidence from the live ``lrom`` package."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import pickletools
import re
import sys
import textwrap
import tomllib
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
LROM_DIR = ROOT / "lrom"
OUTPUT = ROOT / "docs" / "architecture.json"
CONSUMER_DIRECTORIES = ("notebooks", "tools", "tests", "benchmarks")
MISSING_ROLE = "Missing module docstring."


@dataclass(frozen=True)
class ImportRecord:
    form: str
    module: str | None
    names: tuple[str, ...]
    level: int
    line: int
    import_type: str


def _source_tree(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _import_records(tree: ast.Module) -> list[ImportRecord]:
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    records: list[ImportRecord] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        import_type = "static" if isinstance(parents.get(node), ast.Module) else "dynamic"
        if isinstance(node, ast.Import):
            for alias in node.names:
                records.append(
                    ImportRecord(
                        "import", alias.name, (), 0, node.lineno, import_type
                    )
                )
        else:
            records.append(
                ImportRecord(
                    "from",
                    node.module,
                    tuple(sorted(alias.name for alias in node.names)),
                    node.level,
                    node.lineno,
                    import_type,
                )
            )
    return sorted(
        records,
        key=lambda item: (
            item.line,
            item.form,
            item.module or "",
            item.names,
            item.level,
            item.import_type,
        ),
    )


def _merge_import_type(current: str | None, candidate: str) -> str:
    if current == "static" or candidate == "static":
        return "static"
    return "dynamic"


def _internal_target(record: ImportRecord, modules: set[str]) -> str | None:
    if record.form == "from" and record.level == 1:
        if record.module:
            candidate = record.module.split(".", 1)[0]
            return candidate if candidate in modules else None
        matches = sorted(set(record.names).intersection(modules))
        return matches[0] if len(matches) == 1 else None
    return None


def _external_root(record: ImportRecord) -> str | None:
    if record.level:
        return None
    if not record.module:
        return None
    return record.module.split(".", 1)[0]


def _cycles(modules: Iterable[str], adjacency: dict[str, set[str]]) -> list[list[str]]:
    state = {module: 0 for module in modules}
    stack: list[str] = []
    found: set[tuple[str, ...]] = set()

    def canonical(cycle: list[str]) -> tuple[str, ...]:
        body = cycle[:-1]
        rotations = [tuple(body[index:] + body[:index]) for index in range(len(body))]
        best = min(rotations)
        return best + (best[0],)

    def visit(module: str) -> None:
        state[module] = 1
        stack.append(module)
        for dependency in sorted(adjacency[module]):
            if state[dependency] == 0:
                visit(dependency)
            elif state[dependency] == 1:
                start = stack.index(dependency)
                found.add(canonical(stack[start:] + [dependency]))
        stack.pop()
        state[module] = 2

    for module in sorted(modules):
        if state[module] == 0:
            visit(module)
    return [list(cycle) for cycle in sorted(found)]


def _layers(modules: Iterable[str], adjacency: dict[str, set[str]]) -> dict[str, int]:
    memo: dict[str, int] = {}

    def layer(module: str) -> int:
        if module not in memo:
            dependencies = adjacency[module]
            memo[module] = (
                0 if not dependencies else 1 + max(layer(item) for item in dependencies)
            )
        return memo[module]

    return {module: layer(module) for module in sorted(modules)}


def _requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9_.-]*", requirement)
    if match is None:
        raise ValueError(f"cannot parse dependency requirement: {requirement!r}")
    return match.group(0)


def _project_metadata() -> tuple[dict[str, object], set[str]]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        document = tomllib.load(stream)
    project = document["project"]
    requirements = list(project.get("dependencies", ()))
    for group in project.get("optional-dependencies", {}).values():
        requirements.extend(group)
    distributions = {_requirement_name(item) for item in requirements}
    metadata = {
        "distribution": project["name"],
        "version": project["version"],
        "description": project["description"],
    }
    return metadata, distributions


def _normalized_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution_for_import(import_name: str, distributions: set[str]) -> str:
    normalized_import = _normalized_distribution(import_name)
    exact = [
        item for item in distributions
        if _normalized_distribution(item) == normalized_import
    ]
    if len(exact) == 1:
        return exact[0]
    suffix = [
        item for item in distributions
        if _normalized_distribution(item).endswith("-" + normalized_import)
    ]
    return suffix[0] if len(suffix) == 1 else import_name


def _external_dependency(
    import_name: str, import_type: str, distributions: set[str]
) -> dict[str, str]:
    distribution = _distribution_for_import(import_name, distributions)
    label = distribution
    if distribution != import_name:
        label = f"{distribution} (import {import_name})"
    return {
        "import_name": import_name,
        "distribution": distribution,
        "display_name": label,
        "type": import_type,
    }


def _consumer_target(record: ImportRecord, modules: set[str]) -> str | None:
    root = _external_root(record)
    if root != "lrom" or not record.module:
        return None
    parts = record.module.split(".")
    if len(parts) == 1:
        return "__init__"
    return parts[1] if parts[1] in modules else "__init__"


def _consumer_units(path: Path) -> list[tuple[str, ast.Module, str, int]]:
    relative = path.relative_to(ROOT).as_posix()
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        units = []
        for index, cell in enumerate(notebook.get("cells", ())):
            if cell.get("cell_type") != "code":
                continue
            source_value = cell.get("source", "")
            source = (
                source_value if isinstance(source_value, str) else "".join(source_value)
            )
            try:
                tree = ast.parse(source, filename=f"{relative}#cell-{index}")
            except SyntaxError as exc:
                raise SyntaxError(
                    f"cannot scan Python in {relative} cell {index}: {exc}"
                ) from exc
            units.append((f"{relative}#cell-{index}", tree, "notebook-cell", 0))
        return units

    source, tree = _source_tree(path)
    units = [(relative, tree, "source", 0)]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if "import " not in node.value:
            continue
        try:
            embedded = ast.parse(
                textwrap.dedent(node.value),
                filename=f"{relative}#embedded-{node.lineno}",
            )
        except SyntaxError:
            continue
        if not any(
            isinstance(item, (ast.Import, ast.ImportFrom)) for item in ast.walk(embedded)
        ):
            continue
        units.append(
            (
                f"{relative}#embedded-{node.lineno}",
                embedded,
                "embedded-python",
                node.lineno - 1,
            )
        )
    return units


def _scan_consumers(
    modules: set[str], distributions: set[str]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    consumer_edges: dict[tuple[object, ...], dict[str, object]] = {}
    external_edges: dict[tuple[object, ...], dict[str, object]] = {}
    workspace_edges: dict[tuple[object, ...], dict[str, object]] = {}
    groups: list[dict[str, object]] = []
    local_roots = {
        path.stem for path in ROOT.glob("*.py")
    } | {
        path.name for path in ROOT.iterdir() if path.is_dir()
    }

    for category in CONSUMER_DIRECTORIES:
        directory = ROOT / category
        paths = sorted(
            path for path in directory.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and (
                path.suffix == ".py"
                or (category == "notebooks" and path.suffix == ".ipynb")
            )
        )
        for path in paths:
            for source_name, tree, context, offset in _consumer_units(path):
                for record in _import_records(tree):
                    line = record.line + offset
                    target = _consumer_target(record, modules)
                    if target is not None:
                        key = (
                            category, source_name, target, line,
                            record.import_type, context, record.names,
                        )
                        consumer_edges[key] = {
                            "category": category,
                            "source": source_name,
                            "target": target,
                            "line": line,
                            "type": record.import_type,
                            "context": context,
                            "imported": list(record.names),
                        }
                        continue

                    root = _external_root(record)
                    if root is None or root in sys.stdlib_module_names or root == "lrom":
                        continue
                    if root in CONSUMER_DIRECTORIES:
                        if root != category:
                            key = (category, source_name, root, line, record.import_type)
                            workspace_edges[key] = {
                                "source_category": category,
                                "source": source_name,
                                "target_category": root,
                                "line": line,
                                "type": record.import_type,
                                "context": context,
                            }
                        continue
                    if root in local_roots:
                        continue
                    dependency = _external_dependency(
                        root, record.import_type, distributions
                    )
                    key = (
                        category, source_name, root, line,
                        record.import_type, context,
                    )
                    external_edges[key] = {
                        "category": category,
                        "source": source_name,
                        "line": line,
                        "context": context,
                        **dependency,
                    }

        category_edges = [
            edge for edge in consumer_edges.values() if edge["category"] == category
        ]
        groups.append(
            {
                "category": category,
                "file_count": len(paths),
                "lrom_import_count": len(category_edges),
                "lrom_modules": sorted({edge["target"] for edge in category_edges}),
            }
        )

    sort_key = lambda item: tuple(
        str(item.get(key, ""))
        for key in ("category", "source_category", "source", "target", "target_category", "line", "import_name", "type", "context")
    )
    return (
        sorted(consumer_edges.values(), key=sort_key),
        sorted(external_edges.values(), key=sort_key),
        sorted(workspace_edges.values(), key=sort_key),
        sorted(groups, key=lambda item: str(item["category"])),
    )


def _scan_pickles(modules: set[str]) -> list[dict[str, object]]:
    findings = []
    for path in sorted((ROOT / "models" / "three_window").glob("*.pkl")):
        module_strings: set[str] = set()
        with path.open("rb") as stream:
            for opcode, argument, _position in pickletools.genops(stream):
                values: list[str] = []
                if opcode.name == "GLOBAL" and isinstance(argument, str):
                    values.append(argument.split()[0])
                elif isinstance(argument, str):
                    values.append(argument)
                for value in values:
                    if value == "lrom" or value.startswith("lrom."):
                        module_strings.add(value)
        referenced = sorted(
            {
                value.split(".", 2)[1]
                for value in module_strings
                if value.startswith("lrom.")
                and len(value.split(".")) > 1
                and value.split(".", 2)[1] in modules
            }
        )
        findings.append(
            {
                "source": path.relative_to(ROOT).as_posix(),
                "scan_method": "pickletools opcode string scan; no unpickling",
                "module_strings": sorted(module_strings),
                "modules": referenced,
            }
        )
    return findings


def _qualname(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualname(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _qualname(node.value)
    return ""


def _symbol(
    trees: dict[str, ast.Module], module: str, class_name: str | None, name: str
) -> ast.AST:
    body: Iterable[ast.stmt] = trees[module].body
    if class_name is not None:
        class_node = next(
            (
                item for item in trees[module].body
                if isinstance(item, ast.ClassDef) and item.name == class_name
            ),
            None,
        )
        if class_node is None:
            raise AssertionError(f"missing {module}.{class_name}")
        body = class_node.body
    result = next(
        (
            item for item in body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == name
        ),
        None,
    )
    if result is None:
        owner = f"{module}.{class_name}" if class_name else module
        raise AssertionError(f"missing {owner}.{name}")
    return result


def _call(node: ast.AST, suffix: str) -> ast.Call:
    result = next(
        (
            item for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and (
                _qualname(item.func) == suffix
                or _qualname(item.func).endswith("." + suffix)
            )
        ),
        None,
    )
    if result is None:
        raise AssertionError(f"missing call to {suffix} at line {getattr(node, 'lineno', '?')}")
    return result


def _evidence(module: str, node: ast.AST, symbol: str) -> dict[str, object]:
    return {
        "path": f"lrom/{module}.py",
        "line": node.lineno,
        "symbol": symbol,
    }


def _build_workflow(trees: dict[str, ast.Module]) -> dict[str, object]:
    input_sample = _symbol(trees, "data", "InputSpace", "sample")
    generate = _symbol(trees, "problem", "ScatteringProblem", "generate_training_data")
    fom_solve = _symbol(trees, "fom", "NumerovFOM", "solve")
    training_data = next(
        item for item in trees["data"].body
        if isinstance(item, ast.ClassDef) and item.name == "TrainingData"
    )
    lrom_init = _symbol(trees, "emulator", "ScatteringLROM", "__init__")
    train = _symbol(trees, "emulator", "ScatteringLROM", "train")
    packed_build = _symbol(trees, "emulator", "ScatteringLROM", "_build_online_model")
    learned_fit = _symbol(trees, "reduced", "LearnedROM", "fit")
    get_runtime = _symbol(trees, "backends", None, "get_runtime")
    runtime_class = next(
        item for item in trees["backends"].body
        if isinstance(item, ast.ClassDef) and item.name == "Runtime"
    )
    runtime_solve = _symbol(trees, "backends", "Runtime", "solve")
    router_init = _symbol(trees, "deployment", "HardRoutedScatteringLROM", "__init__")
    router_cross = _symbol(trees, "deployment", "HardRoutedScatteringLROM", "cross_section")
    router_route = _symbol(trees, "deployment", "HardRoutedScatteringLROM", "route_energy")
    router_batch = _symbol(
        trees, "deployment", "HardRoutedScatteringLROM", "partial_wave_s_matrices"
    )
    lrom_cross = _symbol(trees, "emulator", "ScatteringLROM", "cross_section")
    lrom_fast = _symbol(trees, "emulator", "ScatteringLROM", "_fast_cross_section")
    packed_batch = _symbol(trees, "emulator", "_PackedOnlineModel", "s_matrices_batch")
    packed_batch_solve = _symbol(
        trees, "emulator", "_PackedOnlineModel", "s_matrices_from_feature_batch"
    )

    generate_solve_call = _call(generate, "solve")
    generate_data_call = _call(generate, "TrainingData")
    train_fit_call = _call(train, "LearnedROM.fit")
    train_pack_call = _call(train, "_build_online_model")
    runtime_construct_call = _call(get_runtime, "Runtime")
    router_route_call = _call(router_cross, "route_energy")
    router_cross_call = _call(router_cross, "cross_section")
    lrom_fast_call = _call(lrom_cross, "_fast_cross_section")
    packed_s_call = _call(lrom_fast, "s_matrices")
    kernel_build_call = _call(lrom_fast, "ElasticCrossSectionKernel.build")
    kernel_evaluate_call = _call(lrom_fast, "evaluate")
    router_batch_call = _call(router_batch, "s_matrices_batch")
    packed_solver_call = _call(packed_batch_solve, "linear_solver")

    lrom_class = next(
        item for item in trees["emulator"].body
        if isinstance(item, ast.ClassDef) and item.name == "ScatteringLROM"
    )
    lrom_methods = {
        item.name for item in lrom_class.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    router_cross_names = {_qualname(item) for item in ast.walk(router_cross)}
    router_batch_names = {_qualname(item) for item in ast.walk(router_batch)}

    def step(
        source: str,
        target: str,
        message: str,
        evidence: list[dict[str, object]],
        kind: str = "call",
    ) -> dict[str, object]:
        return {
            "from": source,
            "to": target,
            "message": message,
            "kind": kind,
            "evidence": evidence,
        }

    groups = [
        {
            "label": "Offline data generation and fitting",
            "optional": False,
            "steps": [
                step("caller", "input", "sample(count, seed)", [_evidence("data", input_sample, "InputSpace.sample")]),
                step("input", "caller", "named samples", [_evidence("data", input_sample, "InputSpace.sample")], "return"),
                step("caller", "problem", "generate_training_data(samples)", [_evidence("problem", generate, "ScatteringProblem.generate_training_data")]),
                step("problem", "fom", "solve each channel and sample", [_evidence("problem", generate_solve_call, "ScatteringProblem.generate_training_data -> solve"), _evidence("fom", fom_solve, "NumerovFOM.solve")]),
                step("problem", "training", "construct TrainingData", [_evidence("problem", generate_data_call, "TrainingData construction"), _evidence("data", training_data, "TrainingData")]),
                step("training", "caller", "training database", [_evidence("data", training_data, "TrainingData")], "return"),
                step("caller", "lrom", "construct ScatteringLROM(problem, basis, predictors)", [_evidence("emulator", lrom_init, "ScatteringLROM.__init__")]),
                step("caller", "lrom", "train(training_data)", [_evidence("emulator", train, "ScatteringLROM.train")]),
                step("lrom", "learned", "fit one learned equation per channel", [_evidence("emulator", train_fit_call, "ScatteringLROM.train -> LearnedROM.fit"), _evidence("reduced", learned_fit, "LearnedROM.fit")]),
                step("lrom", "packed", "build packed online model", [_evidence("emulator", train_pack_call, "ScatteringLROM.train -> _build_online_model"), _evidence("emulator", packed_build, "ScatteringLROM._build_online_model")]),
            ],
        },
        {
            "label": "Optional runtime selection before deployment",
            "optional": True,
            "steps": [
                step("caller", "backends", "get_runtime(backend, precision)", [_evidence("backends", get_runtime, "get_runtime")]),
                step("backends", "runtime", "construct Runtime", [_evidence("backends", runtime_construct_call, "get_runtime -> Runtime"), _evidence("backends", runtime_class, "Runtime")]),
                step("runtime", "caller", "selected runtime", [_evidence("backends", runtime_class, "Runtime")], "return"),
            ],
        },
        {
            "label": "Deployment setup",
            "optional": False,
            "steps": [
                step("caller", "router", "construct with trained models, energy edges, angles, optional Runtime", [_evidence("deployment", router_init, "HardRoutedScatteringLROM.__init__")]),
            ],
        },
        {
            "label": "Scalar routed cross section",
            "optional": False,
            "steps": [
                step("caller", "router", "cross_section(sample)", [_evidence("deployment", router_cross, "HardRoutedScatteringLROM.cross_section")]),
                step("router", "router", "route_energy(sample E)", [_evidence("deployment", router_route_call, "cross_section -> route_energy"), _evidence("deployment", router_route, "route_energy")]),
                step("router", "lrom", "cross_section(sample, angles)", [_evidence("deployment", router_cross_call, "router -> ScatteringLROM.cross_section"), _evidence("emulator", lrom_cross, "ScatteringLROM.cross_section")]),
                step("lrom", "lrom", "_fast_cross_section", [_evidence("emulator", lrom_fast_call, "cross_section -> _fast_cross_section"), _evidence("emulator", lrom_fast, "ScatteringLROM._fast_cross_section")]),
                step("lrom", "packed", "s_matrices(problem, sample, controls)", [_evidence("emulator", packed_s_call, "_fast_cross_section -> packed.s_matrices")]),
                step("lrom", "kernel", "build once per angle grid, then evaluate", [_evidence("emulator", kernel_build_call, "ElasticCrossSectionKernel.build"), _evidence("emulator", kernel_evaluate_call, "ElasticCrossSectionKernel.evaluate")]),
                step("kernel", "caller", "differential cross section", [_evidence("emulator", kernel_evaluate_call, "cross-section return")], "return"),
            ],
        },
        {
            "label": "Optional Runtime-backed partial-wave batch",
            "optional": True,
            "steps": [
                step("caller", "router", "partial_wave_s_matrices(samples)", [_evidence("deployment", router_batch, "HardRoutedScatteringLROM.partial_wave_s_matrices")]),
                step("router", "packed", "s_matrices_batch with Runtime.solve", [_evidence("deployment", router_batch_call, "partial_wave_s_matrices -> s_matrices_batch")]),
                step("packed", "runtime", "solve reduced systems", [_evidence("emulator", packed_solver_call, "s_matrices_batch -> linear_solver"), _evidence("backends", runtime_solve, "Runtime.solve")]),
                step("router", "caller", "routed partial-wave S matrices", [_evidence("deployment", router_batch, "partial_wave_s_matrices")], "return"),
            ],
        },
    ]

    return {
        "participants": [
            {"id": "caller", "label": "Developer / caller"},
            {"id": "input", "label": "InputSpace"},
            {"id": "problem", "label": "ScatteringProblem"},
            {"id": "fom", "label": "NumerovFOM"},
            {"id": "training", "label": "TrainingData"},
            {"id": "lrom", "label": "ScatteringLROM"},
            {"id": "learned", "label": "LearnedROM"},
            {"id": "backends", "label": "backends.get_runtime"},
            {"id": "runtime", "label": "Runtime"},
            {"id": "router", "label": "HardRoutedScatteringLROM"},
            {"id": "packed", "label": "Packed online model"},
            {"id": "kernel", "label": "ElasticCrossSectionKernel"},
        ],
        "groups": groups,
        "observations": {
            "scattering_lrom_build_present": "build" in lrom_methods,
            "construction_api": "ScatteringLROM.__init__ followed by ScatteringLROM.train",
            "scalar_cross_section_consults_runtime": any(
                name.endswith("linear_solver") or name.endswith("Runtime")
                for name in router_cross_names
            ),
            "partial_wave_batch_consults_runtime": (
                any(name.endswith("linear_solver.solve") for name in router_batch_names)
                and _qualname(packed_solver_call.func) == "linear_solver"
            ),
            "evidence": [
                _evidence("emulator", lrom_class, "ScatteringLROM methods"),
                _evidence("deployment", router_cross, "HardRoutedScatteringLROM.cross_section"),
                _evidence("deployment", router_batch, "HardRoutedScatteringLROM.partial_wave_s_matrices"),
                _evidence("emulator", packed_batch, "_PackedOnlineModel.s_matrices_batch"),
                _evidence("emulator", packed_batch_solve, "_PackedOnlineModel.s_matrices_from_feature_batch"),
            ],
        },
    }


def build_architecture() -> dict[str, object]:
    metadata, distributions = _project_metadata()
    paths = sorted(LROM_DIR.glob("*.py"))
    modules = {path.stem for path in paths}
    sources: dict[str, str] = {}
    trees: dict[str, ast.Module] = {}
    roles: dict[str, str] = {}
    role_sources: dict[str, str] = {}
    edge_types: dict[tuple[str, str], str] = {}
    module_external: dict[str, dict[str, str]] = {module: {} for module in modules}

    for path in paths:
        module = path.stem
        source, tree = _source_tree(path)
        sources[module] = source
        trees[module] = tree
        docstring = ast.get_docstring(tree, clean=False)
        if docstring:
            roles[module] = next(
                line.strip() for line in docstring.splitlines() if line.strip()
            )
            role_sources[module] = "module docstring first line"
        else:
            roles[module] = MISSING_ROLE
            role_sources[module] = "missing module docstring"

        for record in _import_records(tree):
            target = _internal_target(record, modules)
            if target is not None:
                key = (module, target)
                edge_types[key] = _merge_import_type(
                    edge_types.get(key), record.import_type
                )
                continue
            root = _external_root(record)
            if (
                root is None
                or root == "lrom"
                or root in sys.stdlib_module_names
                or root in modules
            ):
                continue
            module_external[module][root] = _merge_import_type(
                module_external[module].get(root), record.import_type
            )

    edges = [
        {"source": source, "target": target, "type": edge_types[(source, target)]}
        for source, target in sorted(edge_types)
    ]
    adjacency = {module: set() for module in modules}
    incoming = {module: set() for module in modules}
    for edge in edges:
        adjacency[edge["source"]].add(edge["target"])
        incoming[edge["target"]].add(edge["source"])

    cycles = _cycles(modules, adjacency)
    assert not cycles, f"internal import graph must remain a DAG; found {cycles}"
    layers = _layers(modules, adjacency)
    assert all(
        layers[edge["source"]] > layers[edge["target"]] for edge in edges
    ), "derived layers must place every dependency below its importer"

    consumer_edges, consumer_external, workspace_edges, consumer_groups = (
        _scan_consumers(modules, distributions)
    )
    pickle_references = _scan_pickles(modules)
    source_referenced = {edge["target"] for edge in consumer_edges}
    pickle_referenced = {
        module
        for finding in pickle_references
        for module in finding["modules"]
    }

    nodes = []
    for module in sorted(modules):
        external_dependencies = [
            _external_dependency(root, import_type, distributions)
            for root, import_type in sorted(module_external[module].items())
        ]
        nodes.append(
            {
                "module": module,
                "path": f"lrom/{module}.py",
                "loc": len(sources[module].splitlines()),
                "role": roles[module],
                "role_source": role_sources[module],
                "layer": layers[module],
                "in_degree": len(incoming[module]),
                "out_degree": len(adjacency[module]),
                "external_dependencies": external_dependencies,
            }
        )

    unreferenced_modules = sorted(
        module for module in modules
        if module != "__init__"
        and not incoming[module]
        and module not in source_referenced
        and module not in pickle_referenced
    )
    total_loc = sum(node["loc"] for node in nodes)
    result = {
        "schema_version": 1,
        "metadata": {
            **metadata,
            "source_root": "lrom",
            "generator": "tools/architecture_graph.py",
            "renderer": "tools/render_architecture.py",
            "evidence_file": "docs/architecture.json",
        },
        "summary": {
            "module_count": len(nodes),
            "total_loc": total_loc,
            "internal_edge_count": len(edges),
            "maximum_layer": max(layers.values(), default=0),
            "consumer_edge_count": len(consumer_edges),
            "pickle_file_count": len(pickle_references),
            "unreferenced_modules": unreferenced_modules,
        },
        "nodes": nodes,
        "edges": edges,
        "consumer_groups": consumer_groups,
        "consumer_edges": consumer_edges,
        "consumer_external_dependencies": consumer_external,
        "workspace_edges": workspace_edges,
        "pickle_references": pickle_references,
        "workflow": _build_workflow(trees),
        "cycles": cycles,
    }
    return result


def main() -> None:
    architecture = build_architecture()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(architecture, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
