#!/usr/bin/env python3
"""Render the architecture evidence file as GitHub-native Mermaid Markdown."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "docs" / "architecture.json"
OUTPUT_PATH = ROOT / "docs" / "ARCHITECTURE.md"


LAYER_COLORS = (
    ("#DBEAFE", "#111827", "#2563EB"),
    ("#DCFCE7", "#111827", "#16A34A"),
    ("#FEF3C7", "#111827", "#D97706"),
    ("#FEE2E2", "#111827", "#DC2626"),
    ("#EDE9FE", "#111827", "#7C3AED"),
    ("#CFFAFE", "#111827", "#0891B2"),
    ("#FCE7F3", "#111827", "#DB2777"),
)


def _identifier(prefix: str, value: str) -> str:
    return prefix + re.sub(r"[^A-Za-z0-9_]", "_", value)


def _label(value: object) -> str:
    return html.escape(str(value), quote=True)


def _markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or singular + "s")


def _aggregate_edge(
    edges: dict[tuple[str, str], dict[str, set[str]]],
    source: str,
    target: str,
    import_type: str,
    context: str,
) -> None:
    entry = edges.setdefault((source, target), {"types": set(), "contexts": set()})
    entry["types"].add(import_type)
    entry["contexts"].add(context)


def _mermaid_edge(
    source: str,
    target: str,
    types: set[str],
    contexts: set[str],
) -> str:
    if contexts == {"embedded-python"}:
        return f"    {source} -. embedded code .-> {target}"
    if types == {"dynamic"}:
        return f"    {source} -. dynamic/optional .-> {target}"
    return f"    {source} --> {target}"


def _context_diagram(data: dict[str, object]) -> str:
    groups = {
        item["category"]: item
        for item in data["consumer_groups"]
    }
    external_nodes: dict[str, str] = {}
    aggregated: dict[tuple[str, str], dict[str, set[str]]] = {}

    for node in data["nodes"]:
        for dependency in node["external_dependencies"]:
            import_name = dependency["import_name"]
            external_nodes[import_name] = dependency["display_name"]
            _aggregate_edge(
                aggregated,
                "package",
                _identifier("x_", import_name),
                dependency["type"],
                "module-source",
            )
    for dependency in data["consumer_external_dependencies"]:
        import_name = dependency["import_name"]
        external_nodes[import_name] = dependency["display_name"]
        _aggregate_edge(
            aggregated,
            _identifier("c_", dependency["category"]),
            _identifier("x_", import_name),
            dependency["type"],
            dependency["context"],
        )
    for edge in data["consumer_edges"]:
        _aggregate_edge(
            aggregated,
            _identifier("c_", edge["category"]),
            "package",
            edge["type"],
            edge["context"],
        )

    node_count = 1 + len(groups) + len(external_nodes)
    if node_count > 20:
        raise ValueError(
            f"context diagram has {node_count} nodes; collapse dependencies before rendering"
        )

    summary = data["summary"]
    metadata = data["metadata"]
    lines = ["```mermaid", "flowchart TB", "    subgraph consumers[\"Consumers and adjacent workspace code\"]", "        direction LR"]
    for category in sorted(groups):
        group = groups[category]
        file_count = group["file_count"]
        module_count = len(group["lrom_modules"])
        if module_count:
            role = (
                f"{file_count} scanned {_plural(file_count, 'file')} · "
                f"imports {module_count} lrom {_plural(module_count, 'module')}"
            )
        else:
            role = f"{file_count} scanned {_plural(file_count, 'file')} · no direct lrom import"
        lines.append(
            f'        {_identifier("c_", category)}["{_label(category)}/<br/>{_label(role)}"]'
        )
    lines.extend(
        [
            "    end",
            (
                f'    package["lrom<br/>{_label(metadata["description"])}<br/>'
                f'{summary["module_count"]} modules · {summary["total_loc"]} LOC"]'
            ),
            "    subgraph external[\"External dependencies found in imports\"]",
            "        direction LR",
        ]
    )
    for import_name in sorted(external_nodes):
        display_name = external_nodes[import_name]
        uses = {
            source for (source, target) in aggregated
            if target == _identifier("x_", import_name)
        }
        role = f"imported by {len(uses)} {_plural(len(uses), 'area')}"
        lines.append(
            f'        {_identifier("x_", import_name)}["{_label(display_name)}<br/>{_label(role)}"]'
        )
    lines.extend(["    end"])
    for (source, target), attributes in sorted(aggregated.items()):
        lines.append(
            _mermaid_edge(
                source, target, attributes["types"], attributes["contexts"]
            )
        )
    lines.extend(
        [
            "    classDef consumer fill:#DBEAFE,color:#111827,stroke:#2563EB,stroke-width:1px",
            "    classDef package fill:#DCFCE7,color:#111827,stroke:#16A34A,stroke-width:2px",
            "    classDef external fill:#FEF3C7,color:#111827,stroke:#D97706,stroke-width:1px",
            "    class package package",
            "    class " + ",".join(_identifier("c_", name) for name in sorted(groups)) + " consumer",
            "    class " + ",".join(_identifier("x_", name) for name in sorted(external_nodes)) + " external",
            "```",
        ]
    )
    return "\n".join(lines)


def _layer_diagram(data: dict[str, object]) -> str:
    nodes = data["nodes"]
    if len(nodes) > 20:
        raise ValueError(
            f"layer diagram has {len(nodes)} nodes; collapse subgraphs before rendering"
        )
    by_layer: dict[int, list[dict[str, object]]] = {}
    for node in nodes:
        by_layer.setdefault(node["layer"], []).append(node)
    lines = ["```mermaid", "flowchart TB"]
    for layer in sorted(by_layer, reverse=True):
        lines.extend(
            [
                f'    subgraph group_{layer}["Layer {layer}"]',
                "        direction LR",
            ]
        )
        for node in sorted(by_layer[layer], key=lambda item: item["module"]):
            node_id = _identifier("m_", node["module"])
            node_label = (
                f'{_label(node["module"])}<br/>'
                f'{_label(str(node["role"]).replace("`", ""))}<br/>'
                f'{node["loc"]} LOC'
            )
            lines.append(f'        {node_id}["{node_label}"]')
        lines.append("    end")
    for edge in sorted(
        data["edges"], key=lambda item: (item["source"], item["target"], item["type"])
    ):
        source = _identifier("m_", edge["source"])
        target = _identifier("m_", edge["target"])
        if edge["type"] == "dynamic":
            lines.append(f"    {source} -. dynamic/optional .-> {target}")
        else:
            lines.append(f"    {source} --> {target}")
    for layer in sorted(by_layer):
        fill, color, stroke = LAYER_COLORS[layer % len(LAYER_COLORS)]
        lines.append(
            f"    classDef layer{layer} fill:{fill},color:{color},stroke:{stroke},stroke-width:1px"
        )
        members = ",".join(
            _identifier("m_", node["module"])
            for node in sorted(by_layer[layer], key=lambda item: item["module"])
        )
        lines.append(f"    class {members} layer{layer}")
    lines.append("```")
    return "\n".join(lines)


def _sequence_diagram(data: dict[str, object]) -> str:
    workflow = data["workflow"]
    participants = workflow["participants"]
    if len(participants) > 20:
        raise ValueError(f"sequence diagram has {len(participants)} participants")
    lines = ["```mermaid", "sequenceDiagram", "    autonumber"]
    for participant in participants:
        kind = "actor" if participant["id"] == "caller" else "participant"
        lines.append(
            f'    {kind} {participant["id"]} as {_label(participant["label"])}'
        )
    for group in workflow["groups"]:
        if group["optional"]:
            lines.append(f'    opt {_label(group["label"])}')
        else:
            participant_order = {
                participant["id"]: index
                for index, participant in enumerate(participants)
            }
            touched = {
                endpoint
                for step in group["steps"]
                for endpoint in (step["from"], step["to"])
            }
            first = min(touched, key=lambda item: participant_order[item])
            last = max(touched, key=lambda item: participant_order[item])
            lines.append(f'    Note over {first},{last}: {_label(group["label"])}')
        for step in group["steps"]:
            arrow = "-->>" if step["kind"] == "return" else "->>"
            lines.append(
                f'    {step["from"]}{arrow}{step["to"]}: {_label(step["message"])}'
            )
        if group["optional"]:
            lines.append("    end")
    lines.append("```")
    return "\n".join(lines)


def _module_table(data: dict[str, object]) -> str:
    dependencies: dict[str, list[str]] = {node["module"]: [] for node in data["nodes"]}
    for edge in data["edges"]:
        suffix = " (dynamic)" if edge["type"] == "dynamic" else ""
        dependencies[edge["source"]].append(edge["target"] + suffix)
    rows = [
        "| Layer | Module | Role from module docstring | LOC | In | Out | Internal dependencies | External dependencies |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ]
    for node in sorted(data["nodes"], key=lambda item: (-item["layer"], item["module"])):
        external = ", ".join(
            dependency["display_name"]
            + (" (dynamic)" if dependency["type"] == "dynamic" else "")
            for dependency in node["external_dependencies"]
        ) or "—"
        internal = ", ".join(sorted(dependencies[node["module"]])) or "—"
        rows.append(
            "| {layer} | `{module}` | {role} | {loc} | {in_degree} | {out_degree} | {internal} | {external} |".format(
                layer=node["layer"],
                module=_markdown(node["module"]),
                role=_markdown(node["role"]),
                loc=node["loc"],
                in_degree=node["in_degree"],
                out_degree=node["out_degree"],
                internal=_markdown(internal),
                external=_markdown(external),
            )
        )
    return "\n".join(rows)


def _caveats(data: dict[str, object]) -> str:
    node_map = {node["module"]: node for node in data["nodes"]}
    backends_dependencies = {
        edge["target"] for edge in data["edges"] if edge["source"] == "backends"
    }
    jax = next(
        dependency for dependency in node_map["backends"]["external_dependencies"]
        if dependency["import_name"] == "jax"
    )
    curated_edges = [
        edge for edge in data["consumer_edges"] if edge["target"] == "curated_data"
    ]
    curated_locations = ", ".join(
        f'`{edge["source"]}` line {edge["line"]}'
        + (" (embedded Python)" if edge["context"] == "embedded-python" else "")
        for edge in curated_edges
    )
    unreferenced = data["summary"]["unreferenced_modules"]
    unreferenced_loc = sum(node_map[module]["loc"] for module in unreferenced)
    pickle_modules = sorted(
        {
            module
            for finding in data["pickle_references"]
            for module in finding["modules"]
        }
    )
    pickle_text = ", ".join(f"`{module}`" for module in pickle_modules) or "none"
    assert not backends_dependencies.intersection({"cpu_batched", "cuda_batched"})
    assert jax["type"] == "dynamic"
    return "\n".join(
        [
            "- **Dynamic backend edge.** `backends` has no internal import edge to `cpu_batched` or `cuda_batched`. Its `jax` dependency is marked dynamic because the import is function-local. This is an absent source edge, not permission to remove either helper. *(Evidence: `nodes[module=backends].external_dependencies`, `edges`.)*",
            f"- **Consumer-only curated data.** `curated_data` has internal in-degree {node_map['curated_data']['in_degree']}, but it is imported at {curated_locations}. It is notebook-facing rather than dead. *(Evidence: `nodes[module=curated_data].in_degree`, `consumer_edges`.)*",
            f"- **Currently unreferenced source modules.** {', '.join(f'`{module}`' for module in unreferenced)} total {unreferenced_loc} LOC and have no inbound internal import, scanned consumer import, or module-like string in the {data['summary']['pickle_file_count']} inspected model pickles. The safe opcode scan found pickle references to {pickle_text}; it never unpickled the artifacts. Static evidence cannot rule out callers outside the scanned tree. *(Evidence: `summary.unreferenced_modules`, `consumer_edges`, `pickle_references`.)*",
        ]
    )


def _recommendations(data: dict[str, object]) -> str:
    nodes = {node["module"]: node for node in data["nodes"]}
    total_loc = data["summary"]["total_loc"]
    emulator = nodes["emulator"]
    percentage = 100.0 * emulator["loc"] / total_loc
    dependency_sets: dict[str, set[str]] = {module: set() for module in nodes}
    for edge in data["edges"]:
        dependency_sets[edge["source"]].add(edge["target"])
    overlap = sorted(dependency_sets["emulator"].intersection(dependency_sets["problem"]))
    unreferenced = data["summary"]["unreferenced_modules"]
    return "\n".join(
        [
            "These observations are non-binding. No package change was made from them.",
            "",
            f"1. **High confidence on strength and mass; no volatility evidence.** `emulator` is {emulator['loc']:,} of {total_loc:,} physical lines ({percentage:.1f}%) and directly imports {emulator['out_degree']} internal modules, the strongest implementation-module fan-out in this graph. That is a concentration worth watching, but this snapshot contains no change-history evidence about volatility, so it does not by itself justify a split. *(Evidence: `summary.total_loc`, `nodes[module=emulator]`, `edges`.)*",
            f"2. **High confidence on distance and overlap; low confidence on defect risk.** `emulator` directly depends on `problem` (distance 1), and the two share {len(overlap)} direct dependencies: {', '.join(f'`{module}`' for module in overlap)}. Before treating that overlap as undesirable coupling, compare their change history to learn whether the relationship is stable or volatile. *(Evidence: `edges` for sources `emulator` and `problem`.)*",
            f"3. **High confidence for this repository snapshot; medium confidence beyond it.** {', '.join(f'`{module}`' for module in unreferenced)} are unreferenced by the scanned source and model opcodes. Document their intended entry points or add integration evidence before any future retention or removal decision. *(Evidence: `summary.unreferenced_modules`, `consumer_edges`, `pickle_references`.)*",
        ]
    )


def render(data: dict[str, object]) -> str:
    summary = data["summary"]
    workflow = data["workflow"]
    observations = workflow["observations"]
    if data["cycles"]:
        raise ValueError(f"cannot render cyclic architecture: {data['cycles']}")
    if observations["scattering_lrom_build_present"]:
        construction_sentence = "The current API includes `ScatteringLROM.build`."
    else:
        construction_sentence = (
            "The current API constructs `ScatteringLROM` directly and then calls `train`; "
            "there is no `ScatteringLROM.build` method."
        )
    runtime_sentence = (
        "The scalar routed `cross_section` path does not consult `Runtime`; the optional "
        "runtime enters the routed partial-wave batch path."
    )
    assert observations["scalar_cross_section_consults_runtime"] is False
    assert observations["partial_wave_batch_consults_runtime"] is True

    sections = [
        "<!-- Generated by tools/render_architecture.py from docs/architecture.json. Do not edit by hand. -->",
        "# `lrom` architecture",
        "",
        (
            f"This source-derived snapshot covers {summary['module_count']} modules, "
            f"{summary['total_loc']:,} physical lines, and {summary['internal_edge_count']} "
            "internal import edges. The internal graph is a DAG. "
            "[Evidence: `architecture.json`](architecture.json) (`summary`, `cycles`)."
        ),
        "",
        "Regenerate it from the package root:",
        "",
        "```bash",
        "python tools/architecture_graph.py",
        "python tools/render_architecture.py",
        "python tools/render_architecture.py --check",
        "```",
        "",
        "The JSON is the evidence; this Markdown is a rendering of it. Solid arrows are static imports or calls. Dashed arrows are dynamic/optional imports or Python embedded in a generator. *(Evidence: `nodes`, `edges`, `consumer_edges`, `workflow`.)*",
        "",
        "## View 1 — Context: what is this, and who calls it?",
        "",
        "Consumer and dependency nodes are derived from imports found under `notebooks/`, `tools/`, `tests/`, `benchmarks/`, and `lrom/`. `benchmarks/` is shown even though it has no direct `lrom` import; its external-reference role remains visible without inventing an edge. *(Evidence: `consumer_groups`, `consumer_edges`, `consumer_external_dependencies`.)*",
        "",
        _context_diagram(data),
        "",
        "## View 2 — Layers: what depends on what?",
        "",
        "Arrows point from importer to dependency, always downward. Layer 0 modules have no internal dependencies; every higher index is one plus the longest dependency path below that module. *(Evidence: `nodes[].layer`, `edges`, `cycles`.)*",
        "",
        _layer_diagram(data),
        "",
        "## View 3 — Flow: how does a call move?",
        "",
        f"{construction_sentence} {runtime_sentence} *(Evidence: `workflow.groups`, `workflow.observations`.)*",
        "",
        _sequence_diagram(data),
        "",
        "## Module inventory",
        "",
        "The table is sorted from the highest derived layer to the leaves. In/out degrees count internal import edges only. *(Evidence: `nodes`, `edges`.)*",
        "",
        _module_table(data),
        "",
        "## Caveats",
        "",
        _caveats(data),
        "",
        "## Recommendations (non-binding, not acted on)",
        "",
        _recommendations(data),
        "",
    ]
    return "\n".join(sections)


def _load_evidence() -> dict[str, object]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _check(rendered: str) -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".architecture-",
            suffix=".md",
            dir=OUTPUT_PATH.parent,
            delete=False,
        ) as stream:
            stream.write(rendered)
            temporary_path = Path(stream.name)
        if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_bytes() != temporary_path.read_bytes():
            print(
                "docs/ARCHITECTURE.md is stale; run python tools/render_architecture.py",
                file=sys.stderr,
            )
            return 1
        print("docs/ARCHITECTURE.md is current")
        return 0
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if docs/ARCHITECTURE.md differs from architecture.json",
    )
    arguments = parser.parse_args()
    rendered = render(_load_evidence())
    if arguments.check:
        return _check(rendered)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
