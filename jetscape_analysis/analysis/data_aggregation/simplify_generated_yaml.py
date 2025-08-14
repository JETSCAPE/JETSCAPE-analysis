from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml

Parameters = dict[str, list[Any]]
Combination = dict[str, Any]


def flatten_single_combinations(node: Combination) -> Combination:
    """Flatten nodes where there is only one combination."""
    while "combinations" in node and len(node["combinations"]) == 1:
        child = node["combinations"][0]
        merged_params: Parameters = {**node.get("parameters", {}), **child.get("parameters", {})}
        node = {**child, "parameters": merged_params}
    if "combinations" in node:
        node["combinations"] = [flatten_single_combinations(c) for c in node["combinations"]]
    return node


def merge_siblings(combos: list[Combination]) -> list[Combination]:
    """Merge sibling combinations conservatively."""
    merged: list[Combination] = []
    used = [False] * len(combos)

    for i, c1 in enumerate(combos):
        if used[i]:
            continue
        group = [c1]
        used[i] = True
        for j in range(i + 1, len(combos)):
            if used[j]:
                continue
            c2 = combos[j]
            # Non-parameter fields must match exactly
            if {k: v for k, v in c1.items() if k != "parameters" and k != "combinations"} != {
                k: v for k, v in c2.items() if k != "parameters" and k != "combinations"
            }:
                continue
            # Parameter keys must match exactly
            if set(c1.get("parameters", {}).keys()) != set(c2.get("parameters", {}).keys()):
                continue
            # Only merge if no combinations blocks (leaf nodes)
            if "combinations" in c1 or "combinations" in c2:
                continue
            group.append(c2)
            used[j] = True

        if len(group) == 1:
            merged.append(c1)
        else:
            merged_params = deepcopy(group[0]["parameters"])
            for sibling in group[1:]:
                for k, vals in sibling["parameters"].items():
                    for v in vals:
                        if v not in merged_params[k]:
                            merged_params[k].append(v)
            merged.append({**{k: v for k, v in group[0].items() if k != "parameters"}, "parameters": merged_params})
    return merged


def simplify(node: Combination) -> Combination:
    """Simplify recursively, preserving necessary nesting."""
    node = flatten_single_combinations(node)
    if "combinations" in node:
        node["combinations"] = [simplify(c) for c in node["combinations"]]
        # Try to merge siblings conservatively
        node["combinations"] = merge_siblings(node["combinations"])
    return node


def simplify_yaml(data: list[Combination]) -> list[Combination]:
    return [simplify(item) for item in data]


def main() -> None:
    import sys

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.yaml output.yaml")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        data: list[Combination] = yaml.safe_load(f)

    simplified = simplify_yaml(data)

    with open(sys.argv[2], "w", encoding="utf-8") as f:
        yaml.safe_dump(simplified, f, sort_keys=False)


if __name__ == "__main__":
    main()
