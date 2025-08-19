from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml

Parameters = dict[str, list[Any]]
Combination = dict[str, Any]

DEBUG = False  # Global debug flag


def debug_print(*args: Any) -> None:
    if DEBUG:
        print("[DEBUG]", *args)


def flatten_single_combinations(node: Combination) -> Combination:
    """Flatten nodes where there is only one combination."""
    while "combinations" in node and len(node["combinations"]) == 1:
        child = node["combinations"][0]
        merged_params: Parameters = {**node.get("parameters", {}), **child.get("parameters", {})}
        node = {**child, "parameters": merged_params}
    if "combinations" in node:
        node["combinations"] = [flatten_single_combinations(c) for c in node["combinations"]]
    return node


def parameters_equal(p1: Parameters, p2: Parameters) -> bool:
    """Check if two parameter dicts have identical keys and values."""
    if set(p1.keys()) != set(p2.keys()):
        return False
    for k in p1:
        if sorted(p1[k]) != sorted(p2[k]):
            return False
    return True


def merge_siblings(combos: list[Combination]) -> list[Combination]:
    """Merge sibling combinations conservatively, but allow merging if nested combos are identical."""
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
            if {k: v for k, v in c1.items() if k not in ("parameters", "combinations")} != {
                k: v for k, v in c2.items() if k not in ("parameters", "combinations")
            }:
                debug_print(f"Not merging index {i} and {j}: non-parameter fields differ")
                continue

            # Parameter keys must match exactly
            if set(c1.get("parameters", {}).keys()) != set(c2.get("parameters", {}).keys()):
                debug_print(f"Not merging index {i} and {j}: parameter keys differ")
                continue

            # If both have combinations, check if they are identical after simplification
            if "combinations" in c1 and "combinations" in c2:
                if yaml.safe_dump(c1["combinations"], sort_keys=True) != yaml.safe_dump(
                    c2["combinations"], sort_keys=True
                ):
                    debug_print(f"Not merging index {i} and {j}: nested combinations differ")
                    continue
                debug_print(f"Merging index {i} and {j}: identical nested combinations")
            elif "combinations" in c1 or "combinations" in c2:
                debug_print(f"Not merging index {i} and {j}: one has combinations, other doesn't")
                continue
            else:
                debug_print(f"Merging index {i} and {j}: both are leaves")

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


def normalize_parameter_order(node: Combination) -> Combination:
    """Normalize parameter nesting order."""
    if "combinations" in node:
        node["combinations"] = sorted(
            (normalize_parameter_order(c) for c in node["combinations"]),
            key=lambda c: tuple(sorted(c.get("parameters", {}).keys())),
        )
    return node


def simplify(node: Combination, aggressive: bool = False) -> Combination:
    """Simplify recursively, preserving necessary nesting."""
    node = flatten_single_combinations(node)
    if aggressive:
        node = normalize_parameter_order(node)
    if "combinations" in node:
        node["combinations"] = [simplify(c, aggressive=aggressive) for c in node["combinations"]]
        node["combinations"] = merge_siblings(node["combinations"])
    return node


def simplify_yaml(data: list[Combination], aggressive: bool = False) -> list[Combination]:
    return [simplify(item, aggressive=aggressive) for item in data]


def main() -> None:
    import sys

    global DEBUG

    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.yaml output.yaml [--aggressive] [--debug]")
        sys.exit(1)

    aggressive = "--aggressive" in sys.argv
    if aggressive:
        sys.argv.remove("--aggressive")

    DEBUG = "--debug" in sys.argv
    if DEBUG:
        sys.argv.remove("--debug")

    with open(sys.argv[1], encoding="utf-8") as f:
        data: list[Combination] = yaml.safe_load(f)

    simplified = simplify_yaml(data, aggressive=aggressive)

    with open(sys.argv[2], "w", encoding="utf-8") as f:
        yaml.safe_dump(simplified, f, sort_keys=False)


if __name__ == "__main__":
    main()
