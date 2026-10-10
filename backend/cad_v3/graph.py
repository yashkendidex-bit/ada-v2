"""Mate-graph utilities. No networkx dependency — this is 60 lines of algorithm.

The mate graph is what replaces coordinates. If a part has no path back to the
datum, it has no defined position; that is the "bolt floating in mid-air" bug,
and it is detected here at spec time rather than discovered in the viewer.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable


def components(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[set[str]]:
    """Connected components via union-find."""
    parent: dict[str, str] = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a not in parent or b not in parent:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: dict[str, set[str]] = {}
    for n in parent:
        groups.setdefault(find(n), set()).add(n)
    return list(groups.values())


def unreachable(datum: str, nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[str]:
    """Nodes with no undirected path to `datum` — i.e. parts with no position."""
    nodes = list(nodes)
    if datum not in nodes:
        return sorted(nodes)
    for comp in components(nodes, edges):
        if datum in comp:
            return sorted(set(nodes) - comp)
    return sorted(nodes)


def mate_order(datum: str, edges: list[tuple[str, str]]) -> list[tuple[str, str, int]]:
    """BFS from the datum outward, orienting each edge parent -> child.

    Returns (parent, child, edge_index) in an order where the parent is always
    already placed. This is what makes assembly deterministic: the traversal
    order, not the order the brain happened to list mates in.
    """
    adj: dict[str, list[tuple[str, int]]] = {}
    for i, (a, b) in enumerate(edges):
        adj.setdefault(a, []).append((b, i))
        adj.setdefault(b, []).append((a, i))

    placed = {datum}
    used: set[int] = set()
    out: list[tuple[str, str, int]] = []
    queue: deque[str] = deque([datum])

    while queue:
        node = queue.popleft()
        for nbr, idx in adj.get(node, []):
            if idx in used:
                continue
            used.add(idx)
            if nbr in placed:
                # Redundant loop closure — the pose is already fully determined.
                out.append((node, nbr, idx))
                continue
            placed.add(nbr)
            out.append((node, nbr, idx))
            queue.append(nbr)

    for i, (a, b) in enumerate(edges):
        if i not in used:
            out.append((a, b, i))
    return out


def redundant_edges(datum: str, edges: list[tuple[str, str]]) -> set[int]:
    """Edge indices that close a loop — informational, not an error.

    Real assemblies are over-constrained on purpose (a bolted flange is mated to
    both the housing and the backplate). We solve the spanning tree and report
    the rest as fit checks instead of trying to satisfy them geometrically.
    """
    seen = {datum}
    keep: set[int] = set()
    for parent, child, idx in mate_order(datum, edges):
        if child in seen and parent in seen:
            keep.add(idx)
        seen.add(child)
        seen.add(parent)
    return keep
