"""Feasibility-aware transportation heuristics for sparse cost matrices.

Blank/NaN cost cells are treated as prohibited routes.  The routines preserve
feasibility after each allocation by checking whether the remaining network
can still satisfy all residual supply and demand.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable

import pandas as pd


METHODS = {
    "NWC": "Restricted North-West Corner",
    "LCM": "Restricted Least Cost Method",
    "VAM": "Restricted Vogel's Approximation Method",
}


def _max_flow_feasible(
    supply: list[int], demand: list[int], allowed: list[list[bool]]
) -> bool:
    """Return True if the allowed bipartite network can meet all residuals."""
    m, n = len(supply), len(demand)
    total_supply = sum(supply)
    if total_supply != sum(demand):
        return False
    if total_supply == 0:
        return True

    # Dinic max-flow: source -> source nodes -> destination nodes -> sink.
    source = 0
    first_row = 1
    first_col = first_row + m
    sink = first_col + n
    node_count = sink + 1
    graph: list[list[list[int]]] = [[] for _ in range(node_count)]

    def add_edge(u: int, v: int, capacity: int) -> None:
        forward = [v, capacity, len(graph[v])]
        reverse = [u, 0, len(graph[u])]
        graph[u].append(forward)
        graph[v].append(reverse)

    for i, value in enumerate(supply):
        if value > 0:
            add_edge(source, first_row + i, value)

    # No route can carry more than total flow.
    for i in range(m):
        if supply[i] <= 0:
            continue
        for j in range(n):
            if demand[j] > 0 and allowed[i][j]:
                add_edge(first_row + i, first_col + j, total_supply)

    for j, value in enumerate(demand):
        if value > 0:
            add_edge(first_col + j, sink, value)

    flow = 0
    while flow < total_supply:
        level = [-1] * node_count
        level[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v, capacity, _ in graph[u]:
                if capacity > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        if level[sink] < 0:
            break

        cursor = [0] * node_count

        def dfs(u: int, pushed: int) -> int:
            if u == sink:
                return pushed
            while cursor[u] < len(graph[u]):
                edge = graph[u][cursor[u]]
                v, capacity, reverse_index = edge
                if capacity > 0 and level[v] == level[u] + 1:
                    amount = dfs(v, min(pushed, capacity))
                    if amount > 0:
                        edge[1] -= amount
                        graph[v][reverse_index][1] += amount
                        return amount
                cursor[u] += 1
            return 0

        while flow < total_supply:
            amount = dfs(source, total_supply - flow)
            if amount == 0:
                break
            flow += amount

    return flow == total_supply


def _max_feasible_allocation(
    i: int,
    j: int,
    supply: list[int],
    demand: list[int],
    allowed: list[list[bool]],
) -> int:
    """Largest integer allocation possible on (i,j) in any feasible solution.

    Feasibility is monotone in a lower bound q: reserve q units on (i,j),
    reduce that row's supply and that column's demand by q, and test whether
    the residual network can still be completed.
    """
    upper = min(supply[i], demand[j])
    if upper <= 0 or not allowed[i][j]:
        return 0

    low = 0
    while low < upper:
        mid = (low + upper + 1) // 2
        residual_supply = supply.copy()
        residual_demand = demand.copy()
        residual_supply[i] -= mid
        residual_demand[j] -= mid
        if _max_flow_feasible(residual_supply, residual_demand, allowed):
            low = mid
        else:
            upper = mid - 1
    return low


def _prepare_inputs(
    supply: pd.Series, demand: pd.Series, costs: pd.DataFrame
) -> tuple[list[str], list[str], list[int], list[int], list[list[float | None]]]:
    if not isinstance(costs, pd.DataFrame):
        raise TypeError("costs harus berupa pandas DataFrame.")

    source_names = [str(x) for x in costs.index]
    destination_names = [str(x) for x in costs.columns]

    supply_series = supply.copy()
    demand_series = demand.copy()
    supply_series.index = supply_series.index.astype(str)
    demand_series.index = demand_series.index.astype(str)

    missing_sources = [name for name in source_names if name not in supply_series.index]
    missing_destinations = [name for name in destination_names if name not in demand_series.index]
    if missing_sources:
        raise ValueError(f"Supply tidak ditemukan untuk sumber: {missing_sources[:5]}")
    if missing_destinations:
        raise ValueError(f"Demand tidak ditemukan untuk tujuan: {missing_destinations[:5]}")

    supply_values = pd.to_numeric(supply_series.reindex(source_names), errors="coerce")
    demand_values = pd.to_numeric(demand_series.reindex(destination_names), errors="coerce")
    if supply_values.isna().any() or demand_values.isna().any():
        raise ValueError("Supply dan demand harus berupa angka dan tidak boleh kosong.")
    if (supply_values < 0).any() or (demand_values < 0).any():
        raise ValueError("Supply dan demand tidak boleh negatif.")

    # The dataset uses kilograms as integer quantities.
    supply_int = [int(round(float(x))) for x in supply_values]
    demand_int = [int(round(float(x))) for x in demand_values]
    if sum(supply_int) != sum(demand_int):
        raise ValueError(
            f"Model belum seimbang: total supply {sum(supply_int):,} kg, "
            f"total demand {sum(demand_int):,} kg."
        )

    aligned = costs.reindex(index=source_names, columns=destination_names)
    aligned = aligned.apply(pd.to_numeric, errors="coerce")
    cost_values: list[list[float | None]] = []
    for row in aligned.itertuples(index=False, name=None):
        cost_values.append([
            None if pd.isna(value) else float(value)
            for value in row
        ])

    return source_names, destination_names, supply_int, demand_int, cost_values


def solve_transportation(
    supply: pd.Series,
    demand: pd.Series,
    costs: pd.DataFrame,
    method: str,
) -> dict:
    """Solve using NWC, LCM, or VAM while respecting prohibited routes.

    This is a feasibility-aware adaptation for sparse transportation tables.
    For a complete cost matrix, the allocation choices reduce to the usual
    NWC, LCM, and VAM rules. Blank costs are never selected.
    """
    method = method.upper().strip()
    if method not in METHODS:
        raise ValueError("method harus salah satu dari: NWC, LCM, VAM.")

    source_names, destination_names, initial_supply, initial_demand, cost_values = (
        _prepare_inputs(supply, demand, costs)
    )
    m, n = len(source_names), len(destination_names)
    remaining_supply = initial_supply.copy()
    remaining_demand = initial_demand.copy()
    allowed = [[cost_values[i][j] is not None for j in range(n)] for i in range(m)]

    if not _max_flow_feasible(remaining_supply, remaining_demand, allowed):
        raise ValueError(
            "Jaringan rute yang memiliki biaya tidak cukup untuk memenuhi seluruh "
            "supply dan demand. Periksa rute kosong atau data supply-demand."
        )

    allocation = [[0 for _ in range(n)] for _ in range(m)]
    steps: list[dict] = []
    cost_order = sorted(
        (cost_values[i][j], i, j)
        for i in range(m)
        for j in range(n)
        if allowed[i][j]
    )

    def commit(i: int, j: int, quantity: int) -> None:
        allocation[i][j] += quantity
        remaining_supply[i] -= quantity
        remaining_demand[j] -= quantity
        unit_cost = cost_values[i][j]
        steps.append({
            "Langkah": len(steps) + 1,
            "Sumber": source_names[i],
            "Tujuan": destination_names[j],
            "Alokasi (kg)": quantity,
            "Biaya (USD/kg)": unit_cost,
            "Subtotal (USD)": quantity * unit_cost,
            "Sisa supply sumber (kg)": remaining_supply[i],
            "Sisa demand tujuan (kg)": remaining_demand[j],
        })
        # The quantity was chosen as the maximum that can be carried by this
        # route in a feasible completion, so the route can now be closed.
        allowed[i][j] = False

    iteration_limit = m * n + 1
    while sum(remaining_supply) > 0:
        if len(steps) > iteration_limit:
            raise RuntimeError("Batas iterasi tercapai; periksa struktur jaringan.")

        selected: tuple[int, int, int] | None = None

        if method == "NWC":
            # Row-major order is the north-west scan order. If a route is
            # blocked or cannot carry flow in any feasible completion, skip it.
            for i in range(m):
                if remaining_supply[i] <= 0:
                    continue
                for j in range(n):
                    if remaining_demand[j] <= 0 or not allowed[i][j]:
                        continue
                    quantity = _max_feasible_allocation(
                        i, j, remaining_supply, remaining_demand, allowed
                    )
                    if quantity > 0:
                        selected = (i, j, quantity)
                        break
                    allowed[i][j] = False
                if selected:
                    break

        elif method == "LCM":
            # Lowest unit cost among routes that can preserve feasibility.
            for _, i, j in cost_order:
                if not allowed[i][j] or remaining_supply[i] <= 0 or remaining_demand[j] <= 0:
                    continue
                quantity = _max_feasible_allocation(
                    i, j, remaining_supply, remaining_demand, allowed
                )
                if quantity > 0:
                    selected = (i, j, quantity)
                    break
                allowed[i][j] = False

        else:  # VAM
            while selected is None:
                candidates: list[tuple[float, float, int, int, int]] = []

                # Row penalties.
                for i in range(m):
                    if remaining_supply[i] <= 0:
                        continue
                    active = [
                        (cost_values[i][j], j)
                        for j in range(n)
                        if allowed[i][j] and remaining_demand[j] > 0
                    ]
                    if not active:
                        continue
                    active.sort(key=lambda item: (item[0], item[1]))
                    penalty = math.inf if len(active) == 1 else active[1][0] - active[0][0]
                    candidates.append((penalty, active[0][0], 0, i, active[0][1]))

                # Column penalties.
                for j in range(n):
                    if remaining_demand[j] <= 0:
                        continue
                    active = [
                        (cost_values[i][j], i)
                        for i in range(m)
                        if allowed[i][j] and remaining_supply[i] > 0
                    ]
                    if not active:
                        continue
                    active.sort(key=lambda item: (item[0], item[1]))
                    penalty = math.inf if len(active) == 1 else active[1][0] - active[0][0]
                    candidates.append((penalty, active[0][0], 1, active[0][1], j))

                if not candidates:
                    break

                # Highest penalty, then lowest selected cost; row wins exact ties.
                candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
                _, _, _, i, j = candidates[0]
                quantity = _max_feasible_allocation(
                    i, j, remaining_supply, remaining_demand, allowed
                )
                if quantity > 0:
                    selected = (i, j, quantity)
                else:
                    allowed[i][j] = False

        if selected is None:
            raise RuntimeError(
                f"{METHODS[method]} tidak menemukan rute lanjutan, "
                "padahal model awal dinyatakan feasible. Periksa data."
            )

        commit(*selected)

    allocation_df = pd.DataFrame(
        allocation, index=source_names, columns=destination_names
    )
    steps_df = pd.DataFrame(steps)
    total_cost = float(steps_df["Subtotal (USD)"].sum()) if not steps_df.empty else 0.0

    # Final integrity checks.
    if any(
        int(allocation_df.loc[name].sum()) != initial_supply[i]
        for i, name in enumerate(source_names)
    ):
        raise RuntimeError("Hasil akhir tidak memenuhi seluruh supply.")
    if any(int(allocation_df[name].sum()) != initial_demand[j] for j, name in enumerate(destination_names)):
        raise RuntimeError("Hasil akhir tidak memenuhi seluruh demand.")
    for i in range(m):
        for j in range(n):
            if cost_values[i][j] is None and allocation[i][j] != 0:
                raise RuntimeError("Algoritma mengalokasikan barang ke rute terlarang.")

    return {
        "method": method,
        "method_name": METHODS[method],
        "allocation": allocation_df,
        "steps": steps_df,
        "total_cost": total_cost,
        "allocated_kg": int(sum(initial_supply)),
        "nonzero_routes": int((allocation_df > 0).to_numpy().sum()),
        "source_names": source_names,
        "destination_names": destination_names,
    }


def run_all_methods(
    supply: pd.Series, demand: pd.Series, costs: pd.DataFrame
) -> dict[str, dict]:
    """Run all three methods and return their result dictionaries."""
    return {
        method: solve_transportation(supply, demand, costs, method)
        for method in ("NWC", "LCM", "VAM")
    }


def solve_modi(
    supply: pd.Series,
    demand: pd.Series,
    costs: pd.DataFrame,
    initial_allocation: pd.DataFrame,
    *,
    tolerance: float = 1e-9,
    max_iterations: int = 5000,
) -> dict:
    """Improve a feasible basic solution with the MODI (u-v) method.

    Blank cost cells are prohibited routes.  The initial positive allocations
    must form a spanning-tree basis (m+n-1 basic cells), which is the case for
    the Restricted VAM solution produced by this project and the supplied
    SCMS workbook.

    Entering variables use the most-negative reduced cost.  If a basis repeats,
    the routine switches to Bland-style lexicographic entering selection to
    reduce the risk of cycling in degenerate pivots.
    """
    source_names, destination_names, initial_supply, initial_demand, cost_values = (
        _prepare_inputs(supply, demand, costs)
    )
    m, n = len(source_names), len(destination_names)
    cost = cost_values

    if isinstance(initial_allocation, pd.DataFrame):
        allocation_df = initial_allocation.reindex(
            index=source_names, columns=destination_names
        )
        allocation_df = allocation_df.apply(pd.to_numeric, errors="coerce")
        if allocation_df.isna().any().any():
            raise ValueError("Matriks alokasi awal mengandung nilai kosong/nonangka.")
        allocation = [
            [float(value) for value in row]
            for row in allocation_df.itertuples(index=False, name=None)
        ]
    else:
        allocation = [
            [float(value) for value in row]
            for row in initial_allocation
        ]

    if len(allocation) != m or any(len(row) != n for row in allocation):
        raise ValueError("Ukuran alokasi awal tidak sesuai dengan matriks biaya.")

    # Validate nonnegative allocation, forbidden routes, and all constraints.
    for i in range(m):
        for j in range(n):
            value = allocation[i][j]
            if value < -tolerance:
                raise ValueError("Alokasi awal tidak boleh negatif.")
            if cost[i][j] is None and abs(value) > tolerance:
                raise ValueError(
                    f"Alokasi awal menggunakan rute terlarang: "
                    f"{source_names[i]} -> {destination_names[j]}."
                )

    for i, name in enumerate(source_names):
        row_total = sum(allocation[i])
        if abs(row_total - initial_supply[i]) > tolerance:
            raise ValueError(
                f"Alokasi awal tidak memenuhi supply {name}: "
                f"{row_total:,.6f} != {initial_supply[i]:,}."
            )
    for j, name in enumerate(destination_names):
        column_total = sum(allocation[i][j] for i in range(m))
        if abs(column_total - initial_demand[j]) > tolerance:
            raise ValueError(
                f"Alokasi awal tidak memenuhi demand {name}: "
                f"{column_total:,.6f} != {initial_demand[j]:,}."
            )

    # A non-degenerate basic feasible solution for a connected network has
    # m+n-1 basic cells and those edges connect all source/destination nodes.
    basis = {
        (i, j)
        for i in range(m)
        for j in range(n)
        if allocation[i][j] > tolerance
    }
    expected_basis_size = m + n - 1
    if len(basis) != expected_basis_size:
        raise ValueError(
            "Solusi awal belum membentuk basis pohon nondegenerat: "
            f"ditemukan {len(basis)} alokasi positif, sedangkan diperlukan "
            f"{expected_basis_size}. Dataset ini diharapkan menggunakan solusi "
            "awal Restricted VAM yang membentuk basis pohon."
        )

    def basis_path(
        current_basis: set[tuple[int, int]],
        start: tuple[str, int],
        end: tuple[str, int],
    ) -> list[tuple[int, int]]:
        """Return the unique ordered basis-edge path between two bipartite nodes."""
        adjacency: dict[tuple[str, int], list[tuple[tuple[str, int], tuple[int, int]]]] = {}
        for row, column in current_basis:
            row_node = ("s", row)
            col_node = ("d", column)
            adjacency.setdefault(row_node, []).append((col_node, (row, column)))
            adjacency.setdefault(col_node, []).append((row_node, (row, column)))

        queue = [start]
        parent: dict[
            tuple[str, int],
            tuple[tuple[str, int] | None, tuple[int, int] | None],
        ] = {start: (None, None)}
        for node in queue:
            if node == end:
                break
            for neighbor, edge in adjacency.get(node, []):
                if neighbor not in parent:
                    parent[neighbor] = (node, edge)
                    queue.append(neighbor)

        if end not in parent:
            raise RuntimeError("Basis tidak terhubung; siklus MODI tidak dapat dibentuk.")

        path: list[tuple[int, int]] = []
        current = end
        while current != start:
            previous, edge = parent[current]
            if previous is None or edge is None:
                raise RuntimeError("Gagal merekonstruksi siklus MODI.")
            path.append(edge)
            current = previous
        path.reverse()
        return path

    def compute_potentials(
        current_basis: set[tuple[int, int]],
    ) -> tuple[list[float | None], list[float | None]]:
        """Solve u_i + v_j = c_ij along the current spanning-tree basis."""
        u: list[float | None] = [None] * m
        v: list[float | None] = [None] * n
        u[0] = 0.0
        changed = True
        while changed:
            changed = False
            for i, j in sorted(current_basis):
                if u[i] is not None and v[j] is None:
                    assert cost[i][j] is not None
                    v[j] = float(cost[i][j]) - float(u[i])
                    changed = True
                elif v[j] is not None and u[i] is None:
                    assert cost[i][j] is not None
                    u[i] = float(cost[i][j]) - float(v[j])
                    changed = True
        if any(value is None for value in u) or any(value is None for value in v):
            raise RuntimeError("Tidak semua potensial u dan v dapat ditentukan.")
        return u, v

    def objective(current_allocation: list[list[float]]) -> float:
        return sum(
            current_allocation[i][j] * float(cost[i][j])
            for i in range(m)
            for j in range(n)
            if cost[i][j] is not None
        )

    # Store the starting objective and pivot log.
    starting_cost = objective(allocation)
    history: list[dict] = []
    seen_bases: set[frozenset[tuple[int, int]]] = {frozenset(basis)}
    bland_mode = False

    for iteration in range(1, max_iterations + 1):
        u, v = compute_potentials(basis)
        improving: list[tuple[float, int, int]] = []
        for i in range(m):
            for j in range(n):
                if cost[i][j] is None or (i, j) in basis:
                    continue
                reduced_cost = float(cost[i][j]) - float(u[i]) - float(v[j])
                if reduced_cost < -tolerance:
                    improving.append((reduced_cost, i, j))

        # MODI optimality condition for a minimization problem:
        # every allowed nonbasic cell must have c_ij - u_i - v_j >= 0.
        if not improving:
            break

        if bland_mode:
            _, entering_i, entering_j = min(improving, key=lambda item: (item[1], item[2]))
        else:
            _, entering_i, entering_j = min(
                improving, key=lambda item: (item[0], item[1], item[2])
            )

        entering_reduced_cost = (
            float(cost[entering_i][entering_j])
            - float(u[entering_i])
            - float(v[entering_j])
        )
        path = basis_path(
            basis,
            ("s", entering_i),
            ("d", entering_j),
        )
        if not path or len(path) % 2 == 0:
            raise RuntimeError("Lintasan siklus MODI tidak valid untuk jaringan bipartit.")

        # The entering cell receives '+'. Along the unique tree path, signs
        # alternate '-, +, -, ...'; theta is the minimum allocation on '-'.
        minus_edges = path[0::2]
        theta = min(allocation[i][j] for i, j in minus_edges)
        leaving_candidates = [
            edge
            for edge in minus_edges
            if abs(allocation[edge[0]][edge[1]] - theta) <= tolerance
        ]
        leaving_i, leaving_j = min(leaving_candidates)

        before = objective(allocation)
        allocation[entering_i][entering_j] += theta
        cycle_labels = [
            f"+ {source_names[entering_i]} → {destination_names[entering_j]}"
        ]
        for path_index, (i, j) in enumerate(path):
            if path_index % 2 == 0:
                allocation[i][j] -= theta
                sign = "−"
            else:
                allocation[i][j] += theta
                sign = "+"
            if abs(allocation[i][j]) <= tolerance:
                allocation[i][j] = 0.0
            cycle_labels.append(
                f"{sign} {source_names[i]} → {destination_names[j]}"
            )

        after = objective(allocation)
        # A negative reduced cost should never increase the objective.
        if after > before + max(1e-5, abs(before) * 1e-10):
            raise RuntimeError("Pivot MODI meningkatkan biaya; periksa siklus dan tanda.")

        basis.remove((leaving_i, leaving_j))
        basis.add((entering_i, entering_j))
        basis_signature = frozenset(basis)
        if basis_signature in seen_bases:
            bland_mode = True
        seen_bases.add(basis_signature)

        history.append({
            "Iterasi": iteration,
            "Rute masuk": f"{source_names[entering_i]} → {destination_names[entering_j]}",
            "Indeks perbaikan (USD/kg)": entering_reduced_cost,
            "Theta (kg)": theta,
            "Rute keluar": f"{source_names[leaving_i]} → {destination_names[leaving_j]}",
            "Biaya sebelum (USD)": before,
            "Biaya sesudah (USD)": after,
            "Siklus": " | ".join(cycle_labels),
        })
    else:
        raise RuntimeError(
            f"MODI belum konvergen setelah {max_iterations:,} iterasi."
        )

    # Recompute final potentials and reduced costs for the optimality check.
    final_u, final_v = compute_potentials(basis)
    reduced_values: list[list[float | None]] = []
    nonbasic_reduced_costs: list[float] = []
    for i in range(m):
        row: list[float | None] = []
        for j in range(n):
            if cost[i][j] is None:
                row.append(None)
            elif (i, j) in basis:
                row.append(0.0)
            else:
                value = float(cost[i][j]) - float(final_u[i]) - float(final_v[j])
                row.append(value)
                nonbasic_reduced_costs.append(value)
        reduced_values.append(row)

    final_cost = objective(allocation)
    allocation_result = pd.DataFrame(
        allocation,
        index=source_names,
        columns=destination_names,
    )
    reduced_cost_result = pd.DataFrame(
        reduced_values,
        index=source_names,
        columns=destination_names,
    )

    # Final feasibility and route checks.
    for i, name in enumerate(source_names):
        if abs(sum(allocation[i]) - initial_supply[i]) > tolerance:
            raise RuntimeError(f"Hasil MODI tidak memenuhi supply {name}.")
    for j, name in enumerate(destination_names):
        if abs(sum(allocation[i][j] for i in range(m)) - initial_demand[j]) > tolerance:
            raise RuntimeError(f"Hasil MODI tidak memenuhi demand {destination_names[j]}.")
    for i in range(m):
        for j in range(n):
            if cost[i][j] is None and abs(allocation[i][j]) > tolerance:
                raise RuntimeError("Hasil MODI menggunakan rute yang tidak tersedia.")

    minimum_reduced_cost = min(nonbasic_reduced_costs, default=0.0)
    if minimum_reduced_cost < -tolerance:
        raise RuntimeError("Kondisi optimalitas MODI belum terpenuhi.")

    iterations_df = pd.DataFrame(history)
    return {
        "method": "MODI",
        "method_name": "MODI (optimality improvement)",
        "starting_method": "VAM",
        "starting_cost": starting_cost,
        "allocation": allocation_result,
        "iterations": iterations_df,
        "total_cost": final_cost,
        "cost_reduction": starting_cost - final_cost,
        "allocated_kg": int(sum(initial_supply)),
        "nonzero_routes": int((allocation_result > tolerance).to_numpy().sum()),
        "u_potentials": pd.Series(final_u, index=source_names, name="u_i"),
        "v_potentials": pd.Series(final_v, index=destination_names, name="v_j"),
        "reduced_costs": reduced_cost_result,
        "minimum_reduced_cost": minimum_reduced_cost,
        "basis_size": len(basis),
        "optimality_verified": True,
    }
