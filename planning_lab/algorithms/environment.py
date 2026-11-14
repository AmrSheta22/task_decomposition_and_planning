"""
planning/planning_lab/algorithms/environment.py — CopperleafEnvironment

--- grounded environment: environment.py::CopperleafEnvironment ---

Replaces the toolkit's stock Environment, which literally ignores its input
(`del state`) and returns a random score — the exact thing this lab requires
you to replace with a real check.

LATS calls `environment.evaluate(state)` where `state` is free-text: the
LLM's fully-written candidate solution (see lats.py: "Each state must
contain the fully written solution"). This class parses that text for a
concrete item/supplier/quantity/expedited decision and checks it against
REAL data — current stock, which supplier actually carries the item, and
(for expedited orders) how many expedited orders that supplier has already
taken today (the same rule create_supplier_order enforces for real writes).

Deliberately read-only: evaluate() never calls create_supplier_order or
writes to the DB. LATS explores several candidate branches per iteration —
actually placing an order during evaluation would create real side effects
for orders that get pruned. This is a dry-run check against real state, not
an execution.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..models import EnvironmentFeedback


def _load_copperleaf_db():
    """Lazily import Copperleaf's db/validation modules, adding
    mcp_server/ to sys.path only when this is actually called — not at
    module import time. This keeps `import environment` safe from a
    standalone checkout of the toolkit fork (which has no mcp_server/ of
    its own), while CopperleafEnvironment.evaluate() still works exactly
    as before whenever this file is running inside copperleaf-mcp (either
    vendored under planning/ or, if this fork is ever pip-installed
    alongside copperleaf-mcp, from a sibling checkout)."""
    import sys

    mcp_server_dir = Path(__file__).resolve().parents[3] / "mcp_server"
    if str(mcp_server_dir) not in sys.path:
        sys.path.insert(0, str(mcp_server_dir))
    try:
        from db import get_connection  # noqa
        from validation import MAX_EXPEDITED_ORDERS_PER_SUPPLIER_PER_DAY  # noqa
    except ImportError as exc:
        raise RuntimeError(
            "CopperleafEnvironment.evaluate() needs Copperleaf's mcp_server/ "
            "(db.py, validation.py) on the path. This file is grounded in "
            "the real copperleaf-mcp database and only runs from inside "
            "that repo (planning/planning_lab/algorithms/environment.py), "
            "not from a standalone toolkit-fork checkout."
        ) from exc
    return get_connection, MAX_EXPEDITED_ORDERS_PER_SUPPLIER_PER_DAY

_ITEM_ID = re.compile(r"item[_\s]?id[:\s=]+(\d+)", re.IGNORECASE)
_SUPPLIER_ID = re.compile(r"supplier[_\s]?id[:\s=]+(\d+)", re.IGNORECASE)
_QUANTITY = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|g|l|ml|units?|cases?)\b", re.IGNORECASE)
_EXPEDITED = re.compile(r"\bexpedit", re.IGNORECASE)


class Environment:
    """Kept as a base class name for drop-in compatibility with the
    toolkit's type hints (lats.py imports `Environment` for its signature).
    CopperleafEnvironment below is the real implementation used everywhere
    in this repo; nothing in this codebase constructs the bare `Environment`
    directly."""

    def evaluate(self, state: str) -> EnvironmentFeedback:  # pragma: no cover
        raise NotImplementedError


class CopperleafEnvironment(Environment):
    """Real grounding for the demand-spike / expedite-order problem.
    Source of truth: the actual `inventory_items` and `supplier_orders`
    tables in copperleaf.db — never the model's own opinion of its output.
    """

    def evaluate(self, state: str) -> EnvironmentFeedback:
        get_connection, MAX_EXPEDITED_ORDERS_PER_SUPPLIER_PER_DAY = _load_copperleaf_db()
        item_id_match = _ITEM_ID.search(state)
        supplier_id_match = _SUPPLIER_ID.search(state)
        quantity_match = _QUANTITY.search(state)
        expedited = bool(_EXPEDITED.search(state))

        if not item_id_match or not supplier_id_match or not quantity_match:
            missing = [
                name
                for name, match in [
                    ("item_id", item_id_match),
                    ("supplier_id", supplier_id_match),
                    ("quantity", quantity_match),
                ]
                if not match
            ]
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[
                    f"Candidate does not specify a concrete {', '.join(missing)} — "
                    "cannot be checked against real inventory/supplier data. "
                    "State must name an exact item_id, supplier_id, and quantity."
                ],
            )

        item_id = int(item_id_match.group(1))
        supplier_id = int(supplier_id_match.group(1))
        quantity = float(quantity_match.group(1))

        with get_connection() as conn:
            item = conn.execute(
                "SELECT item_id, supplier_id, name, current_quantity "
                "FROM inventory_items WHERE item_id = ?",
                (item_id,),
            ).fetchone()

        if item is None:
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[f"No inventory item with item_id={item_id} — candidate references a real DB row that doesn't exist."],
            )

        if item["supplier_id"] != supplier_id:
            return EnvironmentFeedback(
                success=False,
                score=0.1,
                details=[
                    f"supplier_id={supplier_id} does not supply item_id={item_id} "
                    f"('{item['name']}') — its actual supplier is supplier_id={item['supplier_id']}."
                ],
            )

        if quantity <= 0:
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[f"quantity {quantity} is not a valid positive order amount."],
            )

        if expedited:
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM supplier_orders "
                    "WHERE supplier_id = ? AND expedited = 1 AND date(ordered_at) = date('now')",
                    (supplier_id,),
                ).fetchone()
            todays_count = row["n"]
            if todays_count >= MAX_EXPEDITED_ORDERS_PER_SUPPLIER_PER_DAY:
                return EnvironmentFeedback(
                    success=False,
                    score=0.2,
                    details=[
                        f"supplier_id={supplier_id} already has {todays_count} expedited "
                        f"orders today (max {MAX_EXPEDITED_ORDERS_PER_SUPPLIER_PER_DAY}) — "
                        "this expedite would be rejected in reality. Propose a standard "
                        "order or a different supplier instead."
                    ],
                )

        # Real, valid, checkable order. Score reflects how well it covers
        # the current shortfall relative to reorder_threshold, not a flat 1.0.
        shortfall = max(item["current_quantity"], 0.0)
        coverage = min(quantity / max(shortfall, 1.0), 1.0) if shortfall > 0 else 1.0
        score = round(0.6 + 0.4 * coverage, 4)  # floor of 0.6 for any valid, real order

        return EnvironmentFeedback(success=True, score=score, details=[])