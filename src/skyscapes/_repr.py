"""Shared formatting helpers for hierarchical ``__repr__`` methods.

Each leaf class in skyscapes (atmospheres, disks, stars, zodi) defines
its own multi-line ``__repr__``; aggregator classes (Planet, System,
Scene) use :func:`indent` to nest the children one level deeper. The
result is a tree-shaped summary that lets you call ``repr(scene)`` in
a notebook and see everything in the scene at a glance.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def indent(text: str, prefix: str = "  ") -> str:
    """Prefix every line of ``text`` with ``prefix``.

    Used by aggregator-class reprs (``Scene``, ``System``, ``Planet``)
    to nest the child reprs one level deeper.
    """
    return "\n".join(prefix + line for line in text.split("\n"))


def fmt_scalar_or_array(x: Array, fmt: str = ".3g", max_items: int = 3) -> str:
    """Format a scalar or small array compactly.

    For a scalar (shape () or (1,)): formats as ``"3.14"``.
    For a short array: formats as ``"[3.14, 2.72, 1.41]"``.
    For a long array: formats as ``"[3.14, 2.72, ... (N=20)]"``.
    """
    a = jnp.asarray(x)
    if a.shape == () or a.shape == (1,):
        v = float(a.reshape(-1)[0])
        return f"{v:{fmt}}"
    if a.size <= max_items:
        return "[" + ", ".join(f"{float(v):{fmt}}" for v in a) + "]"
    head = ", ".join(f"{float(v):{fmt}}" for v in a[:max_items])
    return f"[{head}, ... (N={a.size})]"
