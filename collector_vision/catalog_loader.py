"""Default catalog loader with deterministic v1 compatibility dispatch."""

from __future__ import annotations

import os
from typing import Any, TypeAlias

from collector_vision.catalog import Catalog as CatalogV1
from collector_vision.catalog_v2 import CatalogV2
from collector_vision.hfd import HFD

CatalogLike: TypeAlias = CatalogV1 | CatalogV2


class Catalog:
    """Load the recommended v2 catalog or an explicitly requested v1 catalog."""

    def __new__(cls, *args: object, **kwargs: object) -> Catalog:
        raise TypeError("Catalog performs I/O through Catalog.load(...), not its constructor")

    @classmethod
    def load(cls, target: str | os.PathLike[str] | HFD | object, **kwargs: Any):
        """Load v2 by game, or v1 when *target* identifies an NPZ catalog.

        Game names and aliases select Catalog v2. Paths, ``hf://`` references,
        and :class:`HFD` objects retain the Catalog v1 loading contract.
        """
        if _is_v1_source(target):
            if kwargs:
                options = ", ".join(sorted(kwargs))
                raise TypeError(f"Catalog v1 loading does not accept options: {options}")
            return CatalogV1.load(target)
        return CatalogV2.load(target, **kwargs)


def _is_v1_source(source: object) -> bool:
    if isinstance(source, (os.PathLike, HFD)):
        return True
    if not isinstance(source, str):
        return False
    if source.startswith("hf://") or source.lower().endswith(".npz"):
        return True
    if source.startswith((".", "~", os.sep)) or "/" in source or "\\" in source:
        return True
    return False
