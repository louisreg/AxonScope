"""Backend-neutral planning for repeated membrane parameter rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class MembraneRowPlan:
    """Map population rows onto unique membrane descriptions.

    The plan contains no backend arrays or executable membrane programs. It
    identifies equivalent descriptive rows so the selected runtime can lower
    each one once and gather the resulting dynamic initial state.
    """

    row_parameter_indices: np.ndarray
    representative_item_indices: np.ndarray
    signatures: tuple[tuple[Any, ...], ...]
    model_signatures: tuple[tuple[Any, ...], ...]
    unique_row_model_indices: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        self.row_parameter_indices.setflags(write=False)
        self.representative_item_indices.setflags(write=False)
        for indices in self.unique_row_model_indices:
            indices.setflags(write=False)

    @classmethod
    def from_dispatch_items(cls, items: Sequence[Any]) -> "MembraneRowPlan":
        """Build a unique-row plan from runtime-neutral dispatch metadata."""

        rows = tuple(items)
        if not rows:
            raise ValueError("items cannot be empty.")
        row_indices = np.empty((len(rows),), dtype=np.int32)
        representatives: list[int] = []
        signatures: list[tuple[Any, ...]] = []
        unique_by_signature: dict[tuple[Any, ...], int] = {}
        unique_by_solver_identity: dict[tuple[int, float], int] = {}
        for item_index, item in enumerate(rows):
            v_init = float(getattr(item.simulation, "v_init", 0.0))
            identity_key = (id(item.solver_axon), v_init)
            parameter_index = unique_by_solver_identity.get(identity_key)
            if parameter_index is not None:
                row_indices[item_index] = parameter_index
                continue
            signature = (
                tuple(item.membrane_signature),
                int(item.solver_axon.n_compartments),
                v_init,
            )
            parameter_index = unique_by_signature.get(signature)
            if parameter_index is None:
                parameter_index = len(representatives)
                unique_by_signature[signature] = parameter_index
                representatives.append(item_index)
                signatures.append(signature)
            unique_by_solver_identity[identity_key] = parameter_index
            row_indices[item_index] = parameter_index

        model_signatures: list[tuple[Any, ...]] = []
        model_by_signature: dict[tuple[Any, ...], int] = {}
        model_by_identity: dict[int, int] = {}
        unique_row_model_indices: list[np.ndarray] = []
        for item_index in representatives:
            item = rows[item_index]
            indices = np.empty((len(item.membrane_signature),), dtype=np.int32)
            for compartment_index, (model, signature) in enumerate(
                zip(
                    item.solver_axon.membrane_models,
                    item.membrane_signature,
                    strict=True,
                )
            ):
                model_index = model_by_identity.get(id(model))
                if model_index is None:
                    model_index = model_by_signature.get(signature)
                if model_index is None:
                    model_index = len(model_signatures)
                    model_signatures.append(signature)
                    model_by_signature[signature] = model_index
                model_by_identity[id(model)] = model_index
                indices[compartment_index] = model_index
            unique_row_model_indices.append(indices)
        return cls(
            row_parameter_indices=row_indices,
            representative_item_indices=np.asarray(representatives, dtype=np.int32),
            signatures=tuple(signatures),
            model_signatures=tuple(model_signatures),
            unique_row_model_indices=tuple(unique_row_model_indices),
        )

    @property
    def size(self) -> int:
        return int(self.row_parameter_indices.shape[0])

    @property
    def unique_count(self) -> int:
        return int(self.representative_item_indices.shape[0])

    @property
    def cache_hits(self) -> int:
        return self.size - self.unique_count

    @property
    def unique_model_count(self) -> int:
        return len(self.model_signatures)
