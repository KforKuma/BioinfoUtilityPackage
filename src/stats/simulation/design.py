from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def validate_factor_levels(
    disease_levels: Sequence[str],
    tissue_levels: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate the factor design currently supported by abundance generators."""
    diseases = tuple(str(level) for level in disease_levels)
    tissues = tuple(str(level) for level in tissue_levels)
    if len(diseases) < 2 or len(set(diseases)) != len(diseases) or any(not x for x in diseases):
        raise ValueError("`disease_levels` must contain at least two unique non-empty levels.")
    if len(tissues) != 2 or len(set(tissues)) != 2 or any(not x for x in tissues):
        raise ValueError(
            "The current truth contract supports exactly two unique non-empty tissue levels."
        )
    return diseases, tissues


def build_simulation_metadata(
    *,
    n_donors: int,
    n_samples_per_donor: int,
    disease_levels: Sequence[str],
    tissue_levels: Sequence[str],
    rng: np.random.Generator,
    assignment_strategy: str = "balanced",
) -> pd.DataFrame:
    """Build donor/sample metadata with an explicit randomized assignment strategy.

    ``balanced`` randomizes labels while keeping disease donor counts and tissue counts
    within each disease stratum at most one apart. ``random`` preserves the historical
    independent assignment behavior for compatibility.
    """
    diseases, tissues = validate_factor_levels(disease_levels, tissue_levels)
    if not isinstance(n_donors, int) or isinstance(n_donors, bool) or n_donors <= 0:
        raise ValueError("`n_donors` must be a positive integer.")
    if (
        not isinstance(n_samples_per_donor, int)
        or isinstance(n_samples_per_donor, bool)
        or n_samples_per_donor <= 0
    ):
        raise ValueError("`n_samples_per_donor` must be a positive integer.")
    if assignment_strategy not in {"balanced", "random"}:
        raise ValueError("`assignment_strategy` must be 'balanced' or 'random'.")
    if assignment_strategy == "balanced" and n_donors < len(diseases):
        raise ValueError("Balanced assignment requires at least one donor per disease level.")

    donor_ids = [f"D{index + 1}" for index in range(n_donors)]
    if assignment_strategy == "random":
        donor_diseases = rng.choice(diseases, size=n_donors).tolist()
        tissue_assignments = {
            disease: rng.choice(tissues, size=donor_diseases.count(disease) * n_samples_per_donor).tolist()
            for disease in diseases
        }
    else:
        donor_diseases = list(np.resize(np.asarray(diseases, dtype=object), n_donors))
        rng.shuffle(donor_diseases)
        tissue_assignments = {}
        for disease in diseases:
            slots = donor_diseases.count(disease) * n_samples_per_donor
            if slots < len(tissues):
                raise ValueError(
                    "Balanced assignment requires every disease stratum to contain every tissue level."
                )
            assignments = list(np.resize(np.asarray(tissues, dtype=object), slots))
            rng.shuffle(assignments)
            tissue_assignments[disease] = assignments

    cursors = {disease: 0 for disease in diseases}
    records: list[dict[str, object]] = []
    for donor_id, disease in zip(donor_ids, donor_diseases, strict=True):
        start = cursors[disease]
        stop = start + n_samples_per_donor
        donor_tissues = tissue_assignments[disease][start:stop]
        cursors[disease] = stop
        for sample_index, tissue in enumerate(donor_tissues, start=1):
            records.append({
                "donor_id": donor_id,
                "disease": disease,
                "tissue": tissue,
                "sample_id": f"{donor_id}_S{sample_index}",
                "assignment_strategy": assignment_strategy,
            })
    metadata = pd.DataFrame(records)
    if assignment_strategy == "balanced":
        disease_counts = metadata.groupby("disease")["donor_id"].nunique()
        tissue_imbalances = (
            metadata.groupby(["disease", "tissue"]).size()
            .groupby(level="disease").agg(lambda values: values.max() - values.min())
        )
        if disease_counts.max() - disease_counts.min() > 1 or tissue_imbalances.gt(1).any():
            raise AssertionError("Balanced metadata construction violated its design invariant.")
    return metadata
