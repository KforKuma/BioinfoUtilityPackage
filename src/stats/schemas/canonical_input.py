from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import numpy as np
import pandas as pd


CANONICAL_INPUT_HASH_ALGORITHM = "canonical-csv-sha256-v2"
LEGACY_INPUT_HASH_ALGORITHM = "canonical-csv-sha256-v1"


@dataclass(frozen=True)
class CanonicalDAInput:
    abundance_long: pd.DataFrame
    sample_manifest: pd.DataFrame
    cell_type_manifest: pd.DataFrame
    contrast_specification: pd.DataFrame

    def validate(self) -> "CanonicalDAInput":
        abundance_required = {"sample_id", "cell_type", "count", "total_count", "proportion"}
        sample_required = {"sample_id", "inclusion_status"}
        cell_required = {"cell_type", "inclusion_status"}
        contrast_required = {
            "contrast_id", "contrast_type", "contrast_definition", "group_1", "group_2",
        }
        for name, frame, required in (
            ("abundance_long", self.abundance_long, abundance_required),
            ("sample_manifest", self.sample_manifest, sample_required),
            ("cell_type_manifest", self.cell_type_manifest, cell_required),
            ("contrast_specification", self.contrast_specification, contrast_required),
        ):
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

        if self.sample_manifest["sample_id"].duplicated().any():
            raise ValueError("`sample_manifest.sample_id` must be unique.")
        if self.cell_type_manifest["cell_type"].duplicated().any():
            raise ValueError("`cell_type_manifest.cell_type` must be unique.")
        if self.contrast_specification["contrast_id"].duplicated().any():
            raise ValueError("`contrast_specification.contrast_id` must be unique.")
        if self.abundance_long.duplicated(["sample_id", "cell_type"]).any():
            raise ValueError("`abundance_long` must have one row per sample_id x cell_type.")

        counts = pd.to_numeric(self.abundance_long["count"], errors="coerce")
        if counts.isna().any() or (~np.isfinite(counts)).any() or (counts < 0).any():
            raise ValueError("`count` must contain finite non-negative integers.")
        if not np.allclose(counts, np.round(counts)):
            raise ValueError("`count` must contain integer values.")

        included_samples = self.sample_manifest.loc[
            self.sample_manifest["inclusion_status"].eq("included"), "sample_id"
        ].astype(str)
        included_cell_types = self.cell_type_manifest.loc[
            self.cell_type_manifest["inclusion_status"].eq("included"), "cell_type"
        ].astype(str)
        expected = pd.MultiIndex.from_product(
            [included_samples, included_cell_types], names=["sample_id", "cell_type"]
        )
        observed = pd.MultiIndex.from_frame(
            self.abundance_long[["sample_id", "cell_type"]].astype(str)
        )
        missing_pairs = expected.difference(observed)
        if len(missing_pairs):
            raise ValueError(
                "`abundance_long` is not a complete included sample x cell-type product; "
                f"first missing pairs: {missing_pairs[:5].tolist()}"
            )

        per_sample_sum = self.abundance_long.groupby("sample_id", sort=False)["count"].transform("sum")
        total_count = pd.to_numeric(self.abundance_long["total_count"], errors="coerce")
        if not np.allclose(per_sample_sum, total_count, rtol=0, atol=0):
            raise ValueError("`total_count` must equal the sample-wise sum of `count`.")
        proportion = pd.to_numeric(self.abundance_long["proportion"], errors="coerce")
        expected_proportion = counts / total_count.replace(0, np.nan)
        if not np.allclose(proportion, expected_proportion, equal_nan=True, rtol=1e-10, atol=1e-12):
            raise ValueError("`proportion` must equal count / total_count.")
        return self

    def input_hash(self, algorithm: str = CANONICAL_INPUT_HASH_ALGORITHM) -> str:
        """Hash semantic input with a CSV-roundtrip-stable representation."""
        if algorithm not in {CANONICAL_INPUT_HASH_ALGORITHM, LEGACY_INPUT_HASH_ALGORITHM}:
            raise ValueError(f"Unsupported canonical input hash algorithm: {algorithm!r}")
        digest = sha256()
        for frame in (
            self.abundance_long,
            self.sample_manifest,
            self.cell_type_manifest,
            self.contrast_specification,
        ):
            csv_options = {"index": False, "lineterminator": "\n"}
            if algorithm == CANONICAL_INPUT_HASH_ALGORITHM:
                columns = sorted(frame.columns)
                normalized = frame.loc[:, columns].sort_values(columns, kind="stable")
                # Default float formatting is not stable when a derived proportion
                # moves by one machine epsilon after a CSV handoff.
                csv_options.update({"float_format": "%.10g", "na_rep": "<NA>"})
            else:
                # Preserve the precise v1 representation for historical manifests.
                normalized = frame.sort_index(axis=1).sort_values(
                    list(frame.columns), kind="stable"
                )
            digest.update(normalized.to_csv(**csv_options).encode("utf-8"))
        return digest.hexdigest()
