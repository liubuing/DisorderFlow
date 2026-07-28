#!/usr/bin/env python
"""Validated ingestion of experimental StateContrast binding and HDX data."""
from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from pathlib import Path

SCHEMA_VERSION = "statecontrast.experimental.v1"
EVIDENCE_TIERS = {"tier_1_primary", "tier_2_processed", "tier_3_reported"}
QUALITY_FLAGS = {"pass", "warning", "exclude"}
CENSORING_VALUES = {"none", "left", "right", "interval", "not_quantified"}

BINDING_REQUIRED_FIELDS = (
    "schema_version", "experiment_id", "assay_type", "binder_id", "state_id",
    "replicate_id", "metric", "value", "unit", "censoring", "quality_flag",
    "evidence_tier",
)
HDX_REQUIRED_FIELDS = (
    "schema_version", "experiment_id", "protein_id", "state_id", "peptide_id",
    "peptide_start", "peptide_end", "timepoint", "timepoint_unit", "uptake_value",
    "uptake_unit", "replicate_id", "censoring", "quality_flag", "evidence_tier",
)

BINDING_SCHEMA = {
    "schema_version": SCHEMA_VERSION,
    "format": "tidy_spr_bli_state_matrix",
    "required_fields": BINDING_REQUIRED_FIELDS,
    "unit_field": "unit",
    "replicate_field": "replicate_id",
    "censoring_field": "censoring",
    "quality_field": "quality_flag",
    "evidence_tier_field": "evidence_tier",
}
HDX_SCHEMA = {
    "schema_version": SCHEMA_VERSION,
    "format": "hdx_peptide_uptake",
    "required_fields": HDX_REQUIRED_FIELDS,
    "unit_fields": ("timepoint_unit", "uptake_unit"),
    "replicate_field": "replicate_id",
    "censoring_field": "censoring",
    "quality_field": "quality_flag",
    "evidence_tier_field": "evidence_tier",
}

_BINDING_UNITS = {
    "kd": {"M", "mM", "uM", "nM", "pM"},
    "ka": {"1/(M*s)", "M^-1 s^-1"},
    "koff": {"1/s", "s^-1"},
    "response": {"RU", "nm"},
}
_HDX_TIME_UNITS = {"s", "min", "h"}
_HDX_UPTAKE_UNITS = {"Da", "%"}


class ExperimentalDataValidationError(ValueError):
    """Raised when experimental input fails its explicit schema contract."""


def _require_columns(rows: list[dict], required: Iterable[str], source: Path) -> None:
    if not rows:
        raise ExperimentalDataValidationError(f"{source}: CSV contains no data rows")
    missing = [field for field in required if field not in rows[0]]
    if missing:
        raise ExperimentalDataValidationError(f"{source}: missing columns {missing!r}")


def _required_text(row: dict, field: str, row_number: int) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise ExperimentalDataValidationError(f"row {row_number}: {field} is required")
    return value


def _number(row: dict, field: str, row_number: int, *, allow_blank: bool = False) -> float | None:
    text = str(row.get(field, "")).strip()
    if allow_blank and not text:
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise ExperimentalDataValidationError(f"row {row_number}: {field} must be numeric") from error
    if not math.isfinite(value):
        raise ExperimentalDataValidationError(f"row {row_number}: {field} must be finite")
    return value


def _validate_common(row: dict, row_number: int) -> dict:
    version = _required_text(row, "schema_version", row_number)
    if version != SCHEMA_VERSION:
        raise ExperimentalDataValidationError(
            f"row {row_number}: unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}"
        )
    censoring = _required_text(row, "censoring", row_number)
    quality = _required_text(row, "quality_flag", row_number)
    tier = _required_text(row, "evidence_tier", row_number)
    if censoring not in CENSORING_VALUES:
        raise ExperimentalDataValidationError(f"row {row_number}: invalid censoring {censoring!r}")
    if quality not in QUALITY_FLAGS:
        raise ExperimentalDataValidationError(f"row {row_number}: invalid quality_flag {quality!r}")
    if tier not in EVIDENCE_TIERS:
        raise ExperimentalDataValidationError(f"row {row_number}: invalid evidence_tier {tier!r}")
    _required_text(row, "replicate_id", row_number)
    return {"censoring": censoring, "quality_flag": quality, "evidence_tier": tier}


def validate_binding_rows(rows: Iterable[dict]) -> list[dict]:
    """Validate tidy SPR/BLI state-matrix rows without converting units."""
    validated = []
    for row_number, source_row in enumerate(rows, 2):
        row = dict(source_row)
        common = _validate_common(row, row_number)
        assay = _required_text(row, "assay_type", row_number).upper()
        if assay not in {"SPR", "BLI"}:
            raise ExperimentalDataValidationError(f"row {row_number}: assay_type must be SPR or BLI")
        metric = _required_text(row, "metric", row_number).lower()
        unit = _required_text(row, "unit", row_number)
        if metric not in _BINDING_UNITS or unit not in _BINDING_UNITS[metric]:
            raise ExperimentalDataValidationError(
                f"row {row_number}: unsupported metric/unit pair {metric!r}/{unit!r}"
            )
        value = _number(row, "value", row_number, allow_blank=common["censoring"] == "not_quantified")
        if common["censoring"] != "not_quantified" and value is not None and value < 0:
            raise ExperimentalDataValidationError(f"row {row_number}: value cannot be negative")
        for field in ("experiment_id", "binder_id", "state_id"):
            _required_text(row, field, row_number)
        row.update(common)
        row.update({"assay_type": assay, "metric": metric, "value": value, "unit": unit})
        validated.append(row)
    return validated


def validate_hdx_rows(rows: Iterable[dict]) -> list[dict]:
    """Validate HDX peptide uptake rows without imputing uptake or timepoints."""
    validated = []
    for row_number, source_row in enumerate(rows, 2):
        row = dict(source_row)
        common = _validate_common(row, row_number)
        for field in ("experiment_id", "protein_id", "state_id", "peptide_id"):
            _required_text(row, field, row_number)
        try:
            start = int(_required_text(row, "peptide_start", row_number))
            end = int(_required_text(row, "peptide_end", row_number))
        except ValueError as error:
            raise ExperimentalDataValidationError(
                f"row {row_number}: peptide_start and peptide_end must be integers"
            ) from error
        if start < 1 or end < start:
            raise ExperimentalDataValidationError(f"row {row_number}: invalid peptide residue range")
        timepoint = _number(row, "timepoint", row_number)
        if timepoint is not None and timepoint < 0:
            raise ExperimentalDataValidationError(f"row {row_number}: timepoint cannot be negative")
        time_unit = _required_text(row, "timepoint_unit", row_number)
        uptake_unit = _required_text(row, "uptake_unit", row_number)
        if time_unit not in _HDX_TIME_UNITS:
            raise ExperimentalDataValidationError(f"row {row_number}: invalid timepoint_unit {time_unit!r}")
        if uptake_unit not in _HDX_UPTAKE_UNITS:
            raise ExperimentalDataValidationError(f"row {row_number}: invalid uptake_unit {uptake_unit!r}")
        uptake = _number(
            row, "uptake_value", row_number,
            allow_blank=common["censoring"] == "not_quantified",
        )
        if uptake is not None and (uptake < 0 or uptake_unit == "%" and uptake > 100):
            raise ExperimentalDataValidationError(f"row {row_number}: uptake_value is out of range")
        row.update(common)
        row.update({
            "peptide_start": start, "peptide_end": end, "timepoint": timepoint,
            "timepoint_unit": time_unit, "uptake_value": uptake, "uptake_unit": uptake_unit,
        })
        validated.append(row)
    return validated


def _read_csv(path: str | Path, required: Iterable[str]) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        raise ExperimentalDataValidationError(f"CSV does not exist: {source}")
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ExperimentalDataValidationError(f"could not read CSV: {source}") from error
    _require_columns(rows, required, source)
    return rows


def load_binding_state_matrix_csv(path: str | Path) -> list[dict]:
    """Load a tidy SPR/BLI matrix: one state, replicate, and metric per row."""
    return validate_binding_rows(_read_csv(path, BINDING_REQUIRED_FIELDS))


def load_hdx_peptide_uptake_csv(path: str | Path) -> list[dict]:
    """Load one HDX peptide/timepoint/replicate uptake observation per row."""
    return validate_hdx_rows(_read_csv(path, HDX_REQUIRED_FIELDS))
