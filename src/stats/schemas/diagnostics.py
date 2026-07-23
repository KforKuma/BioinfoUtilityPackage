from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MethodDiagnostics:
    diagnostic_id: str
    analysis_id: str
    method: str
    method_version: str
    status: str
    input_hash: str
    started_at: str
    finished_at: str | None = None
    native_output_path: str | None = None
    converged: bool | None = None
    warnings: list[str] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def finish(self, *, status: str | None = None) -> None:
        if status is not None:
            self.status = status
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["warnings"] = "\n".join(self.warnings)
        return record

