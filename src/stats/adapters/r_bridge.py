from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from src.stats.adapters._shared import NativeAdapterError, NativeInput


class RScriptBridge:
    """UTF-8 file-protocol bridge for isolated R differential-abundance backends."""

    def __init__(
        self,
        rscript: str | Path,
        script: str | Path | None = None,
        *,
        library_path: str | Path | None = None,
        cmdstan_path: str | Path | None = None,
        staging_root: str | Path | None = None,
        timeout_seconds: int = 3600,
    ) -> None:
        self.rscript = Path(rscript)
        self.script = (
            Path(script) if script else
            Path(__file__).resolve().parents[1] / "backends" / "r" / "run_da_method.R"
        )
        self.library_path = Path(library_path) if library_path else None
        self.cmdstan_path = Path(cmdstan_path) if cmdstan_path else None
        if self.cmdstan_path is None and self.library_path is not None:
            cmdstan_root = self.library_path.parent / "cmdstan"
            candidates = sorted(
                path
                for path in cmdstan_root.glob("cmdstan-*")
                if path.is_dir()
                and (path / "bin" / "stanc.exe").is_file()
                and (path / "bin" / "stansummary.exe").is_file()
            )
            if candidates:
                self.cmdstan_path = candidates[-1]
        self.staging_root = Path(staging_root) if staging_root else None
        self.timeout_seconds = timeout_seconds

    def run(self, method: str, native_input: NativeInput) -> pd.DataFrame:
        if not self.rscript.is_file():
            raise NativeAdapterError("runtime_unavailable", f"Rscript was not found: {self.rscript}")
        if not self.script.is_file():
            raise NativeAdapterError("runtime_unavailable", f"R backend script was not found: {self.script}")
        if self.staging_root is not None:
            self.staging_root.mkdir(parents=True, exist_ok=True)

        staging_root = self.staging_root
        if method == "sccomp" and staging_root is not None:
            try:
                str(staging_root.resolve()).encode("ascii")
            except UnicodeEncodeError:
                # CmdStan records the working path in CSV comments. cmdstanr on
                # Windows can parse those comments with the active ANSI codepage,
                # so use an ASCII system temp path for this native backend only.
                staging_root = None

        with tempfile.TemporaryDirectory(
            prefix=f"da_{method}_",
            dir=str(staging_root) if staging_root else None,
        ) as work_dir_string:
            work_dir = Path(work_dir_string)
            native_input.abundance.to_csv(work_dir / "abundance.csv", index=False, encoding="utf-8")
            native_input.sample_manifest.to_csv(work_dir / "sample_manifest.csv", index=False, encoding="utf-8")
            native_input.cell_type_manifest.to_csv(work_dir / "cell_type_manifest.csv", index=False, encoding="utf-8")
            with (work_dir / "run_spec.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {"contrast": native_input.contrast, "options": native_input.options},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            command = [str(self.rscript), "--vanilla", str(self.script), method, str(work_dir)]
            env = os.environ.copy() if self.library_path is not None or self.cmdstan_path is not None else None
            if self.library_path is not None:
                env["R_LIBS_USER"] = str(self.library_path)
            if self.cmdstan_path is not None:
                env["CMDSTAN"] = str(self.cmdstan_path)
                tbb_path = self.cmdstan_path / "stan" / "lib" / "stan_math" / "lib" / "tbb"
                if tbb_path.is_dir():
                    env["PATH"] = str(tbb_path) + os.pathsep + env.get("PATH", "")
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                raise NativeAdapterError("native_timeout", f"{method} exceeded {self.timeout_seconds} seconds.") from exc

            if completed.returncode != 0:
                message = (completed.stderr or completed.stdout or "R backend failed").strip()
                reason = "dependency_unavailable" if "there is no package called" in message.lower() else "native_execution_failed"
                raise NativeAdapterError(reason, message[-4000:])
            output_path = work_dir / "native_output.csv"
            if not output_path.is_file():
                raise NativeAdapterError("native_output_missing", "R backend did not create native_output.csv.")
            output = pd.read_csv(output_path, encoding="utf-8")
            diagnostics_path = work_dir / "diagnostics.json"
            diagnostics: dict = {}
            if diagnostics_path.is_file():
                with diagnostics_path.open(encoding="utf-8") as handle:
                    diagnostics = json.load(handle)
            diagnostics["stdout"] = completed.stdout
            diagnostics["stderr"] = completed.stderr
            output.attrs["diagnostics"] = diagnostics
            return output
