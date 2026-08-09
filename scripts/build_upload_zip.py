from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


INCLUDED_DIRECTORIES = (
    "config",
    "data_pipeline",
    "deployment",
    "docs",
    "evaluation",
    "experiments",
    "models",
    "scripts",
    "tests",
    "training",
    "utils",
    "validation",
)
INCLUDED_FILES = (
    ".gitignore",
    ".python-version",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "run_system.py",
    "uv.lock",
)
EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".uv-cache",
    "datasets",
    "evaluation_outputs",
    "logs",
}
EXCLUDED_FILES = {
    "Hybrid_soc_Estimator_code.zip",
    "best_model.pt",
}


def _is_allowed(relative_path: Path) -> bool:
    if any(part in EXCLUDED_DIRECTORIES for part in relative_path.parts):
        return False
    if relative_path.name in EXCLUDED_FILES:
        return False
    if relative_path.suffix in {".pyc", ".pyo"}:
        return False
    return True


def project_files(project_root: str | Path) -> list[Path]:
    root = Path(project_root).resolve()
    candidates: list[Path] = []

    for filename in INCLUDED_FILES:
        path = root / filename
        if path.is_file():
            candidates.append(path)

    for directory_name in INCLUDED_DIRECTORIES:
        directory = root / directory_name
        if directory.is_dir():
            candidates.extend(path for path in directory.rglob("*") if path.is_file())

    return sorted(
        path
        for path in candidates
        if _is_allowed(path.relative_to(root))
    )


def build_archive(
    project_root: str | Path,
    archive_path: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    output = (
        Path(archive_path).resolve()
        if archive_path is not None
        else root / "Hybrid_soc_Estimator_code.zip"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive_root = root.name
        for path in project_files(root):
            relative_path = path.relative_to(root)
            archive.write(path, Path(archive_root) / relative_path)

    return output


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    archive_path = build_archive(project_root)
    print(f"Upload archive written to {archive_path}")
    print(f"Files included: {len(project_files(project_root))}")


if __name__ == "__main__":
    main()
