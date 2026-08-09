from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from scripts.build_upload_zip import build_archive


class UploadArchiveTests(unittest.TestCase):
    def test_archive_contains_source_and_excludes_generated_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "Hybrid_soc_Estimator"
            root.mkdir()
            for relative_path in (
                "run_system.py",
                "README.md",
                "config/config.py",
                "evaluation/metrics.py",
                "models/lstm_soc_model.py",
                "tests/test_metrics.py",
            ):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("source", encoding="utf-8")

            for relative_path in (
                "datasets/processed/X_test.npy",
                "evaluation_outputs/metrics.json",
                "logs/training.log",
                "models/best_model.pt",
                ".venv/pyvenv.cfg",
            ):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("generated", encoding="utf-8")

            archive_path = build_archive(root)

            with ZipFile(archive_path) as archive:
                names = set(archive.namelist())

            self.assertIn("Hybrid_soc_Estimator/run_system.py", names)
            self.assertIn("Hybrid_soc_Estimator/evaluation/metrics.py", names)
            self.assertIn("Hybrid_soc_Estimator/tests/test_metrics.py", names)
            self.assertNotIn("Hybrid_soc_Estimator/datasets/processed/X_test.npy", names)
            self.assertNotIn("Hybrid_soc_Estimator/evaluation_outputs/metrics.json", names)
            self.assertNotIn("Hybrid_soc_Estimator/logs/training.log", names)
            self.assertNotIn("Hybrid_soc_Estimator/models/best_model.pt", names)
            self.assertNotIn("Hybrid_soc_Estimator/.venv/pyvenv.cfg", names)


if __name__ == "__main__":
    unittest.main()
