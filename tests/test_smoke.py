import tempfile
import unittest
from pathlib import Path

from src.io.generate_data import generate_rules, save_rules_json
from src.main import run_pipeline
from src.ga.engine import GAConfig
from src.evaluation.fitness import FitnessConfig


class SmokeTest(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rules_path = td / "rules.json"
            save_rules_json(generate_rules(120, seed=2), rules_path)

            out_dir = td / "out"
            run_pipeline(
                rules_path=rules_path,
                out_dir=out_dir,
                n_packets=800,
                seed=2,
                ga_cfg=GAConfig(population_size=20, generations=10),
                fit_cfg=FitnessConfig(min_coverage=0.95),
                student_name="Test",
            )

            self.assertTrue((out_dir / "metrics.json").exists())
            self.assertTrue((out_dir / "report.pdf").exists())


if __name__ == "__main__":
    unittest.main()
