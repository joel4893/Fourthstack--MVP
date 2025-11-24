import os
from typing import Dict, Optional

import torch
import pandas as pd
import os
from typing import Dict, Optional

import torch
import pandas as pd
from scipy.stats import ks_2samp

try:
    from .hybrid_generator import HybridGenerator
except Exception:
    # allow running the module directly (e.g. via importlib) or from package
    from fourthstack.synthetic_engine.hybrid_generator import HybridGenerator


class ExperimentRunner:
    """Minimal experiment harness to generate synthetic data and report similarity.

    This file provides a lightweight runner intended for smoke tests and
    small local experiments.
    """

    def __init__(self, real_data_path: str):
        # load tabular JSONL data
        self.real_df = pd.read_json(real_data_path, lines=True)
        self.data_dim = int(self.real_df.shape[1])
        # instantiate a hybrid generator matching the column dimensionality
        self.model = HybridGenerator(noise_dim=32, data_dim=self.data_dim)
        self.model.eval()

    def generate(self, n: int = 200, device: Optional[str] = None) -> pd.DataFrame:
        """Generate `n` synthetic rows and return a DataFrame with the same columns
        as the real dataset.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        noise = torch.randn(n, 32, device=device)
        with torch.no_grad():
            synth = self.model(noise).cpu().numpy()

        return pd.DataFrame(synth, columns=self.real_df.columns)

    def similarity_report(self, synthetic_df: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Return a dict mapping column -> KS statistic (float) or None if not
        applicable (e.g., non-numeric columns or errors).
        """
        report: Dict[str, Optional[float]] = {}
        for col in self.real_df.columns:
            try:
                real_col = pd.to_numeric(self.real_df[col], errors="coerce").dropna()
                synth_col = pd.to_numeric(synthetic_df[col], errors="coerce").dropna()
                if len(real_col) == 0 or len(synth_col) == 0:
                    report[col] = None
                    continue

                ks = ks_2samp(real_col, synth_col).statistic
                report[col] = float(ks)
            except Exception:
                report[col] = None

        return report


if __name__ == "__main__":
    # try to find a sensible default preprocessed dataset in the repo
    default_path = os.path.join(os.path.dirname(__file__), "..", "data", "dummy_thinfile.preprocessed.jsonl")
    default_path = os.path.normpath(default_path)
    if not os.path.exists(default_path):
        # fallback: check known repo path
        default_path = os.path.join(os.getcwd(), "fourthstack", "data", "dummy_thinfile.preprocessed.jsonl")

    if os.path.exists(default_path):
        runner = ExperimentRunner(default_path)
        synth_df = runner.generate(200)
        print("Synthetic sample head:")
        print(synth_df.head())

        report = runner.similarity_report(synth_df)
        print("\n=== KS Divergence Report ===")
        for k, v in report.items():
            print(f"{k}: {v}")
    else:
        print("No default preprocessed dataset found; please supply a path to run the experiment runner.")