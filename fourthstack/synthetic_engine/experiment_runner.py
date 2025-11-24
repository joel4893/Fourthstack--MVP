import torch
import pandas as pd
import numpy as np
from hybrid_generator import HybridGenerator
from scipy.stats import ks_2samp

class ExperimentRunner:
    def __init__(self, real_data_path):
            self.real_df = pd.read_json(real_data_path, lines=True)
                    self.data_dim = self.real_df.shape[1]
                            self.model = HybridGenerator(noise_dim=32, data_dim=self.data_dim)

                                def generate(self, n=200):
                                        noise = torch.randn(n, 32)
                                                with torch.no_grad():
                                                            synth = self.model(noise).numpy()
                                                                    return pd.DataFrame(synth, columns=self.real_df.columns)

                                                                        def similarity_report(self, synthetic_df):
                                                                                report = {}
                                                                                        for col in self.real_df.columns:
                                                                                                    try:
                                                                                                                    ks = ks_2samp(self.real_df[col], synthetic_df[col]).statistic
                                                                                                                                    report[col] = float(ks)
                                                                                                                                                except:
                                                                                                                                                                report[col] = None
                                                                                                                                                                        return report


                                                                                                                                                                        if __name__ == "__main__":
                                                                                                                                                                            runner = ExperimentRunner("fourthstack/data/dummy_thinfile_preprocessed.jsonl")
                                                                                                                                                                                synth_df = runner.generate(200)
                                                                                                                                                                                    print("Synthetic sample head:")
                                                                                                                                                                                        print(synth_df.head())

                                                                                                                                                                                            report = runner.similarity_report(synth_df)
                                                                                                                                                                                                print("\n=== KS Divergence Report ===")
                                                                                                                                                                                                    for k, v in report.items():
                                                                                                                                                                                                            print(f"{k}: {v}")