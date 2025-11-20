from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    test_loss: float
    test_top1_acc: float
    test_top5_acc: float
    mAP: float
    similarity: float

    @staticmethod
    def average(results: list[ExperimentResult]) -> ExperimentResult:
        n = len(results)
        if n == 0:
            return ExperimentResult(
                test_loss=float("nan"),
                test_top1_acc=float("nan"),
                test_top5_acc=float("nan"),
                mAP=float("nan"),
                similarity=float("nan"),
            )
        return ExperimentResult(
            test_loss=sum(r.test_loss for r in results) / n,
            test_top1_acc=sum(r.test_top1_acc for r in results) / n,
            test_top5_acc=sum(r.test_top5_acc for r in results) / n,
            mAP=sum(r.mAP for r in results) / n,
            similarity=sum(r.similarity for r in results) / n,
        )

    def write_json(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(
                {
                    "test_loss": self.test_loss,
                    "test_top1_acc": self.test_top1_acc,
                    "test_top5_acc": self.test_top5_acc,
                    "mAP": self.mAP,
                    "similarity": self.similarity,
                },
                f,
                indent=4,
            )


class Summarizer:
    def __init__(
        self,
        exp_dir: str,
        glob_pattern: str,
        output_dir: str,
        exp_results: list[ExperimentResult] = [],
    ):
        self.exp_dir = exp_dir
        self.glob_pattern = glob_pattern
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.exp_results = exp_results if exp_results else self.read_results()

        self.summary = ExperimentResult.average(self.exp_results)

    def read_results(self) -> list[ExperimentResult]:
        """
        Recursively finds test_results.json files where the parent directory
        matches the keyword.
        """
        results: list[ExperimentResult] = []

        dirs = glob.glob(os.path.join(self.exp_dir, f"*{self.glob_pattern}*"))
        if not dirs:
            print(
                f"No directories found matching pattern '{self.glob_pattern}' in {self.exp_dir}"
            )
            return results

        ## Find all test_results.json files recursively that are contained somewhere in the dirs filtered above
        files = []
        for d in dirs:
            files.extend(
                glob.glob(os.path.join(d, "**", "test_results.json"), recursive=True)
            )

        for file_path in files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)[0]
                    result = ExperimentResult(
                        test_loss=data.get("test_loss", float("nan")),
                        test_top1_acc=data.get("test_top1_acc", float("nan")),
                        test_top5_acc=data.get("test_top5_acc", float("nan")),
                        mAP=data.get("mAP", float("nan")),
                        similarity=data.get("similarity", float("nan")),
                    )
                    results.append(result)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        return results

    def print_summary(self):
        print(f"Summary for experiments matching '{self.glob_pattern}':")
        print(f"  Test Loss: {self.summary.test_loss:.4f}")
        print(f"  Test Top-1 Accuracy: {self.summary.test_top1_acc:.2f}%")
        print(f"  Test Top-5 Accuracy: {self.summary.test_top5_acc:.2f}%")
        print(f"  mAP: {self.summary.mAP:.4f}")
        print(f"  Similarity: {self.summary.similarity:.4f}")

    def write_summary(self):
        summary_path = os.path.join(
            self.output_dir, f"summary_{self.glob_pattern.replace('*', '_')}.json"
        )
        self.summary.write_json(summary_path)
        print(f"Summary written to {summary_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Summarize experiment results.")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="exp",
        help="Directory containing experiment results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_summary",
        help="Directory to save summary results.",
    )
    parser.add_argument(
        "--glob_pattern",
        type=str,
        default="eeg_inter*ubp*",
        help="Glob pattern for subject experiments.",
    )

    args = parser.parse_args()

    summarizer = Summarizer(
        exp_dir=args.exp_dir, glob_pattern=args.glob_pattern, output_dir=args.output_dir
    )
    summarizer.write_summary()


if __name__ == "__main__":
    main()
