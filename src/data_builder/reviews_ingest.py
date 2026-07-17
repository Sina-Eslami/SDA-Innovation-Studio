from pathlib import Path
import pandas as pd

class ReviewsTabular:
    def __init__(self):
        self.project_root = Path.cwd().parent.parent

        self.txt_file_path = (
            self.project_root / "data" / "static" / "amazon_reviews.txt"
        )

        self.df = None

    def tabular(self):
        print("Reading:", self.txt_file_path)
        print("Exists:", self.txt_file_path.exists())

        self.df = pd.read_csv(
            self.txt_file_path,
            sep="\t",
            quotechar='"',
            encoding="utf-8"
        )

        output_dir = self.project_root / "data" / "static"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(
            output_dir / "raw-reviews.csv",
            index=False
        )

        return self.df