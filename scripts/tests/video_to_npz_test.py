from pathlib import Path
from scripts.pipelines.video_to_evaluation import evaluation_pipeline
import matplotlib.pyplot as plt

def main():
    BASE_DIR = Path(__file__).resolve().parents[2]  # goes to Enterprise/

    directory_path = BASE_DIR / "raw_data" / "lat_pulldown"
    count = 0

    for item in directory_path.iterdir():
        video_path = str(item)

        output_path = BASE_DIR / "npz" / "barbell_npz" / f"lat_pulldown"

        frames = evaluation_pipeline(
            video_path,
            output_path
        )

        count += 1

if __name__ == "__main__":
    main()