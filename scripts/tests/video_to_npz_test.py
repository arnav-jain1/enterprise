from pathlib import Path
from scripts.pipelines.video_to_npz import video_to_npz
import matplotlib.pyplot as plt

def main():
    BASE_DIR = Path(__file__).resolve().parents[2]  # goes to Enterprise/

    directory_path = BASE_DIR / "raw_data" / "bench_press"
    count = 0

    for item in directory_path.iterdir():
        video_path = str(item)

        output_path = BASE_DIR / "npz" / "bench_press_npz" / f"bench_press_{count}"

        frames = video_to_npz(
            video_path,
            output_path
        )

        count += 1
    test = []
    test_vel = []

    # fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    # ax[0].plot(test)
    # ax[1].plot(test_vel)
    # ax[1].set_title("Right Elbow Velocity Over Time")
    # ax[0].set_xlabel("Frame")
    # ax[0].set_ylabel("Angle")
    # ax[1].set_xlabel("Frame")
    # ax[1].set_ylabel("Velocity")
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()