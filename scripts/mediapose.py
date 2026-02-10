from pathlib import Path
from scripts.pipelines.video_to_npz import video_to_npz
from scripts.geometry import check_savgol_filter, get_shoulder_angle_array
from matplotlib import pyplot as plt

def main():
    directory_path = Path('../barbell')
    count = 0

    for item in directory_path.iterdir():
        video_path = str(item)
        output_path = f"../barbell_npz/barbell_{count}"

        frames = video_to_npz(
            video_path,
            output_path
        )

        fig, ax = check_savgol_filter(
        get_shoulder_angle_array(frames, side="right"))
        plt.show()

        count += 1



if __name__ == "__main__":
    main()