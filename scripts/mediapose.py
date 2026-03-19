from pathlib import Path
from pipelines.video_to_npz import video_to_npz
from geometry import get_specified_angle_array, get_all_angles_arrays
import matplotlib.pyplot as plt

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

        count += 1
    test = []
    test_vel = []
    for frame in frames:
        test.append(frame.angles["right_elbow"])
        test_vel.append(frame.velocity["right_elbow"])

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(test)
    ax[1].plot(test_vel)
    ax[1].set_title("Right Elbow Velocity Over Time")
    ax[0].set_xlabel("Frame")
    ax[0].set_ylabel("Angle")
    ax[1].set_xlabel("Frame")
    ax[1].set_ylabel("Velocity")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()