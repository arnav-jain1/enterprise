import numpy as np
import pandas as pd


def npz_to_dataframe(npz_path):
    data = np.load(npz_path)

    dfs = []

    for key in data.files:
        arr = data[key]

        # Skip landmarks unless you explicitly want them
        if key == "landmarks":
            continue

        # 2D arrays: (T, F)
        if arr.ndim == 2:
            df = pd.DataFrame(arr)
            df.columns = [f"{key}_{i}" for i in range(arr.shape[1])]
            dfs.append(df)

        # 1D arrays: (T,)
        elif arr.ndim == 1:
            dfs.append(pd.DataFrame({key: arr}))

    # Concatenate on time axis
    return pd.concat(dfs, axis=1)

def frames_to_numpy(frames):
    landmarks_list = []
    angles_list = []
    velocity_list = []
    acceleration_list = []
    motion_list = []
    displacement_list = []

    for frame in frames:
        landmarks_list.append(frame.landmarks)

        angles_list.append([
            frame.angles.get("right_elbow", 0.0),
            frame.angles.get("left_elbow", 0.0),
            frame.angles.get("right_shoulder", 0.0),
            frame.angles.get("left_shoulder", 0.0),
        ])

        velocity_list.append([
            frame.velocity.get("right_elbow", 0.0),
            frame.velocity.get("left_elbow", 0.0),
        ])

        acceleration_list.append([
            frame.acceleration.get("right_elbow", 0.0),
            frame.acceleration.get("left_elbow", 0.0),
        ])

        motion_list.append([
            frame.motion.get("right_shoulder", 0.0),
            frame.motion.get("left_shoulder", 0.0),
        ])

        displacement_list.append([
            frame.displacement.get("right_elbow", 0.0),
            frame.displacement.get("left_elbow", 0.0),
        ])

    return (
        np.stack(landmarks_list).astype(np.float32),
        np.array(angles_list, dtype=np.float32),
        np.array(velocity_list, dtype=np.float32),
        np.array(acceleration_list, dtype=np.float32),
        np.array(motion_list, dtype=np.float32),
        np.array(displacement_list, dtype=np.float32),
    )