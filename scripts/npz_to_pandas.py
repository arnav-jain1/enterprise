import numpy as np
import pandas as pd


# ============================================================
# NPZ → Pandas DataFrame Conversion
# ============================================================

def npz_to_dataframe(npz_path):
    """
    Convert a saved NPZ feature file into a pandas DataFrame.

    Each array inside the NPZ becomes a column group in the
    resulting DataFrame.

    Parameters
    ----------
    npz_path : str or Path
        Path to the .npz file

    Returns
    -------
    pd.DataFrame
        Combined dataframe of all features (time aligned)
    """

    data = np.load(npz_path)
    dfs = []

    for key in data.files:
        arr = data[key]

        # ------------------------------------------------
        # Skip raw landmark data unless explicitly needed
        # ------------------------------------------------
        if key == "landmarks":
            continue

        # ------------------------------------------------
        # 2D arrays: (T, F) → multiple feature columns
        # ------------------------------------------------
        if arr.ndim == 2:
            df = pd.DataFrame(arr)
            df.columns = [f"{key}_{i}" for i in range(arr.shape[1])]
            dfs.append(df)

        # ------------------------------------------------
        # 1D arrays: (T,) → single feature column
        # ------------------------------------------------
        elif arr.ndim == 1:
            dfs.append(pd.DataFrame({key: arr}))

    # ------------------------------------------------
    # Combine all features along the time axis
    # ------------------------------------------------
    return pd.concat(dfs, axis=1)


# ============================================================
# Frame Objects → NumPy Arrays
# ============================================================

def frames_to_numpy(frames):
    """
    Convert a list of Frame objects into NumPy arrays for
    saving to disk.

    Extracted features include:
    - landmarks
    - angles
    - velocities
    - accelerations
    - motion metrics
    - displacement metrics

    Parameters
    ----------
    frames : list[Frame]

    Returns
    -------
    tuple of numpy arrays
    """

    landmarks_list = []
    angles_list = []
    velocity_list = []
    acceleration_list = []
    motion_list = []
    displacement_list = []

    for frame in frames:

        # ------------------------------------------------
        # Raw pose landmarks
        # ------------------------------------------------
        landmarks_list.append(frame.landmarks)

        # ------------------------------------------------
        # Joint angles
        # ------------------------------------------------
        angles_list.append([
            frame.angles.get("right_elbow", 0.0),
            frame.angles.get("left_elbow", 0.0),
            frame.angles.get("right_shoulder", 0.0),
            frame.angles.get("left_shoulder", 0.0),
        ])

        # ------------------------------------------------
        # Velocities
        # ------------------------------------------------
        velocity_list.append([
            frame.velocity.get("right_elbow", 0.0),
            frame.velocity.get("left_elbow", 0.0),
        ])

        # ------------------------------------------------
        # Accelerations
        # ------------------------------------------------
        acceleration_list.append([
            frame.acceleration.get("right_elbow", 0.0),
            frame.acceleration.get("left_elbow", 0.0),
        ])

        # ------------------------------------------------
        # Segment motion (rotation between frames)
        # ------------------------------------------------
        motion_list.append([
            frame.motion.get("right_shoulder", 0.0),
            frame.motion.get("left_shoulder", 0.0),
        ])

        # ------------------------------------------------
        # Landmark displacement
        # ------------------------------------------------
        displacement_list.append([
            frame.displacement.get("right_elbow", 0.0),
            frame.displacement.get("left_elbow", 0.0),
        ])

    # ------------------------------------------------
    # Convert lists → NumPy arrays
    # ------------------------------------------------
    return (
        np.stack(landmarks_list).astype(np.float32),
        np.array(angles_list, dtype=np.float32),
        np.array(velocity_list, dtype=np.float32),
        np.array(acceleration_list, dtype=np.float32),
        np.array(motion_list, dtype=np.float32),
        np.array(displacement_list, dtype=np.float32),
    )