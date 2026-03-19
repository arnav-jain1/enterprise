import numpy as np

# ------------------------------------------------
# Feature schema (EDIT THIS to add new features)
# ------------------------------------------------
FEATURE_SCHEMA = {
    "angles": [
        "right_elbow", "left_elbow",
        "right_shoulder", "left_shoulder",
        "right_torso", "left_torso",
        "right_wrist", "left_wrist",
    ],

    # Velocity mirrors angles
    "velocity": [
        "right_elbow", "left_elbow",
        "right_shoulder", "left_shoulder",
        "right_torso", "left_torso",
        "right_wrist", "left_wrist",
    ],

    # Acceleration mirrors velocity
    "acceleration": [
        "right_elbow", "left_elbow",
        "right_shoulder", "left_shoulder",
        "right_torso", "left_torso",
        "right_wrist", "left_wrist",
    ],

    # Motion (currently only shoulder rotation)
    "motion": [
        "right_shoulder", "left_shoulder",
    ],

    # Displacement (currently elbows only)
    "displacement": [
        "right_elbow", "left_elbow",
    ],
}


# ------------------------------------------------
# Helper: safely extract features
# ------------------------------------------------
def extract_features(frame_dict, keys):
    frame_dict = frame_dict or {}
    return [frame_dict.get(k, 0.0) for k in keys]


# ------------------------------------------------
# Main conversion function
# ------------------------------------------------
def frames_to_numpy(frames):
    """
    Convert a list of Frame objects into NumPy arrays.

    Uses FEATURE_SCHEMA to dynamically extract features.
    """

    if len(frames) == 0:
        raise ValueError("No frames to convert.")

    landmarks_list = []
    angles_list = []
    velocity_list = []
    acceleration_list = []
    motion_list = []
    displacement_list = []

    for frame in frames:

        # -----------------------------
        # Landmarks
        # -----------------------------
        landmarks_list.append(frame.landmarks)

        # -----------------------------
        # Feature extraction (schema-driven)
        # -----------------------------
        angles_list.append(
            extract_features(frame.angles, FEATURE_SCHEMA["angles"])
        )

        velocity_list.append(
            extract_features(frame.velocity, FEATURE_SCHEMA["velocity"])
        )

        acceleration_list.append(
            extract_features(frame.acceleration, FEATURE_SCHEMA["acceleration"])
        )

        motion_list.append(
            extract_features(frame.motion, FEATURE_SCHEMA["motion"])
        )

        displacement_list.append(
            extract_features(frame.displacement, FEATURE_SCHEMA["displacement"])
        )

    # -----------------------------
    # Convert to NumPy arrays
    # -----------------------------
    return (
        np.stack(landmarks_list).astype(np.float32),
        np.array(angles_list, dtype=np.float32),
        np.array(velocity_list, dtype=np.float32),
        np.array(acceleration_list, dtype=np.float32),
        np.array(motion_list, dtype=np.float32),
        np.array(displacement_list, dtype=np.float32),
    )