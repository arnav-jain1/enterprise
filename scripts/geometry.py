import numpy as np
from scipy.signal import savgol_filter, medfilt
from matplotlib import pyplot as plt


# ============================================================
# Vector / Geometry Utilities
# ============================================================

def vector(p1, p2):
    """
    Compute the 2D vector from point p2 → p1.

    Parameters
    ----------
    p1 : ndarray
    p2 : ndarray

    Returns
    -------
    ndarray
        2D vector (x, y)
    """
    return p1[:2] - p2[:2]


def signed_angle(u, v):
    """
    Compute the signed angle from vector u → v.

    Range: (-180°, 180°]

    Parameters
    ----------
    u : ndarray
        First vector
    v : ndarray
        Second vector

    Returns
    -------
    float
        Angle in degrees
    """
    cross = u[0] * v[1] - u[1] * v[0]
    dot   = u[0] * v[0] + u[1] * v[1]

    return np.degrees(np.arctan2(cross, dot))


# ============================================================
# Joint / Segment Angles
# ============================================================

def joint_angle(landmarks, U, O, V):
    """
    Compute angle at joint O formed by points U–O–V.

    Example:
        elbow angle = shoulder–elbow–wrist

    Parameters
    ----------
    landmarks : ndarray
        Pose landmarks (33 x 4)
    U : int
        Upstream landmark index
    O : int
        Joint landmark index
    V : int
        Downstream landmark index

    Returns
    -------
    float
        Joint angle in degrees
    """
    u = vector(landmarks[U], landmarks[O])
    v = vector(landmarks[V], landmarks[O])

    angle = signed_angle(u, v)

    return 180 - abs(angle)


def segment_orientation_horizontal(landmarks, A, B):
    """
    Orientation of segment B → A relative to horizontal.

    Example:
        torso orientation = hip → shoulder

    Returns
    -------
    float
        Angle in degrees
    """
    v = vector(landmarks[A], landmarks[B])

    return np.degrees(np.arctan2(v[1], v[0]))


def segment_orientation_vertical(landmarks, A, B):
    v = vector(landmarks[A], landmarks[B])

    return abs(np.degrees(np.arctan2(v[0], v[1])))

def segment_motion_angle(prev_landmarks, curr_landmarks, A, B):
    """
    Signed rotation of segment B → A between two frames.

    Useful for detecting motion of limbs between frames.

    Returns
    -------
    float
        Rotation angle in degrees
    """
    u = vector(prev_landmarks[A], prev_landmarks[B])
    v = vector(curr_landmarks[A], curr_landmarks[B])

    return signed_angle(u, v)


def uniform_angle(angles, a, b):
    return (angles[a] + angles[b]) / 2


# ============================================================
# Motion Metrics
# ============================================================

def point_displacement(prev_landmarks, curr_landmarks, idx):
    """
    Euclidean displacement of a single landmark between frames.

    Parameters
    ----------
    prev_landmarks : ndarray
    curr_landmarks : ndarray
    idx : int
        Landmark index

    Returns
    -------
    float
        Distance moved between frames
    """
    return np.linalg.norm(
        curr_landmarks[idx][:2] - prev_landmarks[idx][:2]
    )


# ============================================================
# Frame Angle Extraction
# ============================================================

def get_specified_angle_array(frames, angle="elbow", side="right"):
    """
    Extract a time-series array of a specific joint angle
    across all frames.

    Parameters
    ----------
    frames : list[Frame]
    angle : str
        Joint name (e.g. "elbow", "shoulder")
    side : str
        "left", "right", or "average"

    Returns
    -------
    list
        Angle values per frame
    """

    # Average left/right joint
    if side == "average":
        return [
            (frame.angles.get(f"right_{angle}", 0.0) +
             frame.angles.get(f"left_{angle}", 0.0)) / 2
            for frame in frames
        ]

    # Single side
    key = f"{side}_{angle}"
    return [
        frame.angles.get(key, 0.0)
        for frame in frames
    ]


# ============================================================
# Angle Time-Series Extraction + Filtering
# ============================================================

def get_all_angles_arrays(frames):
    """
    Extract all angle time-series from frames and apply
    filtering (median + Savitzky-Golay).

    Pipeline:
        raw angles
        → median filter (remove spikes)
        → Savitzky-Golay (smooth curve)

    Parameters
    ----------
    frames : list[Frame]

    Returns
    -------
    dict
        {angle_name : filtered_angle_array}
    """

    angles = {}

    # ----------------------------------------------
    # Build time-series arrays
    # ----------------------------------------------
    for frame in frames:
        for key, value in frame.angles.items():

            if key not in angles:
                angles[key] = []

            angles[key].append(value)

    # ----------------------------------------------
    # Apply filtering to each angle signal
    # ----------------------------------------------
    for key in angles:

        arr = np.array(angles[key])

        for i in range(1, len(arr)):
            if abs(arr[i] - arr[i-1]) > 60:  # elbow cannot change this fast
                arr[i] = arr[i-1]

        angles[key] = savgol_filter(
            medfilt(arr, kernel_size=3),
            polyorder=3,
            window_length=9
        )
        angles[key] = savgol_filter(
            medfilt(np.array(angles[key]), kernel_size=3),
            polyorder=3,
            window_length=9
        )

    return angles