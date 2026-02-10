import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt

def vector(p1, p2):
    """Vector from p2 → p1 (2D)."""
    return p1[:2] - p2[:2]

def signed_angle(u, v):
    """
    Signed angle from vector u to v in degrees.
    Range: (-180, 180]
    """
    cross = u[0] * v[1] - u[1] * v[0]
    dot   = u[0] * v[0] + u[1] * v[1]
    return np.degrees(np.arctan2(cross, dot))

def joint_angle(landmarks, U, O, V):
    """
    Angle at joint O formed by points U–O–V.
    """
    u = vector(landmarks[U], landmarks[O])
    v = vector(landmarks[V], landmarks[O])
    return signed_angle(u, v)

def segment_orientation(landmarks, A, B):
    """
    Orientation of segment B→A relative to horizontal.
    """
    v = vector(landmarks[A], landmarks[B])
    return np.degrees(np.arctan2(v[1], v[0]))

def segment_motion_angle(prev_landmarks, curr_landmarks, A, B):
    """
    Signed rotation of segment B→A between frames.
    """
    u = vector(prev_landmarks[A], prev_landmarks[B])
    v = vector(curr_landmarks[A], curr_landmarks[B])
    return signed_angle(u, v)

def point_displacement(prev_landmarks, curr_landmarks, idx):
    """
    Euclidean displacement of a landmark between frames.
    """
    return np.linalg.norm(
        curr_landmarks[idx][:2] - prev_landmarks[idx][:2]
    )

def get_shoulder_angle_array(frames, side="right"):
    key = f"{side}_shoulder"
    return [
        frame.angles.get(key, 0.0)
        for frame in frames
    ]

def check_savgol_filter(array):
    fig, ax = plt.subplots(2, 2, figsize=(10, 4))
    ax[0, 0].plot(array, label='Shoulder Angle')
    ax[0, 0].set_title('Shoulder Angle Over Time')
    ax[0, 0].set_xlabel('Frame')
    ax[0, 0].set_ylabel('Angle (degrees)')
        
    ax[1, 0].plot(savgol_filter(array, 5, 2), label='Default', color='orange')
    ax[1, 0].set_title('Default')
    ax[1, 0].set_xlabel('Frame')
    ax[1, 0].set_ylabel('Angle (degrees)')

    ax[0, 1].plot(savgol_filter(array, 7, 2), label='Increased Window', color='orange')
    ax[0, 1].set_title('Increased Window')
    ax[0, 1].set_xlabel('Frame')
    ax[0, 1].set_ylabel('Angle (degrees)')

    ax[1, 1].plot(savgol_filter(array, 5, 4), label='Increased Order', color='orange')
    ax[1, 1].set_title('Increased Order')
    ax[1, 1].set_xlabel('Frame')
    ax[1, 1].set_ylabel('Angle (degrees)')

    plt.tight_layout()

    return fig, ax