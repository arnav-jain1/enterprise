import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt

def get_angle(frame, U, O, V):
    u = frame[U][:2] - frame[O][:2]
    v = frame[V][:2] - frame[O][:2]

    cross = u[0]*v[1] - u[1]*v[0]
    dot   = u[0]*v[0] + u[1]*v[1]

    return np.degrees(np.arctan2(cross, dot))

def get_motion_angle(prev, curr, U, O):
    u = prev[U][:2] - prev[O][:2]
    v = curr[U][:2] - curr[O][:2]

    cross = u[0]*v[1] - u[1]*v[0]
    dot   = u[0]*v[0] + u[1]*v[1]

    return np.degrees(np.arctan2(cross, dot))

def get_shoulder_angle_array(frames):
    angles = []
    for frame in frames:
        angles.append(frame.get_angles()[8])
    return angles

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
    


def main():
    with np.load("../barbell_npz/barbell_0.npz") as data:
        print(data['angles'])

if __name__ == '__main__':
    main()