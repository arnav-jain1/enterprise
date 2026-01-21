import numpy as np

def get_angle(frame, u, o, v):
    u = frame[u] - frame[o]
    v = frame[v] - frame[o]

    u_magnitude = np.linalg.norm(u)
    v_magnitude = np.linalg.norm(v)

    cos_theta = np.dot(u, v) / (u_magnitude * v_magnitude)
    cos_theta_clip = np.clip(cos_theta, -1.0, 1.0)

    theta_radians = np.arccos(cos_theta_clip)

    theta_degrees = theta_radians * (180 / np.pi)

    return theta_degrees


def calculate_elbow_angles(frame):
    return {"right_elbow" : get_angle(frame, 16, 14, 12), "left_elbow" : get_angle(frame, 15, 13, 11)}


def calculate_shoulder_angles(frame):
    return {"right_shoulder" : get_angle(frame, 14, 12, 24), "left_shoulder" : get_angle(frame, 13, 11, 23)}

def calculate_torso_angles(frame):
    return {"right_torso" : get_angle(frame, 12, 24, 26), "left_torso" : get_angle(frame, 11, 23, 25)}

def calculate_wrist_angles(frame):
    return {"right_wrist" : get_angle(frame, 20, 16, 14), "left_wrist" : get_angle(frame, 19, 15, 13)}


# with np.load('../barbell_npz/barbell_0.npz') as data
#     frame = data["arr_0"][0]
#     print(calculate_elbow_angles(frame))