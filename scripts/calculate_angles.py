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


with np.load('../barbell_npz/barbell_0.npz') as data:
    frame = data["arr_0"][0]
    print(calculate_elbow_angles(frame))