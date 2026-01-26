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