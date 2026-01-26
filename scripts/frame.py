from calculate_angles import get_angle
import numpy as np

class Frame:
    def __init__(self, landmarks):
        self.landmarks = landmarks

        self.theta_right_elbow = None
        self.theta_left_elbow = None

        self.theta_right_shoulder = None
        self.theta_left_shoulder = None

        self.theta_right_torso = None
        self.theta_left_torso = None

        self.theta_right_wrist = None
        self.theta_left_wrist = None

        self.theta_hip = None

        self.theta_knee = None

        self.right_elbow_velocity = None
        self.left_elbow_velocity = None

        self.right_elbow_acceleration = None
        self.left_elbow_acceleration = None

    def calculate_elbow_angles(self):
        self.theta_right_elbow = get_angle(self.landmarks, 16, 14, 12)
        self.theta_left_elbow = get_angle(self.landmarks, 15, 13, 11)

    def calculate_shoulder_angles(self):
        self.theta_right_shoulder = get_angle(self.landmarks, 14, 12, 24)
        self.theta_left_shoulder = get_angle(self.landmarks, 13, 11, 23)

    def calculate_torso_angles(self):
        self.theta_right_torso = get_angle(self.landmarks, 12, 24, 26)
        self.theta_left_torso = get_angle(self.landmarks, 11, 23, 25)

    def calculate_wrist_angles(self):
        self.theta_right_wrist = get_angle(self.landmarks, 20, 16, 14)
        self.theta_left_wrist = get_angle(self.landmarks, 19, 15, 13)

    def calculate_angles(self):
        self.calculate_elbow_angles()
        self.calculate_shoulder_angles()
        self.calculate_torso_angles()
        self.calculate_wrist_angles()

    def get_angles(self):
        return np.array([
                    self.theta_right_elbow,
                    self.theta_left_elbow,
                    self.theta_right_shoulder,
                    self.theta_left_shoulder,
                    self.theta_right_torso,
                    self.theta_left_torso,
                    self.theta_right_wrist,
                    self.theta_left_wrist,
                    self.theta_hip,
                    self.theta_knee
                ], dtype=np.float32)
    
    def calculate_angular_velocity(self, prev, fps):
        if prev == None:
            self.left_elbow_velocity = 0
            self.right_elbow_velocity = 0
            return

        self.left_elbow_velocity = (self.theta_left_elbow - prev.theta_left_elbow) * fps
        self.right_elbow_velocity = (self.theta_right_elbow - prev.theta_right_elbow) * fps

    def calculate_angular_acceleration(self, prev, fps):
        if prev == None:
            self.left_elbow_acceleration = 0.0
            self.right_elbow_acceleration = 0.0
            return

        self.left_elbow_acceleration = (self.left_elbow_velocity - prev.left_elbow_velocity) * fps
        self.right_elbow_acceleration = (self.right_elbow_velocity - prev.right_elbow_velocity) * fps
    
    def sanity_check(self, prev, fps):
        print(self.left_elbow_velocity, prev.theta_left_elbow, fps)

    
    def __str__(self):
        def fmt(val):
            return f"{val:.2f}" if val is not None else "None"

        return (
            "Frame Angles:\n"
            f"  Elbow    | Right: {fmt(self.theta_right_elbow)} | Left: {fmt(self.theta_left_elbow)}\n"
            f"  Shoulder | Right: {fmt(self.theta_right_shoulder)} | Left: {fmt(self.theta_left_shoulder)}\n"
            f"  Torso    | Right: {fmt(self.theta_right_torso)} | Left: {fmt(self.theta_left_torso)}\n"
            f"  Wrist    | Right: {fmt(self.theta_right_wrist)} | Left: {fmt(self.theta_left_wrist)}\n"
            f"  Hip      | {fmt(self.theta_hip)}\n"
            f"  Knee     | {fmt(self.theta_knee)}"
        )
