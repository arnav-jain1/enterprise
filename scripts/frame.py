from calculate_angles import get_angle

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

    def get_elbow_angles(self):
        self.theta_right_elbow = get_angle(self.landmarks, 16, 14, 12)
        self.theta_left_elbow = get_angle(self.landmarks, 15, 13, 11)

    def get_shoulder_angles(self):
        self.theta_right_shoulder = get_angle(self.landmarks, 14, 12, 24)
        self.theta_left_shoulder = get_angle(self.landmarks, 13, 11, 23)

    def get_torso_angles(self):
        self.theta_right_torso = get_angle(self.landmarks, 12, 24, 26)
        self.theta_left_torso = get_angle(self.landmarks, 11, 23, 25)

    def get_wrist_angles(self):
        self.theta_right_wrist = get_angle(self.landmarks, 20, 16, 14)
        self.theta_left_wrist = get_angle(self.landmarks, 19, 15, 13)

    def get_angles(self):
        self.get_elbow_angles()
        self.get_shoulder_angles()
        self.get_torso_angles()
        self.get_wrist_angles()
    
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
