from geometry import joint_angle, point_displacement, segment_motion_angle
from extractions.base_extractor import BaseExtractor


class BicepCurlExtractor(BaseExtractor):
    def compute(self, frame, prev_frame, fps):
        # ---- 1. Angles ----
        frame.angles = self.calculate_angles(frame.landmarks)

        # ---- 2. Velocity ----
        if prev_frame is None:
            frame.velocity = {
                "right_elbow": 0.0,
                "left_elbow": 0.0,
            }
        else:
            frame.velocity = self.calculate_velocities(
                frame.angles,
                prev_frame.angles,
                fps
            )

        # ---- 3. Acceleration ----
        if prev_frame is None:
            frame.acceleration = {
                "right_elbow": 0.0,
                "left_elbow": 0.0,
            }
        else:
            frame.acceleration = {
                "right_elbow": (
                    frame.velocity["right_elbow"]
                    - prev_frame.velocity.get("right_elbow", 0.0)
                ) * fps,
                "left_elbow": (
                    frame.velocity["left_elbow"]
                    - prev_frame.velocity.get("left_elbow", 0.0)
                ) * fps,
            }

        # ---- 4. Shoulder motion (cheating) ----
        frame.motion = self.get_shoulder_motion_angle(
            prev_frame.landmarks if prev_frame else None,
            frame.landmarks
        )

        # ---- 5. Elbow displacement (stability) ----
        frame.displacement = self.calculate_displacement(
            prev_frame.landmarks if prev_frame else None,
            frame.landmarks
        )

        return frame

    '''WORKOUT SPECIFIC CALCULATIONS'''

    def calculate_wrist_angles(self, landmarks):
        return {
            "right_wrist": joint_angle(landmarks, 20, 16, 14),
            "left_wrist":  joint_angle(landmarks, 19, 15, 13),
        }

    def calculate_angles(self, landmarks):
        angles = {}
        angles.update(self.calculate_elbow_angles(landmarks))
        angles.update(self.calculate_shoulder_angles(landmarks))
        angles.update(self.calculate_torso_angles(landmarks))
        angles.update(self.calculate_wrist_angles(landmarks))
        return angles
