from geometry import joint_angle, point_displacement, segment_motion_angle


class BicepCurlExtractor:
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
            frame.velocity = self.calculate_angular_velocity(
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
        frame.displacement = self.calculate_elbow_displacement(
            prev_frame.landmarks if prev_frame else None,
            frame.landmarks
        )

    def calculate_elbow_angles(self, landmarks):
        return {
        "right_elbow": joint_angle(landmarks, 16, 14, 12),
        "left_elbow":  joint_angle(landmarks, 15, 13, 11),
        }


    def calculate_shoulder_angles(self, landmarks):
        return {
            "right_shoulder": joint_angle(landmarks, 14, 12, 24),
            "left_shoulder":  joint_angle(landmarks, 13, 11, 23),
        }

    def calculate_torso_angles(self, landmarks):
        return {
            "right_torso": joint_angle(landmarks, 12, 24, 26),
            "left_torso":  joint_angle(landmarks, 11, 23, 25),
        }

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


    def calculate_angular_velocity(self, curr_angles, prev_angles, fps):
        if prev_angles is None:
            return {
                "right_elbow": 0.0,
                "left_elbow":  0.0,
            }

        return {
            "right_elbow": (curr_angles["right_elbow"] - prev_angles["right_elbow"]) * fps,
            "left_elbow":  (curr_angles["left_elbow"]  - prev_angles["left_elbow"])  * fps,
        }

    def calculate_angular_velocity(self, curr_angles, prev_angles, fps):
        if prev_angles is None:
            return {
                "right_elbow": 0.0,
                "left_elbow":  0.0,
            }

        return {
            "right_elbow": (curr_angles["right_elbow"] - prev_angles["right_elbow"]) * fps,
            "left_elbow":  (curr_angles["left_elbow"]  - prev_angles["left_elbow"])  * fps,
        }

    def get_shoulder_motion_angle(self, prev_landmarks, curr_landmarks):
        if prev_landmarks is None:
            return {
                "right_shoulder": 0.0,
                "left_shoulder":  0.0,
            }

        return {
            "right_shoulder": segment_motion_angle(
                prev_landmarks, curr_landmarks, 14, 12
            ),
            "left_shoulder": segment_motion_angle(
                prev_landmarks, curr_landmarks, 13, 11
            ),
        }
    
  
    def calculate_elbow_displacement(self, prev_landmarks, curr_landmarks):
        if prev_landmarks is None:
            return {
                "right_elbow": 0.0,
                "left_elbow":  0.0,
            }

        return {
            "right_elbow": point_displacement(
                prev_landmarks, curr_landmarks, 14
            ),
            "left_elbow": point_displacement(
                prev_landmarks, curr_landmarks, 13
            ),
        }
