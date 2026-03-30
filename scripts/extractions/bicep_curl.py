from scripts.extractions.base_extractor import BaseExtractor
from scripts.geometry import uniform_value

class BicepCurlExtractor(BaseExtractor):

    def calculate_angles(self, landmarks):
        angles = {}
        angles.update(self.calculate_elbow_angles(landmarks, False))
        angles.update(self.calculate_shoulder_angles(landmarks))
        angles.update(self.calculate_torso_angles(landmarks))
        angles.update(self.calculate_wrist_angles(landmarks))
        return angles
    
    def calculate_motion(self, prev_landmarks, curr_landmarks):
        motion = {}
        motion.update(self.get_shoulder_motion_angle(prev_landmarks, curr_landmarks))
        return motion
    
    def calculate_displacement(self, prev_landmarks, curr_landmarks):
        displacement = {}
        displacement.update(self.calculate_elbow_displacement(prev_landmarks, curr_landmarks))
        return displacement
    
    def calculate_additional_features(self, frame):
        """
        Compute bicep-curl-specific features.
        """

        features = {}

        # ----------------------------------------
        # Elbow dominance (main signal)
        # ----------------------------------------
        features["elbow_flexion"] = frame.angles.get("right_elbow", 0.0)

        # ----------------------------------------
        # Shoulder compensation (cheating)
        # ----------------------------------------
        features["shoulder_movement"] = abs(
            frame.motion.get("right_shoulder", 0.0)
        )

        # ----------------------------------------
        # Elbow drift (should stay fixed)
        # ----------------------------------------
        features["elbow_stability"] = frame.displacement.get("right_elbow", 0.0)

        # ----------------------------------------
        # Torso lean (cheating)
        # ----------------------------------------
        features["torso_angle"] = frame.angles.get("right_torso", 0.0)

        frame.features.update(features)

        return features


    def calculate_phase(self, frame):
        """
        Determine movement phase based on elbow velocity.
        """

        vel = frame.velocity.get("right_elbow", 0.0)

        if vel > 0:
            phase = "concentric"   # lifting
        elif vel < 0:
            phase = "eccentric"    # lowering
        else:
            phase = "static"

        frame.phase["right_elbow"] = phase
        return phase

    def evaluate_form(self, frame):
        """
        Evaluate form quality for a single frame.
        """

        issues = []

        # Shoulder cheating
        if abs(frame.motion.get("right_shoulder", 0.0)) > 15:
            issues.append("shoulder_swing")
        else:
            issues.append("shoulder_stable")

        # Elbow drifting
        if uniform_value(frame.displacement, "right_elbow", "left_elbow") > 0.005:
            issues.append("elbow_moving")
        else:
            issues.append("elbow_stable")
        # Torso leaning
        if abs(uniform_value(frame.angles, "right_torso", "left_torso")) > 20:
            issues.append("torso_lean")
        else:            
            issues.append("torso_stable")

        frame.features["form_issues"] = issues
        return issues