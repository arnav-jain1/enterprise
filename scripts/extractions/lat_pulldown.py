from scripts.extractions.base_extractor import BaseExtractor
import numpy as np


class LatPulldownExtractor(BaseExtractor):

    # ============================================================
    # ANGLES
    # ============================================================

    def calculate_angles(self, landmarks):
        angles = {}

        # Primary joints
        angles.update(self.calculate_elbow_angles(landmarks, True))
        angles.update(self.calculate_shoulder_angles(landmarks))
        angles.update(self.calculate_torso_angles(landmarks, True))

        return angles

    # ============================================================
    # DISPLACEMENT
    # ============================================================

    def calculate_displacement(self, prev, curr):
        if prev is None:
            return {
                "right_wrist": 0.0,
                "left_wrist":  0.0,
                "right_elbow": 0.0,
                "left_elbow":  0.0,
            }

        return {
            # vertical movement (y-axis) → bar path
            "right_wrist": abs(curr[16][1] - prev[16][1]),
            "left_wrist":  abs(curr[15][1] - prev[15][1]),

            # elbow movement (stability)
            "right_elbow": abs(curr[14][1] - prev[14][1]),
            "left_elbow":  abs(curr[13][1] - prev[13][1]),
        }

    # ============================================================
    # MOTION
    # ============================================================

    def calculate_motion(self, prev, curr):
        if prev is None:
            return {
                "right_wrist": 0.0,
                "left_wrist":  0.0,
            }

        return {
            # downward pull = positive/negative depending on coordinate system
            "right_wrist": curr[16][1] - prev[16][1],
            "left_wrist":  curr[15][1] - prev[15][1],
        }

    # ============================================================
    # ADDITIONAL FEATURES
    # ============================================================

    def calculate_additional_features(self, frame):
        features = {}

        # ----------------------------------------
        # Elbow angle (main ROM signal)
        # ----------------------------------------
        features["elbow_flexion"] = self.compute_uniform_value(
            frame.angles, "right_elbow", "left_elbow"
        )

        # ----------------------------------------
        # Shoulder involvement
        # ----------------------------------------
        features["shoulder_angle"] = self.compute_uniform_value(
            frame.angles, "right_shoulder", "left_shoulder"
        )

        # ----------------------------------------
        # Torso lean (cheating detection)
        # ----------------------------------------
        features["torso_angle"] = self.compute_uniform_value(
            frame.angles, "right_torso", "left_torso"
        )

        # ----------------------------------------
        # Stability (wrist drift)
        # ----------------------------------------
        features["wrist_symmetry"] = abs(
            frame.displacement.get("right_wrist", 0.0)
            - frame.displacement.get("left_wrist", 0.0)
        )

        # ----------------------------------------
        # Stability (elbow drift)
        # ----------------------------------------
        features["elbow_stability"] = self.compute_uniform_value(
            frame.displacement, "right_elbow", "left_elbow"
        )

        # ----------------------------------------
        # Symmetry (left vs right)
        # ----------------------------------------
        features["elbow_symmetry"] = abs(
            frame.angles.get("right_elbow", 0.0)
            - frame.angles.get("left_elbow", 0.0)
        )

        frame.features.update(features)
        return features

    # ============================================================
    # PHASE DETECTION
    # ============================================================

    def calculate_phase(self, frame, prev_frame=None):
        """
        Concentric → pulling down (elbow flexing)
        Eccentric  → returning up (elbow extending)
        """

        if prev_frame is None:
            frame.phase["lat_pulldown"] = "static"
            return "static"

        delta = (
            frame.angles.get("right_elbow", 0.0)
            - prev_frame.angles.get("right_elbow", 0.0)
        )

        if delta < 0:
            phase = "concentric"   # pulling down
        elif delta > 0:
            phase = "eccentric"    # going up
        else:
            phase = "static"

        frame.phase["lat_pulldown"] = phase
        return phase

    # ============================================================
    # FORM EVALUATION
    # ============================================================

    def evaluate_form(self, frame):
        issues = []

        # ----------------------------------------
        # Excessive lean back
        # ----------------------------------------
        if abs(frame.features.get("torso_angle", 0.0)) > 25:
            issues.append("excessive_lean")

        # ----------------------------------------
        # Elbow instability
        # ----------------------------------------
        if frame.features.get("elbow_stability", 0.0) > 0.03:
            issues.append("unstable_elbows")
        
        # ----------------------------------------
        # Left/right wrist imbalance
        # ----------------------------------------
        if frame.features.get("wrist_symmetry", 0.0) > .01:
            issues.append("wrist_imbalance")

        # ----------------------------------------
        # Left/right elbow imbalance
        # ----------------------------------------
        if frame.features.get("elbow_symmetry", 0.0) > 20:
            issues.append("arm_imbalance")

        frame.features["form_issues"] = issues
        return issues