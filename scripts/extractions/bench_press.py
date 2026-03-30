from scripts.extractions.base_extractor import BaseExtractor
import numpy as np

'''
Overhead view bench press extractor.

Visible landmarks overhead:
    - Shoulders (11, 12)
    - Elbows (13, 14)
    - Wrists (15, 16)

What we can reliably measure:
    - Elbow angle (ROM, phase detection)
    - Elbow flare (elbows tracking out relative to wrists/shoulders)
    - Grip width (wrist spacing relative to shoulder width)
    - Lateral bar path drift (wrists drifting left/right)
    - Left/right symmetry

What we cannot measure overhead:
    - Bar path vertical (depth/ROM in the press direction)
    - Torso/arch
    - Hip drive
'''


class BenchPressExtractor(BaseExtractor):

    # ============================================================
    # ANGLES
    # ============================================================

    def calculate_angles(self, landmarks):
        angles = {}
        angles.update(self.calculate_elbow_angles(landmarks, abs_bool=True))
        angles.update(self.calculate_shoulder_angles(landmarks))
        angles.update(self.calculate_elbow_flare(landmarks))
        return angles

    # ============================================================
    # DISPLACEMENT
    # ============================================================

    def calculate_displacement(self, prev, curr):
        if prev is None:
            return {
                "right_elbow": 0.0,
                "left_elbow":  0.0,
                "right_wrist": 0.0,
                "left_wrist":  0.0,
            }

        return {
            "right_elbow": abs(curr[14][0] - prev[14][0]),  # lateral drift
            "left_elbow":  abs(curr[13][0] - prev[13][0]),
            "right_wrist": abs(curr[16][0] - prev[16][0]),
            "left_wrist":  abs(curr[15][0] - prev[15][0]),
        }

    # ============================================================
    # MOTION
    # ============================================================

    def calculate_motion(self, prev, curr):
        """
        Overhead: wrist lateral motion (x-axis drift).
        Phase is derived from elbow angle change, not wrist vertical.
        """
        if prev is None:
            return {
                "right_wrist": 0.0,
                "left_wrist":  0.0,
            }

        return {
            "right_wrist": curr[16][0] - prev[16][0],  # x drift
            "left_wrist":  curr[15][0] - prev[15][0],
        }

    # ============================================================
    # ADDITIONAL FEATURES
    # ============================================================

    def calculate_additional_features(self, frame):
        features = {}

        # Elbow ROM (primary rep signal overhead)
        features["elbow_extension"] = frame.angles.get("right_elbow", 0.0)

        # Elbow flare
        features["right_elbow_flare"] = frame.angles.get("right_elbow_flare", 0.0)
        features["left_elbow_flare"]  = frame.angles.get("left_elbow_flare", 0.0)
        features["avg_elbow_flare"]   = self.compute_uniform_value(
            frame.angles, "right_elbow_flare", "left_elbow_flare"
        )

        # Grip width ratio (needs raw landmarks)
        if frame.landmarks is not None:
            features["grip_width_ratio"] = self.calculate_grip_width_ratio(
                frame.landmarks
            )

        # Lateral wrist drift (bar path consistency)
        features["right_wrist_drift"] = frame.displacement.get("right_wrist", 0.0)
        features["left_wrist_drift"]  = frame.displacement.get("left_wrist", 0.0)

        # Elbow lateral stability
        features["elbow_stability"] = self.compute_uniform_value(
            frame.displacement, "right_elbow", "left_elbow"
        )

        # Left/right symmetry (elbow angles)
        features["symmetry"] = abs(
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
        Overhead: phase from elbow angle delta.

        Concentric (pressing up)  → elbow angle increasing (extending)
        Eccentric  (lowering)     → elbow angle decreasing (flexing)
        """
        if prev_frame is None:
            frame.phase["bench"] = "static"
            return "static"

        delta = (
            frame.angles.get("right_elbow", 0.0)
            - prev_frame.angles.get("right_elbow", 0.0)
        )

        if delta > 0:
            phase = "concentric"
        elif delta < 0:
            phase = "eccentric"
        else:
            phase = "static"

        frame.phase["bench"] = phase
        return phase

    # ============================================================
    # FORM EVALUATION
    # ============================================================

    def evaluate_form(self, frame):
        issues = []

        # ----------------------------------------
        # Symmetry — left vs right elbow angle
        # ----------------------------------------
        if self.calculate_symmetry(frame.landmarks, 16, 15) > .07:
            issues.append("asymmetry")

        # ----------------------------------------
        # Grip width — outside 1.2–2.0x shoulder width is a flag
        # ----------------------------------------
        grip_ratio = frame.features.get("grip_width_ratio", 0.0)
        if grip_ratio < 1.2:
            issues.append("grip_too_narrow")
        elif grip_ratio > 3:
            issues.append("grip_too_wide")

        # ----------------------------------------
        # Lateral wrist drift — bar path instability
        # ----------------------------------------
        right_drift = frame.features.get("right_wrist_drift", 0.0)
        left_drift  = frame.features.get("left_wrist_drift", 0.0)
        if right_drift > 0.03 or left_drift > 0.03:
            issues.append("bar_path_drift")

        print(self.calculate_symmetry(frame.landmarks, 16, 15))
        frame.features["form_issues"] = issues
        return issues