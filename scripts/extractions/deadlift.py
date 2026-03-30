from scripts.extractions.base_extractor import BaseExtractor
from scripts.geometry import uniform_angle

class DeadliftExtractor(BaseExtractor):

    # ============================================================
    # ANGLES
    # ============================================================
    def calculate_angles(self, landmarks):
        angles = {}

        # Core joints for deadlift
        angles.update(self.calculate_torso_angles(landmarks))   # back angle
        angles.update(self.calculate_hip_angles(landmarks))     # YOU NEED TO ADD THIS (see below)
        angles.update(self.calculate_knee_angles(landmarks))    # YOU NEED TO ADD THIS

        return angles

    # ============================================================
    # MOTION (directional)
    # ============================================================
    def calculate_motion(self, prev, curr):

        if prev is None:
            return {
                "hip": 0.0,
                "shoulder": 0.0,
                "bar_vertical": 0.0,
                "bar_horizontal": 0.0,
                "right_wrist": 0.0,
                "left_wrist": 0.0
            }

        motion = {}

        # Hip vertical movement (main driver)
        motion["hip"] = curr[24][1] - prev[24][1]

        # Shoulder vertical movement
        motion["shoulder"] = curr[12][1] - prev[12][1]

        # Bar proxy (wrist)
        motion["bar_vertical"] = curr[16][1] - prev[16][1]
        motion["bar_horizontal"] = curr[16][0] - prev[16][0]

        motion["right_wrist"] = curr[16][1] - prev[16][1]
        motion["left_wrist"] = curr[15][1] - prev[15][1]

        return motion

    # ============================================================
    # DISPLACEMENT (stability)
    # ============================================================
    def calculate_displacement(self, prev, curr):

        displacement = {}

        displacement.update(self.calculate_hip_displacement(prev, curr))
        displacement.update(self.calculate_shoulder_displacement(prev, curr))
        displacement.update(self.calculate_wrist_displacement(prev, curr))

        return displacement

    # ============================================================
    # FEATURES (deadlift-specific)
    # ============================================================
    def calculate_additional_features(self, frame):

        features = {}

        # ----------------------------------------
        # Hip hinge (main signal)
        # ----------------------------------------
        features["hip_hinge"] = uniform_angle(frame.angles, "right_hip", "left_hip")

        # ----------------------------------------
        # Back angle (critical)
        # ----------------------------------------
        features["back_angle"] = uniform_angle(frame.angles, "right_torso", "left_torso")

        # ----------------------------------------
        # Bar path
        # ----------------------------------------
        features["bar_vertical"] = frame.motion.get("bar_vertical", 0.0)
        features["bar_horizontal"] = frame.motion.get("bar_horizontal", 0.0)

        # ----------------------------------------
        # Knee extension
        # ----------------------------------------
        features["knee_extension"] = uniform_angle(frame.angles, "left_knee", "right_knee")

        # ----------------------------------------
        # Alignment (YOU ALREADY ADDED THIS 🔥)
        # ----------------------------------------
        alignment = self.check_feet_shoulder_alignment(frame.landmarks)
        features["alignment_error"] = alignment["deviation"]

        # ----------------------------------------
        # Symmetry
        # ----------------------------------------
        features["symmetry"] = abs(
            frame.angles.get("right_hip", 0.0)
            - frame.angles.get("left_hip", 0.0)
        )

        frame.features.update(features)
        return features

    # ============================================================
    # PHASE
    # ============================================================
    def calculate_phase(self, frame):

        vel = frame.motion.get("hip", 0.0)
        phase = self.get_movement_phase(vel)

        frame.phase["hip"] = phase
        return phase

    # ============================================================
    # FORM EVALUATION
    # ============================================================
    def evaluate_form(self, frame):

        issues = []

        # ----------------------------------------
        # Back rounding
        # ----------------------------------------
        if abs(frame.features.get("back_angle", 0.0)) < 120:
            issues.append("back_rounding")

        # ----------------------------------------
        # Bar drifting forward/back
        # ----------------------------------------
        if abs(frame.features.get("bar_horizontal", 0.0)) > 0.02:
            issues.append("bar_drift")

        # # ----------------------------------------
        # # Lockout
        # # ----------------------------------------
        # if frame.features.get("knee_extension", 180) < 150:
        #     issues.append("incomplete_lockout")

        # ----------------------------------------
        # Setup alignment
        # ----------------------------------------
        if frame.features.get("alignment_error", 0.0) > 0.08:
            issues.append("bad_setup")


        frame.features["form_issues"] = issues

        return issues