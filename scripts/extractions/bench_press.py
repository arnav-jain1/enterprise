from scripts.extractions.base_extractor import BaseExtractor

'''
Left/right elbow angle

Left/right shoulder angle

Wrist vertical position

Elbow velocity

Wrist velocity

Symmetry metric
'''


class BenchPressExtractor(BaseExtractor):
    
    def calculate_angles(self, landmarks):
        angles = {}
        angles.update(self.calculate_elbow_angles(landmarks, True))
        angles.update(self.calculate_torso_angles(landmarks))
        return angles
    
    def calculate_displacement(self, prev, curr):
        if prev is None:
            return {
                "right_elbow": 0.0,
                "left_elbow": 0.0,
                "right_wrist": 0.0,
                "left_wrist": 0.0,
            }

        displacement = {}

        displacement["right_elbow"] = abs(curr[14][1] - prev[14][1])
        displacement["left_elbow"] = abs(curr[13][1] - prev[13][1])

        return displacement
    
    def calculate_motion(self, prev, curr):

        if prev is None:
            return {
                "right_wrist": 0.0,
                "left_wrist": 0.0,
                # add all keys you expect
            }

        motion = {}

        # your normal logic
        motion["right_wrist"] = curr[16][1] - prev[16][1]
        motion["left_wrist"] = curr[15][1] - prev[15][1]

        return motion
    
    def calculate_additional_features(self, frame):
        """
        Bench-press-specific features.
        """

        features = {}

        # ----------------------------------------
        # Elbow extension (main signal)
        # ----------------------------------------
        features["elbow_extension"] = frame.angles.get("right_elbow", 0.0)

        # ----------------------------------------
        # Bar path (wrist vertical)
        # ----------------------------------------
        features["bar_path"] = frame.motion.get("right_wrist", 0.0)

        # ----------------------------------------
        # Shoulder compensation
        # ----------------------------------------
        features["shoulder_movement"] = abs(
            frame.motion.get("right_shoulder", 0.0)
        )

        # ----------------------------------------
        # Elbow stability
        # ----------------------------------------
        features["elbow_stability"] = frame.displacement.get("right_elbow", 0.0)

        # ----------------------------------------
        # Torso stability (arching / lifting)
        # ----------------------------------------
        features["torso_angle"] = frame.angles.get("right_torso", 0.0)

        # ----------------------------------------
        # Symmetry (left vs right)
        # ----------------------------------------
        features["symmetry"] = abs(
            frame.angles.get("right_elbow", 0.0)
            - frame.angles.get("left_elbow", 0.0)
        )

        frame.features.update(features)
        
        return features

    def calculate_phase(self, frame):
        """
        Determine phase using wrist velocity (bar movement).
        """

        vel = frame.motion.get("right_wrist", 0.0)

        if vel < 0:
            phase = "eccentric"   # lowering (down)
        elif vel > 0:
            phase = "concentric"  # pressing (up)
        else:
            phase = "static"

        frame.phase["right_wrist"] = phase
        return phase

    def evaluate_form(self, frame):
        """
        Evaluate form for a single frame.
        """

        issues = []

        # ----------------------------------------
        # Symmetry issue
        # ----------------------------------------
        if self.calculate_symmetry(frame.landmarks, 16, 15) > .05:
            issues.append("asymmetry")
            issues.append("instability")
        else:
            issues.append("symmetric")

        # ----------------------------------------
        # Torso instability
        # ----------------------------------------
        if self.compute_uniform_angle(frame.angles, "right_torso", "left_torso") < 100:
            issues.append("back not arched")
        else:
            issues.append("back_arched")

        frame.features["form_issues"] = issues
        return issues