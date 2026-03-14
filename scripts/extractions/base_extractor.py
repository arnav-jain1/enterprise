from abc import abstractmethod
from geometry import joint_angle, point_displacement, segment_motion_angle


class BaseExtractor:
    """
    Base class for exercise feature extraction.

    Each exercise extractor (e.g. BicepCurlExtractor, BenchPressExtractor)
    should inherit from this class and implement the mandatory methods.
    """

    # ============================================================
    # MANDATORY METHODS (must be implemented by subclasses)
    # ============================================================

    @abstractmethod
    def compute(self, frame, prev_frame, fps):
        """
        Main per-frame computation method for the extractor.
        """
        pass

    @abstractmethod
    def calculate_angles(self, landmarks):
        """
        Calculates joint angles for the current frame.
        """
        pass

    @abstractmethod
    def calculate_velocities(self, frames, fps):
        """
        Calculates velocities of features between frames.
        """
        pass

    @abstractmethod
    def calculate_accelerations(self, curr_vel, prev_vel, fps):
        """
        Calculates accelerations from velocities.
        """
        pass


    # ============================================================
    # LANDMARK POSITION FEATURES
    # ============================================================

    def get_wrist_positions(self, landmarks):
        """
        Extract vertical wrist positions.

        Useful for detecting upward/downward movement in exercises.
        """
        return {
            "right_wrist_y": landmarks[16][1],
            "left_wrist_y":  landmarks[15][1],
        }

    # Future expansion example
    # def get_elbow_positions(self, landmarks):
    #     ...


    # ============================================================
    # JOINT ANGLE CALCULATIONS
    # ============================================================

    def calculate_elbow_angles(self, landmarks):
        """
        Compute elbow joint angles.
        """
        return {
            "right_elbow": joint_angle(landmarks, 16, 14, 12),
            "left_elbow":  joint_angle(landmarks, 15, 13, 11),
        }

    def calculate_shoulder_angles(self, landmarks):
        """
        Compute shoulder joint angles.
        """
        return {
            "right_shoulder": joint_angle(landmarks, 14, 12, 24),
            "left_shoulder":  joint_angle(landmarks, 13, 11, 23),
        }

    def calculate_torso_angles(self, landmarks):
        """
        Compute torso angles relative to hips.
        """
        return {
            "right_torso": joint_angle(landmarks, 12, 24, 26),
            "left_torso":  joint_angle(landmarks, 11, 23, 25),
        }

    def calculate_wrist_angles(self, landmarks):
        """
        Compute wrist joint angles.
        """
        return {
            "right_wrist": joint_angle(landmarks, 20, 16, 14),
            "left_wrist":  joint_angle(landmarks, 19, 15, 13),
        }


    # ============================================================
    # SEGMENT MOTION FEATURES
    # ============================================================

    def get_shoulder_motion_angle(self, prev_landmarks, curr_landmarks):
        """
        Measures rotational movement of shoulder segments between frames.
        """

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


    # ============================================================
    # VELOCITY CALCULATIONS
    # ============================================================

    def calculate_velocity(self, curr, prev, fps):
        """
        Compute scalar velocity.
        """
        if prev is None:
            return {key: 0.0 for key in curr}
        return (curr - prev) * fps


    def calculate_velocities(self, curr, prev, fps):
        """
        Compute velocities for a dictionary of values.
        """

        if prev is None:
            return {key: 0.0 for key in curr}

        velocities = {}

        for key in curr:
            velocities[key] = (curr[key] - prev[key]) * fps

        return velocities


    # ============================================================
    # DISPLACEMENT FEATURES
    # ============================================================

    def calculate_elbow_displacement(self, prev_landmarks, curr_landmarks):
        """
        Euclidean displacement of elbows between frames.
        """

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


    # ============================================================
    # ACCELERATION CALCULATIONS
    # ============================================================

    def calculate_acceleration(self, curr_vel, prev_vel, fps):
        """
        Compute accelerations from velocity dictionaries.
        """

        if prev_vel is None:
            return {key: 0.0 for key in curr_vel}

        return {
            key: (curr_vel[key] - prev_vel[key]) * fps
            for key in curr_vel
        }


    # ============================================================
    # SYMMETRY FEATURES
    # ============================================================

    def calculate_symmetry(
        self,
        landmarks,
        right_index,
        left_index,
        coord_index=1,
        name=None
    ):
        """
        Measure symmetry between two body landmarks.

        Parameters
        ----------
        right_index : int
        left_index : int
        coord_index : int
            0 = x coordinate
            1 = y coordinate
        """

        value = abs(
            landmarks[right_index][coord_index]
            - landmarks[left_index][coord_index]
        )

        if name is not None:
            return {name: value}

        return value