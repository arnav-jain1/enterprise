from abc import ABC, abstractmethod
from geometry import (
    get_all_angles_arrays,
    joint_angle,
    point_displacement,
    segment_motion_angle,
)


class BaseExtractor(ABC):
    """
    Base class for exercise feature extraction.

    Provides a shared feature pipeline (angles → velocity → acceleration)
    and reusable biomechanical utilities.

    Subclasses should ONLY implement:
        - calculate_angles()
    """

    # ============================================================
    # REQUIRED (exercise-specific)
    # ============================================================

    @abstractmethod
    def calculate_angles(self, landmarks):
        """
        Compute joint angles for a single frame.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def calculate_motion(self, prev_landmarks, curr_landmarks):
        """
        Compute motion angles for a single frame.

        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def calculate_displacement(self, prev_landmarks, curr_landmarks):
        """
        Compute landmark displacement for a single frame.

        Must be implemented by subclasses.
        """
        pass
    
    # ============================================================
    # PIPELINE (shared across all extractors)
    # ============================================================

    def calculate_frame_velocities(self, frames, fps):
        """
        Compute angular velocities for all frames using
        filtered angle time-series.
        """
        filtered_angles = get_all_angles_arrays(frames)

        for i in range(len(frames)):
            prev = i - 1 if i > 0 else None

            for key in filtered_angles:
                if prev is None:
                    frames[i].velocity[key] = 0.0
                else:
                    frames[i].velocity[key] = self._scalar_velocity(
                        filtered_angles[key][i],
                        filtered_angles[key][prev],
                        fps
                    )

    def calculate_frame_accelerations(self, frames, fps):
        """
        Compute angular accelerations from velocity signals.
        """
        for i in range(len(frames)):
            prev = i - 1 if i > 0 else None

            for key in frames[i].velocity:
                if prev is None:
                    frames[i].acceleration[key] = 0.0
                else:
                    frames[i].acceleration[key] = self._scalar_acceleration(
                        frames[i].velocity[key],
                        frames[prev].velocity[key],
                        fps
                    )

    # ============================================================
    # LOW-LEVEL NUMERIC HELPERS (private)
    # ============================================================

    def _scalar_velocity(self, curr, prev, fps):
        return (curr - prev) * fps

    def _scalar_acceleration(self, curr_vel, prev_vel, fps):
        return (curr_vel - prev_vel) * fps

    # ============================================================
    # LANDMARK POSITION FEATURES
    # ============================================================

    def get_wrist_positions(self, landmarks):
        return {
            "right_wrist_y": landmarks[16][1],
            "left_wrist_y":  landmarks[15][1],
        }

    # ============================================================
    # JOINT ANGLES (reusable building blocks)
    # ============================================================

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

    # ============================================================
    # SEGMENT MOTION
    # ============================================================

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

    # ============================================================
    # DISPLACEMENT
    # ============================================================

    def calculate_elbow_displacement(self, prev_landmarks, curr_landmarks):
        if prev_landmarks is None:
            return {
                "right_elbow": 0.0,
                "left_elbow":  0.0,
            }

        return {
            "right_elbow": point_displacement(prev_landmarks, curr_landmarks, 14),
            "left_elbow":  point_displacement(prev_landmarks, curr_landmarks, 13),
        }

    # ============================================================
    # SYMMETRY
    # ============================================================

    def calculate_symmetry(
        self,
        landmarks,
        right_index,
        left_index,
        coord_index=1,
        name=None
    ):
        value = abs(
            landmarks[right_index][coord_index]
            - landmarks[left_index][coord_index]
        )

        return {name: value} if name else value