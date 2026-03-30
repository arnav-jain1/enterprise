from abc import ABC, abstractmethod
from scripts.geometry import (
    get_all_angles_arrays,
    joint_angle,
    point_displacement,
    segment_motion_angle,
    segment_orientation_horizontal,
    segment_orientation_vertical
)
import numpy as np
from scripts.frame import Frame


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

    def calculate_elbow_angles(self, landmarks, abs_bool):
        if abs_bool:
            return {
            "right_elbow": abs(joint_angle(landmarks, 16, 14, 12)),
            "left_elbow":  abs(joint_angle(landmarks, 15, 13, 11)),
        }

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
            "right_torso": segment_orientation_vertical(landmarks, 12, 24),
            "left_torso":  segment_orientation_vertical(landmarks, 11, 23),
        }

    def calculate_wrist_angles(self, landmarks):
        return {
            "right_wrist": joint_angle(landmarks, 20, 16, 14),
            "left_wrist":  joint_angle(landmarks, 19, 15, 13),
        }
    
    def calculate_hip_angles(self, landmarks):
        return {
            "right_hip": joint_angle(landmarks, 12, 24, 26),  # shoulder-hip-knee
            "left_hip":  joint_angle(landmarks, 11, 23, 25),
        }
    
    def calculate_knee_angles(self, landmarks):
        return {
            "right_knee": joint_angle(landmarks, 24, 26, 28),
            "left_knee":  joint_angle(landmarks, 23, 25, 27),
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

    def calculate_hip_displacement(self, prev_landmarks, curr_landmarks):
        if prev_landmarks is None:
            return {
                "right_hip": 0.0,
                "left_hip":  0.0,
            }

        return {
            "right_hip": point_displacement(prev_landmarks, curr_landmarks, 24),
            "left_hip":  point_displacement(prev_landmarks, curr_landmarks, 23),
        }
    
    def calculate_shoulder_displacement(self, prev_landmarks, curr_landmarks):
        if prev_landmarks is None:
            return {
                "right_shoulder": 0.0,
                "left_shoulder":  0.0,
            }

        return {
            "right_shoulder": point_displacement(prev_landmarks, curr_landmarks, 12),
            "left_shoulder":  point_displacement(prev_landmarks, curr_landmarks, 11),
        }
    
    def calculate_wrist_displacement(self, prev_landmarks, curr_landmarks):
        if prev_landmarks is None:
            return {
                "right_wrist": 0.0,
                "left_wrist":  0.0,
            }

        return {
            "right_wrist": point_displacement(prev_landmarks, curr_landmarks, 16),
            "left_wrist":  point_displacement(prev_landmarks, curr_landmarks, 15),
        }
    
    def calculate_knee_displacement(self, prev_landmarks, curr_landmarks):
        if prev_landmarks is None:
            return {
                "right_knee": 0.0,
                "left_knee":  0.0,
            }

        return {
            "right_knee": point_displacement(prev_landmarks, curr_landmarks, 26),
            "left_knee":  point_displacement(prev_landmarks, curr_landmarks, 25),
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
    
    def check_feet_shoulder_alignment(self, landmarks, threshold=0.08):
        """
        Checks if shoulders are aligned over feet (deadlift setup).

        Parameters
        ----------
        landmarks : list
            MediaPipe pose landmarks
        threshold : float
            Allowed horizontal deviation (normalized coords)

        Returns
        -------
        dict
            {
                "aligned": bool,
                "shoulder_mid_x": float,
                "foot_mid_x": float,
                "deviation": float
            }
        """

        # ---------------------------------------
        # Get X positions (normalized [0,1])
        # ---------------------------------------
        left_shoulder_x = landmarks[11][0]
        right_shoulder_x = landmarks[12][0]
        left_foot_x = landmarks[31][0]
        right_foot_x = landmarks[32][0]

        # ---------------------------------------
        # Compute midpoints
        # ---------------------------------------
        shoulder_mid_x = (left_shoulder_x + right_shoulder_x) / 2
        foot_mid_x = (left_foot_x + right_foot_x) / 2

        # ---------------------------------------
        # Compute deviation
        # ---------------------------------------
        deviation = abs(shoulder_mid_x - foot_mid_x)

        aligned = deviation < threshold

        return {
            "aligned": aligned,
            "shoulder_mid_x": shoulder_mid_x,
            "foot_mid_x": foot_mid_x,
            "deviation": deviation
        }
    
    # ============================================================
    # TEMPORAL / SEQUENCE FEATURES
    # ============================================================

    def detect_reps(self, angle_series):
        """
        Detect repetitions based on local minima in the angle signal.

        For bicep curls:
            - A rep peak occurs at the smallest elbow angle (top of curl)

        Parameters
        ----------
        angle_series : list[float] or np.array

        Returns
        -------
        list[int]
            Indices of detected rep peaks
        """

        reps = []

        for i in range(1, len(angle_series) - 1):
            if angle_series[i - 1] > angle_series[i] < angle_series[i + 1]:
                reps.append(i)

        return reps


    def get_movement_phase(self, velocity):
        """
        Classify movement phase based on velocity sign.

        Parameters
        ----------
        velocity : float

        Returns
        -------
        str
            "concentric"  -> lifting phase
            "eccentric"   -> lowering phase
            "static"      -> near zero movement
        """

        if velocity > 0:
            return "concentric"
        elif velocity < 0:
            return "eccentric"
        return "static"


    # ============================================================
    # RANGE OF MOTION (ROM)
    # ============================================================

    def compute_rom(self, angle_series):
        """
        Compute range of motion for a joint.

        Parameters
        ----------
        angle_series : list[float] or np.array

        Returns
        -------
        float
            Max angle - Min angle
        """

        if len(angle_series) == 0:
            return 0.0

        return max(angle_series) - min(angle_series)


    # ============================================================
    # STABILITY / CONTROL METRICS
    # ============================================================
    def compute_uniform_angle(self, angles, right_index, left_index):
        return (angles[right_index] + angles[left_index]) / 2

    def compute_stability(self, displacement_series):
        """
        Measure movement stability using standard deviation.

        High variance = unstable / excessive movement
        Low variance  = controlled / stable movement

        Parameters
        ----------
        displacement_series : list[float] or np.array

        Returns
        -------
        float
        """

        if len(displacement_series) == 0:
            return 0.0

        return np.std(displacement_series)


    def compute_smoothness(self, acceleration_series):
        """
        Estimate smoothness using jerk (change in acceleration).

        Lower jerk = smoother motion
        Higher jerk = jerky / uncontrolled motion

        Parameters
        ----------
        acceleration_series : list[float] or np.array

        Returns
        -------
        float
            Mean absolute jerk
        """

        if len(acceleration_series) < 2:
            return 0.0

        jerk = np.diff(acceleration_series)
        return np.mean(np.abs(jerk))


    # ============================================================
    # FEATURE AGGREGATION (FRAME → WORKOUT)
    # ============================================================

    def aggregate_features(self, frames):
        """
        Aggregate frame-level features into workout-level metrics.

        This converts raw signals into meaningful summaries.

        Parameters
        ----------
        frames : list[Frame]

        Returns
        -------
        dict
            High-level workout metrics
        """

        if len(frames) == 0:
            return {}

        # ----------------------------------------
        # Extract time-series
        # ----------------------------------------
        elbow_angles = [f.angles.get("right_elbow", 0.0) for f in frames]
        velocities = [f.velocity.get("right_elbow", 0.0) for f in frames]
        accelerations = [f.acceleration.get("right_elbow", 0.0) for f in frames]
        displacements = [f.displacement.get("right_elbow", 0.0) for f in frames]

        # ----------------------------------------
        # Compute metrics
        # ----------------------------------------
        reps = self.detect_reps(elbow_angles)

        metrics = {
            "rep_count": len(reps),
            "range_of_motion": self.compute_rom(elbow_angles),
            "avg_velocity": float(np.mean(velocities)) if velocities else 0.0,
            "stability": self.compute_stability(displacements),
            "smoothness": self.compute_smoothness(accelerations),
        }

        return metrics