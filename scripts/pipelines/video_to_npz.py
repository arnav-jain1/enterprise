from unittest import result

import cv2
import mediapipe as mp
import csv
import numpy as np
from pathlib import Path
import os
from frame import Frame
from matplotlib import pyplot as plt
from extractions.bicep_curl import BicepCurlExtractor
from extractions.bench_press import BenchPressExtractor
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from npz_to_pandas import frames_to_numpy
from geometry import joint_angle, point_displacement, segment_motion_angle, get_all_angles_arrays


# -------------------------------------------------------------
# Create MediaPipe PoseLandmarker (VIDEO mode)
# -------------------------------------------------------------
def create_pose_detector():
    """
    Initializes MediaPipe PoseLandmarker for video processing.
    """

    base_options = python.BaseOptions(
        model_asset_path="models/pose_landmarker_lite.task"
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    detector = vision.PoseLandmarker.create_from_options(options)

    return detector


# -------------------------------------------------------------
# Select extractor depending on exercise name in file path
# -------------------------------------------------------------
def select_extractor(video_path):
    """
    Chooses which extractor to use based on the video filename.
    """

    extractor_types = {
        "barbell biceps curl": BicepCurlExtractor(),
        "bench press": BenchPressExtractor()
    }

    for exercise in extractor_types:
        if exercise in video_path:
            print("Using extractor:", exercise)
            return extractor_types[exercise]
        else:
            print("Not:", exercise)

    raise ValueError("No matching extractor found for video.")


# -------------------------------------------------------------
# Process video and extract frames with raw angles
# -------------------------------------------------------------

def process_video(video_path, detector, extractor):
    """
    Runs pose detection on the video and extracts
    landmarks and joint angles per frame.
    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    frame_number = 0

    while cap.isOpened():

        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        result = detector.detect_for_video(mp_image, timestamp_ms)

        # -------------------------------------------------
        # Robust landmark handling (FIXED)
        # -------------------------------------------------
        if result.pose_landmarks and len(result.pose_landmarks) > 0:

            curr_landmarks = np.zeros((33, 4), dtype=np.float32)

            for i, lm in enumerate(result.pose_landmarks[0]):
                curr_landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]

            RIGHT_ELBOW = 14
            RIGHT_WRIST = 16

            is_low_vis = (
                curr_landmarks[RIGHT_ELBOW][3] < 0.2 or
                curr_landmarks[RIGHT_WRIST][3] < 0.2
            )

            if not frames:
                # always accept first valid frame
                landmarks = curr_landmarks

            else:
                prev = frames[-1].landmarks

                # smooth interpolation (no freezing)
                landmarks = 0.85 * prev + 0.15 * curr_landmarks

        else:
            if frames:
                # fallback ONLY if we already have frames
                landmarks = frames[-1].landmarks.copy()
            else:
                continue

        # -------------------------------------------------
        # Create Frame object
        # -------------------------------------------------
        frame = Frame(landmarks)

        prev_landmarks = frames[-1].landmarks if frames else None

        # -------------------------------------------------
        # Angles
        # -------------------------------------------------
        frame.angles = extractor.calculate_angles(landmarks)

        # -------------------------------------------------
        # Soft angle smoothing (no freezing)
        # -------------------------------------------------
        MAX_ANGLE_DELTA = 90

        if frames:
            prev_angles = frames[-1].angles

            for key in frame.angles:
                delta = abs(frame.angles[key] - prev_angles[key])

                if delta > MAX_ANGLE_DELTA:
                    frame.angles[key] = (
                        0.7 * prev_angles[key] + 0.3 * frame.angles[key]
                    )

                # clamp to valid range
                frame.angles[key] = max(0, min(180, frame.angles[key]))

        # -------------------------------------------------
        # Motion + displacement
        # -------------------------------------------------
        frame.motion = extractor.calculate_motion(prev_landmarks, landmarks)
        frame.displacement = extractor.calculate_displacement(prev_landmarks, landmarks)

        frames.append(frame)
        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frames, fps


# -------------------------------------------------------------
# Calculate Frame Motion Metrics
# -------------------------------------------------------------
def compute_motion_metrics(frames, extractor, fps):
    extractor.calculate_frame_velocities(frames, fps)
    extractor.calculate_frame_accelerations(frames, fps)

    return frames


# -------------------------------------------------------------
# Save frames to compressed NPZ
# -------------------------------------------------------------
def save_npz(frames, output_path):
    """
    Converts Frame objects to numpy arrays and saves them.
    """

    landmarks, angles, velocities, accelerations, motions, displacements = frames_to_numpy(frames)

    np.savez_compressed(
        f"{output_path}.npz",
        landmarks=landmarks,
        angles=angles,
        velocities=velocities,
        accelerations=accelerations,
        motions=motions,
        displacements=displacements
    )


# -------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------
def video_to_npz(video_path, output_path):
    """
    Full processing pipeline:

    video → pose detection → angle extraction → NPZ file
    """

    # 1 Create pose detector
    detector = create_pose_detector()

    # 2 Select correct extractor
    extractor = select_extractor(video_path)

    # 3 Process video frames
    frames, fps = process_video(video_path, detector, extractor)

    # 4 Compute motion metrics
    frames = compute_motion_metrics(frames, extractor, fps)

    # 5 Save processed data
    save_npz(frames, output_path)

    return frames