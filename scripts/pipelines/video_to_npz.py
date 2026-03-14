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

    while cap.isOpened():

        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Timestamp required for VIDEO mode
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Run pose detection
        result = detector.detect_for_video(mp_image, timestamp_ms)

        # Skip frame if no pose detected
        if not result.pose_landmarks:
            continue

        # -------------------------------------------------
        # Convert landmarks → numpy array (33,4)
        # -------------------------------------------------
        landmarks = np.zeros((33, 4), dtype=np.float32)

        for i, lm in enumerate(result.pose_landmarks[0]):
            landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]

        # -------------------------------------------------
        # Create Frame object
        # -------------------------------------------------
        frame = Frame(landmarks)

        # Calculate raw angles
        frame.angles = extractor.calculate_angles(landmarks)

        frames.append(frame)

        # Optional quit key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frames, fps


# -------------------------------------------------------------
# Calculate Frame Motion Metrics
# -------------------------------------------------------------
def compute_motion_metrics(frames, extractor, fps):
    extractor.calculate_velocities(frames, fps)
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