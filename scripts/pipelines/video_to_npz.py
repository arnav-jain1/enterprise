import cv2
import mediapipe as mp
import csv
import numpy as np
from pathlib import Path
import os
from frame import Frame
from geometry import check_savgol_filter, get_shoulder_angle_array
from matplotlib import pyplot as plt
from extractions.bicep_curl import BicepCurlExtractor
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from npz_to_pandas import frames_to_numpy

def video_to_npz(video_path, output_path):
    # ---- MediaPipe PoseLandmarker (VIDEO mode) ----
    base_options = python.BaseOptions(
        model_asset_path="models/pose_landmarker_lite.task"
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    detector = vision.PoseLandmarker.create_from_options(options)

    # ---- Video ----
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    extractor = BicepCurlExtractor()
    pose_array = []
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

        if not result.pose_landmarks:
            continue

        # ---- Convert landmarks to (33, 4) ----
        landmarks = np.zeros((33, 4), dtype=np.float32)
        for i, lm in enumerate(result.pose_landmarks[0]):
            landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]

        frame = Frame(landmarks)
        prev = pose_array[frame_number - 1] if frame_number > 0 else None

        extractor.compute(frame, prev, fps)

        pose_array.append(frame)
        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ---- Save ----
    landmarks, angles, velocities, accelerations, motions, displacements = frames_to_numpy(pose_array)

    np.savez_compressed(
        f"{output_path}.npz",
        landmarks=landmarks,
        angles=angles,
        velocities=velocities,
        accelerations=accelerations,
        motions = motions,
        displacements = displacements
    )

    return pose_array