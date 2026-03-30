import cv2
from pathlib import Path
import mediapipe as mp
import numpy as np
from scripts.frame import Frame
from scripts.extractions.bicep_curl import BicepCurlExtractor
from scripts.extractions.bench_press import BenchPressExtractor
from scripts.extractions.deadlift import DeadliftExtractor
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scripts.pipelines.npz_to_pandas import frames_to_numpy


# -------------------------------------------------------------
# Create MediaPipe PoseLandmarker (VIDEO mode)
# -------------------------------------------------------------
def create_pose_detector():
    base_dir = Path(__file__).resolve().parents[1]  
    # video_to_npz.py is in scripts/pipelines/
    # parents[1] → scripts/

    model_path = base_dir / "models" / "pose_landmarker_lite.task"

    base_options = python.BaseOptions(
        model_asset_path=str(model_path)
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
        "bench press": BenchPressExtractor(),
        "deadlift": DeadliftExtractor()
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
            LEFT_ELBOW = 13
            LEFT_WRIST = 15

            is_low_vis = (
                curr_landmarks[RIGHT_ELBOW][3] < 0.4 or
                curr_landmarks[RIGHT_WRIST][3] < 0.4 or
                curr_landmarks[LEFT_ELBOW][3] < 0.4 or
                curr_landmarks[LEFT_WRIST][3] < 0.4
            )

            if not frames:
                landmarks = curr_landmarks.copy()
            else:
                prev = frames[-1].landmarks
                landmarks = curr_landmarks.copy()

                # -----------------------------
                # RIGHT arm fallback
                # -----------------------------
                if curr_landmarks[RIGHT_ELBOW][3] < 0.2 or curr_landmarks[RIGHT_WRIST][3] < 0.2:
                    landmarks[14] = prev[14]  # elbow
                    landmarks[16] = prev[16]  # wrist

                # -----------------------------
                # LEFT arm fallback (THIS FIXES YOUR BUG)
                # -----------------------------
                if curr_landmarks[LEFT_ELBOW][3] < 0.2 or curr_landmarks[LEFT_WRIST][3] < 0.2:
                    landmarks[13] = prev[13]  # elbow
                    landmarks[15] = prev[15]  # wrist

            if not frames:
                landmarks = curr_landmarks
            else:
                prev = frames[-1].landmarks

                if is_low_vis:
                    # trust previous more
                    landmarks = 0.95 * prev + 0.05 * curr_landmarks
                else:
                    # normal smoothing
                    landmarks = 0.85 * prev + 0.15 * curr_landmarks

        # -------------------------------------------------
        # Create Frame
        # -------------------------------------------------
        frame = Frame(landmarks)

        prev_landmarks = frames[-1].landmarks if frames else None

        # -------------------------------------------------
        # Angles
        # -------------------------------------------------
        frame.angles = extractor.calculate_angles(landmarks)

        # -------------------------------------------------
        # Soft angle smoothing
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

                # Only clamp joint angles
                if any(j in key for j in ["elbow", "knee", "hip", "shoulder"]):
                    frame.angles[key] = max(0, min(180, frame.angles[key]))

        # -------------------------------------------------
        # Motion + displacement
        # -------------------------------------------------
        frame.motion = extractor.calculate_motion(prev_landmarks, landmarks)
        frame.displacement = extractor.calculate_displacement(prev_landmarks, landmarks)

        # -------------------------------------------------
        # Metadata (SET BEFORE APPEND)
        # -------------------------------------------------
        frame.frame_index = frame_number
        frame.timestamp = timestamp_ms

        # -------------------------------------------------
        # Append ONCE
        # -------------------------------------------------
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
    
    for frame in frames:
        extractor.calculate_additional_features(frame)
        extractor.calculate_phase(frame)
        print(extractor.evaluate_form(frame))
        # extractor.evaluate_form(frame)

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