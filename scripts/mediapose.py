import cv2
import mediapipe as mp
import csv
import numpy as np
from pathlib import Path
import os
from frame import Frame


def frames_to_numpy(frames):
    landmarks_list = []
    angles_list = []
    velocity_list = []
    acceleration_list = []

    for frame in frames:
        landmarks_list.append(frame.landmarks)
        angles_list.append(frame.get_angles())
        velocity_list.append([frame.left_elbow_velocity, frame.right_elbow_velocity])
        acceleration_list.append([frame.left_elbow_acceleration, frame.right_elbow_acceleration])
    
    landmarks = np.stack(landmarks_list).astype(np.float32)
    angles = np.stack(angles_list).astype(np.float32)
    velocities = np.stack(velocity_list).astype(np.float32)
    accelerations = np.stack(acceleration_list).astype(np.float32)

    return landmarks, angles, velocities, accelerations
    

def compress_landmarks(video_path, pose, mp_pose, mp_drawing, output_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    pose_array = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = pose.process(frame_rgb)
        landmarks = np.zeros((33, 4), dtype=np.float32)
        if not result.pose_landmarks:
            continue
        if result.pose_landmarks:

            for idx, landmark in enumerate(result.pose_landmarks.landmark):
                landmarks[idx] = [landmark.x, landmark.y, landmark.z, landmark.visibility]

            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        frame = Frame(landmarks)
        frame.calculate_angles()

        if frame_number == 0: prev = None
        else: prev = pose_array[frame_number - 1]

        frame.calculate_angular_velocity(prev, fps)
        frame.calculate_angular_acceleration(prev, fps)

        if 0 < frame_number < 5:
            frame.sanity_check(prev, fps)

        pose_array.append(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        frame_number += 1
    cap.release()
    cv2.destroyAllWindows()

    landmarks, angles, velocities, accelerations = frames_to_numpy(pose_array)

    np.savez_compressed(f"{output_path}.npz", landmarks = landmarks, angles = angles, velocities = velocities, accelerations = accelerations)

    return pose_array

def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    directory_path = Path('../barbell')
    count = 0
    for item in directory_path.iterdir():

        video_path = str(item)
        output_path = f"../barbell_npz/barbell_{count}"

        vid = compress_landmarks(video_path, pose, mp_pose, mp_drawing, output_path)
        
        count += 1


if __name__ == "__main__":
    main()