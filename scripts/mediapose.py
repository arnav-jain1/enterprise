import cv2
import mediapipe as mp
import csv
import numpy as np
from pathlib import Path
import os
from frame import Frame



def compress_landmarks(video_path, pose, mp_pose, mp_drawing):

    cap = cv2.VideoCapture(video_path)

    frame_number = 0
    pose_array = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = pose.process(frame_rgb)
        landmarks = np.zeros((33, 4), dtype=np.float32)
        if result.pose_landmarks:

            for idx, landmark in enumerate(result.pose_landmarks.landmark):
                landmarks[idx] = [landmark.x, landmark.y, landmark.z, landmark.visibility]

            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        frame = Frame(landmarks)
        frame.get_angles()
        pose_array.append(frame)

        cv2.imshow('MediaPipe Pose', frame.landmarks)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        frame_number += 1
    cap.release()
    cv2.destroyAllWindows()

    # np.savez_compressed(f"{output_path}.npz", pose_array)

    return pose_array

def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    directory_path = Path('../barbell')
    count = 0
    for item in directory_path.iterdir():

        video_path = str(item)
        output_path = f"/barbell_npz/barbell_{count}"

        vid = compress_landmarks(video_path, pose, mp_pose, mp_drawing)
        
        for frame in vid:
            print(frame)
        count += 1


if __name__ == "__main__":
    main()