from geometry import joint_angle, point_displacement, segment_motion_angle, get_all_angles_arrays
from extractions.base_extractor import BaseExtractor


class BicepCurlExtractor(BaseExtractor):
    # def compute(self, frame, prev_frame, fps):

    #     # # ---- 2. Velocity ----
    #     # if prev_frame is None:
    #     #     frame.velocity = {
    #     #         "right_elbow": 0.0,
    #     #         "left_elbow": 0.0,
    #     #     }
    #     # else:
    #     #     frame.velocity = self.calculate_velocities(
    #     #         frame.angles,
    #     #         prev_frame.angles,
    #     #         fps
    #     #     )

    #     # # ---- 3. Acceleration ----
    #     # if prev_frame is None:
    #     #     frame.acceleration = {
    #     #         "right_elbow": 0.0,
    #     #         "left_elbow": 0.0,
    #     #     }
    #     # else:
    #     #     frame.acceleration = {
    #     #         "right_elbow": (
    #     #             frame.velocity["right_elbow"]
    #     #             - prev_frame.velocity.get("right_elbow", 0.0)
    #     #         ) * fps,
    #     #         "left_elbow": (
    #     #             frame.velocity["left_elbow"]
    #     #             - prev_frame.velocity.get("left_elbow", 0.0)
    #     #         ) * fps,
    #     #     }

    #     # ---- 4. Shoulder motion (cheating) ----
    #     # frame.motion = self.get_shoulder_motion_angle(
    #     #     prev_frame.landmarks if prev_frame else None,
    #     #     frame.landmarks
    #     # )

    #     # # ---- 5. Elbow displacement (stability) ----
    #     # frame.displacement = self.calculate_elbow_displacement(
    #     #     prev_frame.landmarks if prev_frame else None,
    #     #     frame.landmarks
    #     # )

    #     return frame

    def calculate_angles(self, landmarks):
        angles = {}
        angles.update(self.calculate_elbow_angles(landmarks))
        angles.update(self.calculate_shoulder_angles(landmarks))
        angles.update(self.calculate_torso_angles(landmarks))
        angles.update(self.calculate_wrist_angles(landmarks))
        return angles
    
    def calculate_velocities(self, frames, fps):
        filtered_angles = get_all_angles_arrays(frames)
        for i in range(0, len(frames)):
            curr = i
            if i == 0:
                prev = None
            else:
                prev = i - 1

            # Velocity
            for key in filtered_angles:
                if prev is None:
                    frames[i].velocity[key] = 0.0
                else:
                    frames[i].velocity[key] = self.calculate_velocity(
                        filtered_angles[key][curr],filtered_angles[key][prev], fps)
        

    
    
