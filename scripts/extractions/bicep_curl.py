from geometry import joint_angle, point_displacement, segment_motion_angle, get_all_angles_arrays
from extractions.base_extractor import BaseExtractor


class BicepCurlExtractor(BaseExtractor):

    def calculate_angles(self, landmarks):
        angles = {}
        angles.update(self.calculate_elbow_angles(landmarks))
        angles.update(self.calculate_shoulder_angles(landmarks))
        angles.update(self.calculate_torso_angles(landmarks))
        angles.update(self.calculate_wrist_angles(landmarks))
        return angles
    
    def calculate_motion(self, prev_landmarks, curr_landmarks):
        motion = {}
        motion.update(self.get_shoulder_motion_angle(prev_landmarks, curr_landmarks))
        return motion
    
    def calculate_displacement(self, prev_landmarks, curr_landmarks):
        displacement = {}
        displacement.update(self.calculate_elbow_displacement(prev_landmarks, curr_landmarks))
        return displacement
