from extractions.base_extractor import BaseExtractor
from frame import Frame

'''
Left/right elbow angle

Left/right shoulder angle

Wrist vertical position

Elbow velocity

Wrist velocity

Symmetry metric
'''


class BenchPressExtractor(BaseExtractor):
    
    def calculate_angles(self, landmarks):
        angles = {}
        angles.update(self.calculate_elbow_angles(landmarks))
        angles.update(self.calculate_shoulder_angles(landmarks))
        return angles

    def calculate_displacement(self, prev_landmarks, curr_landmarks):
        return super().calculate_displacement(prev_landmarks, curr_landmarks)

    def calculate_motion(self, prev_landmarks, curr_landmarks):
        motion = {}
        motion.update(self.get_shoulder_motion_angle(prev_landmarks, curr_landmarks))
    