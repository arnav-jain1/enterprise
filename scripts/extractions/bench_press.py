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

    def compute(self, frame: Frame, prev_frame: Frame, fps):
        #Angles
        frame.angles = self.calculate_angles(frame.landmarks)
        
        #Wrist Vertical Position
        curr_wrist_positions = self.get_wrist_positions(frame.landmarks)
        if prev_frame is not None:
            prev_wrist_positions = self.get_wrist_positions(prev_frame.landmarks)
        else:
            prev_wrist_positions = None

        # #Angular Velocities
        # if prev_frame is None:
        #     frame.velocities = {k: 0.0 for k in frame.angles}
        # else:
        #     frame.velocities = self.calculate_velocities(
        #         frame.angles,
        #         prev_frame.angles,
        #         fps
        #     )


        # #Wrist Vertical Velocity
        # frame.motion.update(self.calculate_velocities(curr_wrist_positions, prev_wrist_positions, fps))
        
        # #Symmetry
        # frame.symmetry.update(self.calculate_symmetry(frame.landmarks, 16, 15, name="wrist_vertical_symmetry"))
        
        return frame
    
    
    def calculate_angles(self, landmarks):
        angles = {}
        angles.update(self.calculate_elbow_angles(landmarks))
        angles.update(self.calculate_shoulder_angles(landmarks))
        return angles
    
    