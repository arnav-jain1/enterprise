from base_extractor import BaseExtractor
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
        prev_wrist_positions = self.get_wrist_positions(frame.landmarks)

        #Angular Velocities
        frame.velocities = self.calculate_velocities(frame.angles, prev_frame.angles, fps)


        #Wrist Vertical Velocity
        frame.motion.update(self.calculate_velocities(curr_wrist_positions, prev_wrist_positions))
        
        #Symmetry
        frame.symmetry.update(self.calculate_symmetry(frame.landmarks, 16, 15, name="wrist_vertical_symmetry"))
        
        return frame

        


    '''Workout Specific'''
    def get_wrist_positions(self, landmarks):
        return {
            "right_wrist_y": landmarks[16][1],
            "left_wrist_y":  landmarks[15][1],
        }
    
    def get_elbow_positions(self, landmarks):
        return {
            
        }
    
    '''Calculate All Angles'''
    
    def calculate_angles(self, landmarks):
        angles = {}
        angles.update(self.calculate_elbow_angles(landmarks))
        angles.update(self.calculate_shoulder_angles(landmarks))
        return angles