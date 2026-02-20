from geometry import joint_angle, point_displacement, segment_motion_angle

class BaseExtractor:
    '''ANGLE CALCULATIONS'''
    def calculate_elbow_angles(self, landmarks):
        return {
        "right_elbow": joint_angle(landmarks, 16, 14, 12),
        "left_elbow":  joint_angle(landmarks, 15, 13, 11),
        }
    
    def calculate_shoulder_angles(self, landmarks):
        return {
            "right_shoulder": joint_angle(landmarks, 14, 12, 24),
            "left_shoulder":  joint_angle(landmarks, 13, 11, 23),
        }

    def calculate_torso_angles(self, landmarks):
        return {
            "right_torso": joint_angle(landmarks, 12, 24, 26),
            "left_torso":  joint_angle(landmarks, 11, 23, 25),
        }
    
    def get_shoulder_motion_angle(self, prev_landmarks, curr_landmarks):
        if prev_landmarks is None:
            return {
                "right_shoulder": 0.0,
                "left_shoulder":  0.0,
            }

        return {
            "right_shoulder": segment_motion_angle(
                prev_landmarks, curr_landmarks, 14, 12
            ),
            "left_shoulder": segment_motion_angle(
                prev_landmarks, curr_landmarks, 13, 11
            ),
        }
    
    '''Velocity Calculations'''
    def calculate_velocity(self, curr, prev, fps):
        return (curr - prev) * fps
    
    def calculate_velocities(self, curr, prev, fps):

        if prev is None:
            return {key: 0.0 for key in curr}

        velocities = {}

        for key in curr:
            velocities[key] = (curr[key] - prev[key]) * fps
             
        return velocities
    
    '''Displacement Calculations'''
    def calculate_elbow_displacement(self, prev_landmarks, curr_landmarks):
        if prev_landmarks is None:
            return {
                "right_elbow": 0.0,
                "left_elbow":  0.0,
            }

        return {
            "right_elbow": point_displacement(
                prev_landmarks, curr_landmarks, 14
            ),
            "left_elbow": point_displacement(
                prev_landmarks, curr_landmarks, 13
            ),
        }
    

    '''Accelerations Calculations'''
    
    def calculate_acceleration(self, curr_vel, prev_vel, fps):
        if prev_vel is None:
            return {key: 0.0 for key in curr_vel}

        return {
            key: (curr_vel[key] - prev_vel[key]) * fps
            for key in curr_vel
        }
    
    '''Symmetry Calculations'''

    def calculate_symmetry(
        self,
        landmarks,
        right_index,
        left_index,
        coord_index=1,
        name=None
    ):

        value = abs(
            landmarks[right_index][coord_index]
            - landmarks[left_index][coord_index]
        )

        if name is not None:
            return {name: value}

        return value



