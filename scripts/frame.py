class Frame:
    """
    Represents a single processed video frame.

    Stores:
    - raw pose landmarks
    - kinematic features (angles, motion)
    - dynamic features (velocity, acceleration)
    - stability metrics (displacement, symmetry)
    - temporal metadata (timestamp, frame index)
    - signal quality (confidence, validity)
    """

    def __init__(self, landmarks):
        # ------------------------------------------------
        # Raw Pose Data
        # ------------------------------------------------
        self.landmarks = landmarks

        # ------------------------------------------------
        # Temporal Info
        # ------------------------------------------------
        self.frame_index = None
        self.timestamp = None

        # ------------------------------------------------
        # Signal Quality
        # ------------------------------------------------
        self.valid = True
        self.confidence = None

        # ------------------------------------------------
        # Kinematic Features
        # ------------------------------------------------
        self.raw_angles = {}    # BEFORE filtering (debugging)
        self.angles = {}        # filtered joint angles
        self.motion = {}        # segment motion

        # ------------------------------------------------
        # Dynamic Features
        # ------------------------------------------------
        self.velocity = {}
        self.acceleration = {}
        self.phase = {}         # movement phase (optional)

        # ------------------------------------------------
        # Stability Metrics
        # ------------------------------------------------
        self.displacement = {}
        self.symmetry = {}

        # ------------------------------------------------
        # Derived / Custom Features
        # ------------------------------------------------
        self.features = {}

    
    def __str__(self):
        """
        Generic debug print for Frame.
        Shows all attributes without hardcoding keys.
        """

        def format_dict(d):
            if not d:
                return "{}"
            return "{ " + ", ".join(f"{k}: {round(v, 2) if isinstance(v, float) else v}" for k, v in d.items()) + " }"

        return (
            f"\nFrame {self.frame_index} | t={self.timestamp} ms\n"
            "----------------------------------------\n"

            f"Angles:        {format_dict(self.angles)}\n"
            f"Motion:        {format_dict(self.motion)}\n"
            f"Velocity:      {format_dict(self.velocity)}\n"
            f"Acceleration:  {format_dict(self.acceleration)}\n"
            f"Displacement:  {format_dict(self.displacement)}\n"
            f"Symmetry:      {format_dict(self.symmetry)}\n"
            f"Phase:         {format_dict(self.phase)}\n"
            f"Features:      {self.features}\n"
        )