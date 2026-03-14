class Frame:
    """
    Represents a single processed video frame.

    Each frame stores:
    - raw pose landmarks
    - joint angles
    - motion metrics
    - velocities
    - accelerations
    - displacements
    - symmetry measurements
    """

    def __init__(self, landmarks):
        """
        Parameters
        ----------
        landmarks : ndarray (33 x 4)
            MediaPipe pose landmarks for the frame.
        """

        # ------------------------------------------------
        # Raw Pose Data
        # ------------------------------------------------
        self.landmarks = landmarks

        # ------------------------------------------------
        # Kinematic Features
        # ------------------------------------------------
        self.angles = {}        # joint angles
        self.motion = {}        # segment motion angles

        # ------------------------------------------------
        # Dynamic Features
        # ------------------------------------------------
        self.velocity = {}      # angular velocity
        self.acceleration = {}  # angular acceleration

        # ------------------------------------------------
        # Stability / Tracking Metrics
        # ------------------------------------------------
        self.displacement = {}  # landmark drift between frames
        self.symmetry = {}      # left-right body symmetry


    # ====================================================
    # Debug / Visualization Output
    # ====================================================
    def __str__(self):
        """
        Pretty-print frame state for debugging.
        """

        def fmt(val):
            return f"{val:.2f}" if val is not None else "None"

        return (
            "Frame State\n"
            "-----------\n"

            # ----------------------------------------
            # Angles
            # ----------------------------------------
            "ANGLES:\n"
            f"  R Elbow: {fmt(self.angles.get('right_elbow'))}\n"
            f"  L Elbow: {fmt(self.angles.get('left_elbow'))}\n"
            f"  R Shoulder: {fmt(self.angles.get('right_shoulder'))}\n"
            "\n"

            # ----------------------------------------
            # Motion
            # ----------------------------------------
            "MOTION:\n"
            f"  R Shoulder Motion: {fmt(self.motion.get('right_shoulder'))}\n"
            "\n"

            # ----------------------------------------
            # Dynamics
            # ----------------------------------------
            "DYNAMICS:\n"
            f"  R Elbow Vel: {fmt(self.velocity.get('right_elbow'))}\n"
            f"  R Elbow Acc: {fmt(self.acceleration.get('right_elbow'))}\n"
            "\n"

            # ----------------------------------------
            # Stability
            # ----------------------------------------
            "STABILITY:\n"
            f"  R Elbow Drift: {fmt(self.displacement.get('right_elbow'))}\n"
        )