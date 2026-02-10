class Frame:
    def __init__(self, landmarks):
        self.landmarks = landmarks

        self.angles = {}
        self.motion = {}
        self.velocity = {}
        self.acceleration = {}
        self.displacement = {}

    def __str__(self):
        def fmt(val):
            return f"{val:.2f}" if val is not None else "None"

        return (
            "Frame State\n"
            "-----------\n"
            "ANGLES:\n"
            f"  R Elbow: {fmt(self.angles.get('right_elbow'))}\n"
            f"  L Elbow: {fmt(self.angles.get('left_elbow'))}\n"
            f"  R Shoulder: {fmt(self.angles.get('right_shoulder'))}\n"
            "\n"
            "MOTION:\n"
            f"  R Shoulder Motion: {fmt(self.motion.get('right_shoulder'))}\n"
            "\n"
            "DYNAMICS:\n"
            f"  R Elbow Vel: {fmt(self.velocity.get('right_elbow'))}\n"
            f"  R Elbow Acc: {fmt(self.acceleration.get('right_elbow'))}\n"
            "\n"
            "STABILITY:\n"
            f"  R Elbow Drift: {fmt(self.displacement.get('right_elbow'))}\n"
        )
