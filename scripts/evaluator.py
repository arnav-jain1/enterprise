import numpy as np
from scipy.signal import savgol_filter


class OverallEvaluator:
    """
    Aggregates frame-level issues and signals into rep-level evaluation.

    Responsibilities:
    - Confirm persistent issues using temporal windows
    - Analyze trajectories (e.g., bar path parabola)
    """

    def __init__(self, frames):
        self.frames = frames

    # --------------------------------------------------
    # 1. WINDOW-BASED ISSUE DETECTION
    # --------------------------------------------------
    def detect_persistent_issues(self, issue_key, window_size=10, min_ratio=0.8):
        """
        Detect if an issue persists across a window of frames.

        Parameters
        ----------
        issue_key : str
            Name of issue (e.g., "back_rounding")
        window_size : int
            Number of consecutive frames to check
        min_ratio : float
            % of frames in window that must contain the issue

        Returns
        -------
        bool
            True if issue is persistent
        """

        issue_flags = [
            issue_key in frame.issues if hasattr(frame, "issues") else False
            for frame in self.frames
        ]

        # Sliding window
        for i in range(len(issue_flags) - window_size + 1):
            window = issue_flags[i:i + window_size]

            if sum(window) / window_size >= min_ratio:
                return True

        return False

    # --------------------------------------------------
    # 2. COUNT LONGEST STREAK (optional stronger check)
    # --------------------------------------------------
    def longest_streak(self, issue_key):
        """
        Returns longest consecutive streak of an issue.
        """
        max_streak = 0
        current = 0

        for frame in self.frames:
            has_issue = issue_key in getattr(frame, "issues", [])

            if has_issue:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0

        return max_streak

    # --------------------------------------------------
    # 3. PARABOLA DETECTION
    # --------------------------------------------------
    def detect_parabolic_motion_robust(values, smooth=True):
        """
        Robust parabola detection using smoothing + polynomial fit.

        Parameters
        ----------
        values : list or np.array
        smooth : bool

        Returns
        -------
        dict
        """

        y = np.array(values)

        # ----------------------------------------
        # 1. Smooth signal (VERY IMPORTANT)
        # ----------------------------------------
        if smooth and len(y) >= 5:
            y = savgol_filter(y, window_length=5, polyorder=2)

        x = np.arange(len(y))

        # ----------------------------------------
        # 2. Fit quadratic
        # ----------------------------------------
        a, b, c = np.polyfit(x, y, 2)
        y_fit = a * x**2 + b * x + c

        # ----------------------------------------
        # 3. Error (normalized)
        # ----------------------------------------
        mse = np.mean((y - y_fit) ** 2)
        variance = np.var(y) + 1e-6  # avoid divide by 0
        normalized_error = mse / variance

        # ----------------------------------------
        # 4. Vertex check (should be inside range)
        # ----------------------------------------
        vertex_x = -b / (2 * a) if a != 0 else None

        valid_vertex = (
            vertex_x is not None and
            0 < vertex_x < len(y) - 1
        )

        return {
            "is_parabolic": normalized_error < 0.2 and valid_vertex,
            "a": a,
            "error": normalized_error,
            "vertex": vertex_x,
            "y_smooth": y
        }
    
