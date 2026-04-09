# server/tasks.py
from typing import List, Tuple
from models import Observation, Action

EPSILON = 0.001  # offset to keep scores strictly inside (0, 1)


class TaskGrader:
    def __init__(self, task_id: str):
        self.task_id = task_id

    def grade(self, trajectory: List[Tuple[Observation, Action, float]]) -> float:
        raise NotImplementedError


class EasyTask(TaskGrader):

    def grade(self, trajectory):
        if not trajectory:
            return 0.5

        rewards = [r for _, _, r in trajectory]
        avg_reward = sum(rewards) / len(rewards)

        # Map [-20, 20] → [0, 1]
        raw = (avg_reward + 20) / 40

        # Clamp strictly inside (0, 1)
        return max(EPSILON, min(1.0 - EPSILON, raw))


class MediumTask(TaskGrader):

    def grade(self, trajectory):
        if not trajectory:
            return 0.5

        total_violations = sum(obs.urgent_count for obs, _, _ in trajectory)

        # Normalise against actual episode length, not a magic constant
        max_violations = len(trajectory)  # one violation per step = worst case

        raw = 1.0 - (total_violations / max_violations)

        # Clamp strictly inside (0, 1)
        return max(EPSILON, min(1.0 - EPSILON, raw))


class HardTask(TaskGrader):

    def grade(self, trajectory):
        if not trajectory:
            return 0.5

        final_backlog = trajectory[-1][0].backlog_size
        raw = 1.0 / (1.0 + final_backlog)

        # Clamp strictly inside (0, 1)
        return max(EPSILON, min(1.0 - EPSILON, raw))
