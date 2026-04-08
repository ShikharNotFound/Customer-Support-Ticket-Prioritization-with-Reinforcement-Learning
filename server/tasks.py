# server/tasks.py
from typing import List, Tuple
from models import Observation, Action

EPSILON = 0.0001  # tiny offset to avoid 0.0 and 1.0

class TaskGrader:
    def __init__(self, task_id: str):
        self.task_id = task_id

    def grade(self, trajectory: List[Tuple[Observation, Action, float]]) -> float:
        raise NotImplementedError

class EasyTask(TaskGrader):
    def grade(self, trajectory):
        rewards = [r for _, _, r in trajectory]
        if not rewards:
            return 0.5  # middle of (0,1)
        avg_reward = sum(rewards) / len(rewards)
        # Map typical reward range [-20,20] to (0,1)
        raw = (avg_reward + 20) / 40  # now in [0,1]
        # Clamp with epsilon
        raw = max(EPSILON, min(1.0 - EPSILON, raw))
        return raw

class MediumTask(TaskGrader):
    def grade(self, trajectory):
        total_violations = 0
        for obs, _, _ in trajectory:
            total_violations += obs.urgent_count
        max_violations = 20
        if max_violations == 0:
            return 0.5
        raw = 1.0 - min(1.0, total_violations / max_violations)
        raw = max(EPSILON, min(1.0 - EPSILON, raw))
        return raw

class HardTask(TaskGrader):
    def grade(self, trajectory):
        if not trajectory:
            return 0.5
        final_backlog = trajectory[-1][0].backlog_size
        raw = 1.0 / (1.0 + final_backlog)  # always >0, but can be 1.0 when backlog=0
        raw = max(EPSILON, min(1.0 - EPSILON, raw))
        return raw
