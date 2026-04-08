# server/tasks.py
from typing import List, Tuple
from models import Observation, Action

class TaskGrader:
    def __init__(self, task_id: str):
        self.task_id = task_id

    def grade(self, trajectory: List[Tuple[Observation, Action, float]]) -> float:
        raise NotImplementedError

class EasyTask(TaskGrader):
    def grade(self, trajectory):
        rewards = [r for _, _, r in trajectory]
        if not rewards:
            return 0.5  # fallback
        avg_reward = sum(rewards) / len(rewards)
        # Map average reward (typically -20..20) to (0.001, 0.999)
        # Clamp raw score to [0,1] then shift slightly inward
        raw = max(0.0, min(1.0, (avg_reward + 20) / 40))  # assumes range -20..20
        # Avoid 0.0 and 1.0
        return max(0.001, min(0.999, raw))

class MediumTask(TaskGrader):
    def grade(self, trajectory):
        total_violations = 0
        for obs, _, _ in trajectory:
            total_violations += obs.urgent_count
        max_violations = 20
        raw = 1.0 - min(1.0, total_violations / max_violations)
        # Avoid 0.0 and 1.0
        return max(0.001, min(0.999, raw))

class HardTask(TaskGrader):
    def grade(self, trajectory):
        if not trajectory:
            return 0.5
        final_backlog = trajectory[-1][0].backlog_size
        raw = 1.0 / (1.0 + final_backlog)
        # raw is always >0, but can be 1.0 when backlog=0
        if raw >= 1.0:
            return 0.999
        if raw <= 0.0:
            return 0.001
        return raw
