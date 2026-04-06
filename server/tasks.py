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
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        return min(1.0, max(0.0, avg_reward / 10.0))

class MediumTask(TaskGrader):
    def grade(self, trajectory):
        total_violations = 0
        for obs, _, _ in trajectory:
            total_violations += obs.urgent_count
        max_violations = 20
        score = 1.0 - min(1.0, total_violations / max_violations)
        return score

class HardTask(TaskGrader):
    def grade(self, trajectory):
        if not trajectory:
            return 0.0
        final_backlog = trajectory[-1][0].backlog_size
        score = 1.0 / (1.0 + final_backlog)
        return score