# server/meta_env_environment.py
import numpy as np
import uuid
from typing import List, Tuple
from models import Ticket, Observation


class TicketEnv:
    def __init__(self, max_tickets: int = 10, episode_steps: int = 200):
        self.max_tickets = max_tickets
        self.episode_steps = episode_steps
        self.current_time = 0.0
        self.tickets: List[Ticket] = []
        self.step_count = 0
        self.done = False
        self.arrival_prob = 0.5
        self.sla_pressure = False  # ✅ added

    def reset(self, num_tickets: int = None, sla_pressure: bool = False) -> Observation:  # ✅ added params
        self.current_time = 0.0
        self.step_count = 0
        self.done = False
        self.tickets = []

        # ✅ Allow task-specific ticket count; fall back to constructor default
        effective_max = num_tickets if num_tickets is not None else self.max_tickets
        self.max_tickets = effective_max

        # ✅ SLA pressure: tighter deadlines for medium/hard tasks
        self.sla_pressure = sla_pressure

        for _ in range(np.random.randint(3, min(8, effective_max + 1))):
            self._add_random_ticket()
        return self._get_observation()

    def step(self, action: int) -> Tuple[Observation, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode already done")
        if action >= len(self.tickets):
            reward = -5.0
            obs = self._get_observation()
            self.step_count += 1
            self.done = self.step_count >= self.episode_steps
            return obs, reward, self.done, {"error": "invalid index"}

        selected = self.tickets.pop(action)
        time_elapsed = selected.solve_time
        self.current_time += time_elapsed

        for t in self.tickets:
            t.waiting_time += time_elapsed

        if np.random.rand() < self.arrival_prob and len(self.tickets) < self.max_tickets:
            self._add_random_ticket()

        reward = self._compute_reward(selected)

        self.step_count += 1
        self.done = self.step_count >= self.episode_steps or len(self.tickets) == 0

        obs = self._get_observation()
        info = {"violations": self._count_sla_violations(), "queue_size": len(self.tickets)}
        return obs, reward, self.done, info

    def get_state_vector(self) -> np.ndarray:
        state = []
        for t in self.tickets:
            state.extend([t.priority, t.waiting_time, t.solve_time, t.sla_deadline, t.customer_value])
        padding = [0.0] * (self.max_tickets * 5 - len(state))
        state.extend(padding)
        return np.array(state, dtype=np.float32)

    def _get_observation(self) -> Observation:
        return Observation(
            tickets=self.tickets.copy(),
            current_time=self.current_time,
            urgent_count=self._count_sla_violations(),
            backlog_size=len(self.tickets)
        )

    def _count_sla_violations(self) -> int:
        return sum(1 for t in self.tickets if self.current_time > t.sla_deadline)

    def _add_random_ticket(self):
        priority = np.random.randint(1, 6)
        solve_time = np.random.uniform(1, 10)

        # ✅ Tighter SLA deadlines under pressure (medium/hard tasks)
        if self.sla_pressure:
            sla_deadline = self.current_time + np.random.uniform(5, 15)
        else:
            sla_deadline = self.current_time + np.random.uniform(10, 30)

        customer_value = np.random.choice([1, 2], p=[0.7, 0.3])
        ticket = Ticket(
            id=str(uuid.uuid4()),
            priority=priority,
            solve_time=solve_time,
            waiting_time=0.0,
            sla_deadline=sla_deadline,
            customer_value=customer_value,
        )
        self.tickets.append(ticket)

    def _compute_reward(self, solved_ticket: Ticket) -> float:
        priority_reward = 2.0 * solved_ticket.priority
        customer_reward = 3.0 if solved_ticket.customer_value == 2 else 0.0
        total_wait = sum(t.waiting_time for t in self.tickets)
        waiting_penalty = -0.05 * total_wait
        violations = self._count_sla_violations()
        sla_penalty = -10.0 * violations
        backlog_penalty = -0.2 * len(self.tickets)
        reward = priority_reward + customer_reward + waiting_penalty + sla_penalty + backlog_penalty
        return np.clip(reward, -20.0, 20.0)
