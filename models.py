# meta_env/models.py
from pydantic import BaseModel
from typing import List
from enum import IntEnum

class Priority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class CustomerValue(IntEnum):
    NORMAL = 1
    PREMIUM = 2

class Ticket(BaseModel):
    id: str
    priority: int
    solve_time: float
    waiting_time: float
    sla_deadline: float
    customer_value: int

class Observation(BaseModel):
    tickets: List[Ticket]
    current_time: float
    urgent_count: int
    backlog_size: int

class Action(BaseModel):
    ticket_index: int

class Reward(BaseModel):
    value: float