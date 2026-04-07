---
title: Ticket Prioritization RL
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: server/app.py
app_port: 7860
pinned: false
---


# Customer Support Ticket Prioritization with Reinforcement Learning

## 📌 Overview

This environment simulates a **real‑world customer support ticket queue** where an agent must decide which ticket to solve next. Tickets arrive dynamically with different:

- **Priorities** (1 = low, 5 = critical)
- **Estimated solving times**
- **SLA deadlines** (tickets become “urgent” if not solved in time)
- **Customer values** (premium vs. normal)

The agent receives a **dense reward signal** at every step, balancing:
- Solving high‑priority and premium tickets (positive reward)
- Reducing waiting time and backlog (negative penalties)
- Avoiding SLA violations (large penalty)

The environment is **OpenEnv‑compliant**, exposing a REST API that can be used by any RL agent. A **baseline inference script** using GPT‑3.5/4 is provided, and a **DQN training script** is included for those who wish to train their own agent.

---

## 🎯 Motivation

Customer support queues are a high‑impact, real‑world problem. Manually prioritising tickets is error‑prone and time‑consuming. This environment allows researchers and practitioners to develop, test, and compare **automated prioritisation policies** using reinforcement learning – with the ultimate goal of improving customer satisfaction and operational efficiency.

---

## 🧠 Environment Details

### Observation Space

At each step, the environment returns an `Observation` (Pydantic model) containing:

- `tickets`: list of active tickets, each with:
  - `id` (str)
  - `priority` (int 1‑5)
  - `solve_time` (float, minutes)
  - `waiting_time` (float, minutes waited so far)
  - `sla_deadline` (float, absolute time when SLA expires)
  - `customer_value` (1 = normal, 2 = premium)
- `current_time` (float)
- `urgent_count` (number of tickets that have missed their SLA)
- `backlog_size` (total tickets in queue)

A **state vector** of fixed size (max_tickets × 5) is also available via `/state` for RL algorithms.

### Action Space

- **Type**: Discrete integer
- **Range**: `0` to `max_tickets - 1` (default 20)
- **Meaning**: Index of the ticket to solve from the current `tickets` list.

If the agent selects an invalid index (e.g., when the queue is empty), a penalty of `-5` is applied and the environment advances one step.

### Reward Function (Dense)

reward = 
    (2.0 × priority_of_solved_ticket)
    + (3.0 if customer is premium else 0)
    - (0.05 × total_waiting_time_of_all_remaining_tickets)
    - (10.0 × number_of_SLA_violations_after_step)
    - (0.2 × current_backlog_size)


The final reward is clipped to the range **[-20, 20]**. This design encourages the agent to:
- Solve high‑priority and premium tickets early
- Keep the queue short
- Avoid missing SLA deadlines

---

## 🧪 Three Tasks with Graders

| Task       | Difficulty | Description                                                          | Grader Logic                                           |
|------------|------------|----------------------------------------------------------------------|--------------------------------------------------------|
| **easy**   | Easy       | Low/medium priority tickets, no tight deadlines.                     | Scores based on average reward per step (clipped 0‑1). |
| **medium** | Medium     | Mixed priorities and SLA deadlines.                                  | Score = 1 − (SLA violations / 20), clamped.            |
| **hard**   | Hard       | High‑priority long tickets competing with many quick medium tickets. | Score = 1 / (1 + final backlog size).                  |

Each task has a **programmatic grader** that returns a score between `0.0` and `1.0` via the `/grade?task_id=...` endpoint.

---


## 🚀 Getting Started

### 1. Run the Environment Server (Local)

```bash
# Clone the repository
git clone https://huggingface.co/spaces/ShikharNotFound/nonchalants
cd nonchalants

# Install dependencies
pip install -r server/requirements.txt

# Start the FastAPI server
uvicorn server.app:app --reload --port 7860