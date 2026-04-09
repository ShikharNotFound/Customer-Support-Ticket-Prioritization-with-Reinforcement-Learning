# server/app.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from models import Observation, Action, Reward
from .meta_env_environment import TicketEnv
from .tasks import EasyTask, MediumTask, HardTask

app = FastAPI()

task_graders = {
    "easy":   EasyTask("easy"),
    "medium": MediumTask("medium"),
    "hard":   HardTask("hard"),
}

# ✅ FIX 2: per-task trajectory storage
task_trajectories = {"easy": [], "medium": [], "hard": []}
current_task_id = "easy"

# ✅ FIX 3: task-specific env configs
TASK_CONFIGS = {
    "easy":   {"num_tickets": 5,  "sla_pressure": False},
    "medium": {"num_tickets": 10, "sla_pressure": True},
    "hard":   {"num_tickets": 20, "sla_pressure": True},
}

env = TicketEnv()

# ... (keep your HTML_PAGE and root/download endpoints unchanged) ...

@app.api_route("/reset", methods=["GET", "POST"])
async def reset(task_id: str = Query("easy")):
    global current_task_id
    if task_id not in task_graders:
        raise HTTPException(400, f"Unknown task_id: {task_id}")

    current_task_id = task_id

    # ✅ FIX 2: clear only THIS task's trajectory
    task_trajectories[task_id] = []

    # ✅ FIX 3: configure env difficulty based on task
    config = TASK_CONFIGS[task_id]
    obs = env.reset(
        num_tickets=config["num_tickets"],
        sla_pressure=config["sla_pressure"],
    )
    return obs


@app.post("/step")
async def step(action: Action):
    obs, reward_val, done, info = env.step(action.ticket_index)
    reward = Reward(value=reward_val)

    # ✅ FIX 2: append to the current task's trajectory
    task_trajectories[current_task_id].append((obs, action, reward_val))

    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state")
async def get_state():
    return {"state_vector": env.get_state_vector().tolist()}


# ✅ FIX 1: GET endpoint with Query param
@app.get("/grade")
async def grade(task_id: str = Query("easy")):
    if task_id not in task_graders:
        raise HTTPException(400, f"Unknown task: {task_id}")

    trajectory = task_trajectories.get(task_id, [])
    if not trajectory:
        return {"score": 0.5}

    score = task_graders[task_id].grade(trajectory)
    score = max(0.001, min(0.999, score))
    return {"score": score}


@app.get("/ping")
async def ping():
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
