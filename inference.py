import os
import sys
import asyncio
import httpx
from openai import AsyncOpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

OPENENV_API_URL = os.getenv("OPENENV_API_URL", "http://127.0.0.1:7860")
client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASK_NAME = "customer-support-ticket-prioritization"
BENCHMARK = "OpenEnv"
MAX_STEPS = 50
MAX_TOTAL_REWARD = 200.0
SUCCESS_SCORE_THRESHOLD = 0.5


TASKS = ["easy", "medium", "hard"]


async def openai_action(obs_dict):
    tickets = obs_dict.get("tickets", [])
    if not tickets:
        return 0
    prompt = "You are a customer support manager. Prioritize tickets by solving the most urgent one first.\n"
    for i, t in enumerate(tickets):
        prompt += (
            f"Ticket {i}: priority={t['priority']}, "
            f"waiting_time={t['waiting_time']:.1f}, "
            f"solve_time={t['solve_time']:.1f}, "
            f"deadline={t['sla_deadline']:.1f}\n"
        )
    prompt += "Respond with only the ticket index (0,1,2,...)."
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        idx = int(response.choices[0].message.content.strip())
        if 0 <= idx < len(tickets):
            return idx
    except Exception as e:
        print(f"OpenAI error: {e}, using heuristic fallback", file=sys.stderr)
    # Heuristic fallback
    best_idx, best_score = 0, -1
    for i, t in enumerate(tickets):
        score = t["priority"] * 10 - t["solve_time"]
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


async def run_task(http: httpx.AsyncClient, task_id: str):
    resp = await http.get(f"{OPENENV_API_URL}/reset?task_id={task_id}")
    if resp.status_code != 200:
        print(f"Reset failed for task {task_id}: {resp.status_code}", file=sys.stderr)
        return

    obs = resp.json()
    rewards = []
    done = False
    error = None


    print(f"[START] task={TASK_NAME}-{task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        action_idx = await openai_action(obs)
        action_str = str(action_idx)

        step_resp = await http.post(
            f"{OPENENV_API_URL}/step",
            json={"ticket_index": action_idx},
        )
        data = step_resp.json()
        reward = data["reward"]["value"]
        done = data["done"]
        obs = data["observation"]
        rewards.append(reward)

        print(
            f"[STEP] step={step} action={action_str} reward={reward:.2f} "
            f"done={str(done).lower()} error={error if error else 'null'}",
            flush=True,
        )

    total_reward = sum(rewards)
    raw_score = total_reward / MAX_TOTAL_REWARD

   
    score = max(0.001, min(0.999, raw_score))
    success = score >= SUCCESS_SCORE_THRESHOLD
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

  
    print(
        f"[END] success={str(success).lower()} steps={len(rewards)} "
        f"score={score:.6f} rewards={rewards_str}",
        flush=True,
    )


async def main():
    async with httpx.AsyncClient() as http:
    
        for task_id in TASKS:
            await run_task(http, task_id)


if __name__ == "__main__":
    asyncio.run(main())
