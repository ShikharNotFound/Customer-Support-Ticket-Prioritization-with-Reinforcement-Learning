# inference.py
import os
import asyncio
import httpx
from openai import AsyncOpenAI

# ---------- Environment variables  ----------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

# OpenEnv server URL (not required by spec, but needed for communication)
# You can override with OPENENV_API_URL, default to localhost:7860
OPENENV_API_URL = os.getenv("OPENENV_API_URL", "http://127.0.0.1:7860")

# ---------- OpenAI client ----------
client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASK_NAME = "customer-support-ticket-prioritization"
BENCHMARK = "OpenEnv"
MAX_STEPS = 50
MAX_TOTAL_REWARD = 200.0
SUCCESS_SCORE_THRESHOLD = 0.7


async def openai_action(obs_dict):
    """Ask GPT which ticket to solve. Return index as integer."""
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
            max_tokens=5
        )
        idx = int(response.choices[0].message.content.strip())
        if 0 <= idx < len(tickets):
            return idx
    except Exception as e:
        # Fallback to heuristic on error
        print(f"OpenAI error: {e}, using heuristic fallback", file=sys.stderr)

    # Heuristic fallback: highest priority, shortest solve time
    best_idx = 0
    best_score = -1
    for i, t in enumerate(tickets):
        score = t['priority'] * 10 - t['solve_time']
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


async def main():
    async with httpx.AsyncClient() as http:
        # Reset environment
        reset_url = f"{OPENENV_API_URL}/reset?task_id=easy"
        resp = await http.get(reset_url)
        if resp.status_code != 200:
            print(f"Reset failed: {resp.status_code}", file=sys.stderr)
            return
        obs = resp.json()

        rewards = []
        done = False
        error = None

        # [START] line
        print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_idx = await openai_action(obs)
            action_str = str(action_idx)

            # Send step to environment
            step_resp = await http.post(
                f"{OPENENV_API_URL}/step",
                json={"ticket_index": action_idx}
            )
            data = step_resp.json()
            reward = data["reward"]["value"]
            done = data["done"]
            obs = data["observation"]
            rewards.append(reward)

            # [STEP] line
            print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error}")

        total_reward = sum(rewards)
        score = max(0.0, min(1.0, total_reward / MAX_TOTAL_REWARD))
        success = score >= SUCCESS_SCORE_THRESHOLD

        # Format rewards list as comma-separated with 2 decimals
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        # [END] line
        print(f"[END] success={str(success).lower()} steps={len(rewards)} rewards={rewards_str}")


if __name__ == "__main__":
    import sys
    asyncio.run(main())
