# inference.py
import asyncio
import httpx
import os
import json
from openai import AsyncOpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY or HF_TOKEN environment variable")

client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
TASK_NAME = "customer-support-ticket-prioritization"
BENCHMARK = "OpenEnv"
MAX_STEPS = 50
MAX_TOTAL_REWARD = 200.0
SUCCESS_SCORE_THRESHOLD = 0.7

def log_start(task, env, model):
    print(f'[START] {{"task": "{task}", "env": "{env}", "model": "{model}"}}', flush=True)

def log_step(step, action, reward, done, error):
    print(f'[STEP] {{"step": {step}, "action": "{action}", "reward": {reward}, "done": {done}, "error": {error}}}', flush=True)

def log_end(success, steps, score, rewards):
    print(f'[END] {{"success": {success}, "steps": {steps}, "score": {score}, "rewards": {rewards}}}', flush=True)

async def openai_action(obs_dict):
    tickets = obs_dict.get("tickets", [])
    if not tickets:
        return 0
    # Create a prompt
    prompt = "You are a customer support manager. Prioritize tickets by solving the most urgent one first.\n"
    for i, t in enumerate(tickets):
        prompt += f"Ticket {i}: priority={t['priority']}, waiting_time={t['waiting_time']:.1f}, solve_time={t['solve_time']:.1f}, deadline={t['sla_deadline']:.1f}\n"
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
        print(f"OpenAI error: {e}, falling back to heuristic")
    # Fallback heuristic
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
        reset_url = f"{API_BASE_URL}/reset?task_id=easy"
        resp = await http.get(reset_url)
        if resp.status_code != 200:
            print(f"Reset failed: {resp.status_code}")
            return
        obs = resp.json()
        rewards = []
        done = False
        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
        for step in range(1, MAX_STEPS+1):
            if done:
                break
            action = await openai_action(obs)
            step_resp = await http.post(f"{API_BASE_URL}/step", json={"ticket_index": action})
            data = step_resp.json()
            reward = data["reward"]["value"]
            done = data["done"]
            obs = data["observation"]
            rewards.append(reward)
            log_step(step=step, action=action, reward=reward, done=done, error=None)
        total_reward = sum(rewards)
        score = max(0.0, min(1.0, total_reward / MAX_TOTAL_REWARD))
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=len(rewards), score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())