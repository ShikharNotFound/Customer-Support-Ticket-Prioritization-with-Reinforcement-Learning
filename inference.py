
import asyncio
import httpx
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
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

def heuristic_action(obs_dict):
    tickets = obs_dict.get("tickets", [])
    if not tickets:
        return 0
    best_idx = 0
    best_score = -1e9
    current_time = obs_dict.get("current_time", 0.0)
    for i, t in enumerate(tickets):
        priority = t['priority']
        solve_time = t['solve_time']
        waiting = t['waiting_time']
        deadline = t['sla_deadline']
        # SLA urgency: how close to deadline? (lower time_left = more urgent)
        time_left = deadline - current_time
        urgency = 10.0 / (1.0 + time_left) if time_left > 0 else 100.0
        # Score: priority weight (5), urgency (2), waiting (0.1), penalise long solve_time
        score = 5 * priority + 2 * urgency + 0.1 * waiting - solve_time
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx

async def main():
    async with httpx.AsyncClient() as client:
        reset_url = f"{API_BASE_URL}/reset?task_id=easy"
        resp = await client.get(reset_url)
        if resp.status_code != 200:
            print(f"Reset failed: {resp.status_code} - {resp.text}")
            return
        obs = resp.json()

        rewards = []
        steps_taken = 0
        done = False

        log_start(task=TASK_NAME, env=BENCHMARK, model="heuristic")

        for step in range(1, MAX_STEPS+1):
            if done:
                break
            action = heuristic_action(obs)
            step_resp = await client.post(f"{API_BASE_URL}/step", json={"ticket_index": action})
            data = step_resp.json()
            reward = data["reward"]["value"]
            done = data["done"]
            obs = data["observation"]

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=None)

        total_reward = sum(rewards)
        score = total_reward / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())