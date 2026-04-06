# server/app.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from models import Observation, Action, Reward
from .meta_env_environment import TicketEnv
from .tasks import EasyTask, MediumTask, HardTask

app = FastAPI()
env = TicketEnv()
task_graders = {
    "easy": EasyTask("easy"),
    "medium": MediumTask("medium"),
    "hard": HardTask("hard")
}
current_trajectory = []

# ---------- HTML root page for human visitors ----------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Ticket Prioritization RL Demo</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 40px auto; padding: 20px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        button, select { margin: 10px 5px; padding: 8px 12px; font-size: 14px; }
        .reward { font-weight: bold; color: green; }
        .done { color: red; }
        pre { background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>🎫 Customer Support Ticket Prioritization RL</h1>
    <p>Interactive demo: reset environment, choose a ticket to solve, and see the reward.</p>
    <button id="resetBtn">🔄 Reset Environment</button>
    <div id="status"></div>
    <table id="ticketTable">
        <thead>
            <tr><th>Index</th><th>Priority</th><th>Waiting Time</th><th>Solve Time</th><th>SLA Deadline</th><th>Customer Value</th></tr>
        </thead>
        <tbody></tbody>
    </table>
    <div>
        <label for="actionSelect">Choose ticket index to solve:</label>
        <select id="actionSelect"></select>
        <button id="stepBtn">⚡ Step (Solve Ticket)</button>
    </div>
    <div>
        <p><strong>Last Reward:</strong> <span id="rewardValue" class="reward">—</span></p>
        <p><strong>Episode Done:</strong> <span id="doneStatus">❌ No</span></p>
        <p><strong>Cumulative Reward (this session):</strong> <span id="cumulativeReward">0.00</span></p>
    </div>
    <hr>
    <p><a href="/download-client" download="inference.py">⬇️ Download inference.py (heuristic agent)</a> – run locally to see full logs.</p>
    <pre>API_BASE_URL = window.location.origin</pre>
    <script>
        let cumulative = 0.0;
        let currentObs = null;

        async function resetEnv() {
            const response = await fetch('/reset?task_id=easy');
            const data = await response.json();
            currentObs = data;
            cumulative = 0.0;
            document.getElementById('cumulativeReward').innerText = cumulative.toFixed(2);
            document.getElementById('rewardValue').innerText = '—';
            document.getElementById('doneStatus').innerHTML = '❌ No';
            renderTickets(data.tickets);
            updateActionSelect(data.tickets.length);
        }

        function renderTickets(tickets) {
            const tbody = document.querySelector('#ticketTable tbody');
            tbody.innerHTML = '';
            tickets.forEach((t, idx) => {
                const row = tbody.insertRow();
                row.insertCell(0).innerText = idx;
                row.insertCell(1).innerText = t.priority;
                row.insertCell(2).innerText = t.waiting_time.toFixed(1);
                row.insertCell(3).innerText = t.solve_time.toFixed(1);
                row.insertCell(4).innerText = t.sla_deadline.toFixed(1);
                row.insertCell(5).innerText = t.customer_value === 2 ? 'Premium' : 'Normal';
            });
        }

        function updateActionSelect(numTickets) {
            const select = document.getElementById('actionSelect');
            select.innerHTML = '';
            for (let i = 0; i < numTickets; i++) {
                const option = document.createElement('option');
                option.value = i;
                option.text = `Ticket ${i}`;
                select.appendChild(option);
            }
            if (numTickets === 0) {
                const option = document.createElement('option');
                option.text = 'No tickets';
                select.appendChild(option);
            }
        }

        async function step() {
            if (!currentObs) {
                alert('Please reset first.');
                return;
            }
            const select = document.getElementById('actionSelect');
            const actionIdx = parseInt(select.value);
            const response = await fetch('/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticket_index: actionIdx })
            });
            const data = await response.json();
            currentObs = data.observation;
            const reward = data.reward.value;
            const done = data.done;
            cumulative += reward;
            document.getElementById('rewardValue').innerHTML = reward.toFixed(2);
            document.getElementById('cumulativeReward').innerHTML = cumulative.toFixed(2);
            document.getElementById('doneStatus').innerHTML = done ? '✅ Yes' : '❌ No';
            renderTickets(currentObs.tickets);
            updateActionSelect(currentObs.tickets.length);
            if (done) {
                alert(`Episode finished! Final cumulative reward: ${cumulative.toFixed(2)}`);
            }
        }

        document.getElementById('resetBtn').addEventListener('click', resetEnv);
        document.getElementById('stepBtn').addEventListener('click', step);
        // Initial reset on page load
        resetEnv();
    </script>
</body>
</html>

"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_PAGE

@app.get("/download-client")
async def download_client():
    from fastapi.responses import FileResponse
    import os
    # Path to client.py (should be in the root of the container)
    client_path = "/app/client.py"
    if os.path.exists(client_path):
        return FileResponse(client_path, media_type="text/plain", filename="client.py")
    else:
        return HTMLResponse("client.py not found. Please ensure it is present in the container.", status_code=404)

# ---------- OpenEnv endpoints ----------
@app.api_route("/reset", methods=["GET", "POST"])
async def reset(task_id: str = Query("easy")):
    global current_trajectory
    obs = env.reset()
    current_trajectory = []
    return obs

@app.post("/step")
async def step(action: Action):
    global current_trajectory
    obs, reward_val, done, info = env.step(action.ticket_index)
    reward = Reward(value=reward_val)
    current_trajectory.append((obs, action, reward_val))
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.get("/state")
async def get_state():
    return {"state_vector": env.get_state_vector().tolist()}

@app.post("/grade")
async def grade(task_id: str):
    if task_id not in task_graders:
        raise HTTPException(400, f"Unknown task {task_id}")
    grader = task_graders[task_id]
    score = grader.grade(current_trajectory)
    return {"score": score}

@app.get("/ping")
async def ping():
    return {"status": "ok"}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()