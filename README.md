---
title: Customer Support Ticket Prioritization RL
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: server/app.py
pinned: false
---

# Customer Support Ticket Prioritization with DQN

## Description
Advanced RL environment where an agent learns to prioritize support tickets to maximise SLA compliance and customer satisfaction.

## Features
- 3 tasks (easy, medium, hard) with programmatic graders
- DQN agent (training script provided)
- OpenEnv compliant API
- HTML root page with documentation and download link

## Setup (local)
```bash
uv sync
uv run uvicorn server.app:app --reload --port 7860