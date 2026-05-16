# START_HERE — How to use this package

## 1. Copy folders into the Astra project root

Copy these folders into:

```txt
C:\Users\Simone\personal_ai\personal_ai
```

Folders:

```txt
astra-dev-council/
scripts/
```

## 2. Make the bash launcher executable

From Git Bash:

```bash
cd /c/Users/Simone/personal_ai/personal_ai
chmod +x scripts/start-astra-agents.sh
```

## 3. Start planning agents first

```bash
./scripts/start-astra-agents.sh --model kimi-k2.5:cloud --phase planning
```

This opens/starts:

- Architect Agent
- Product Agent
- Security Agent

## 4. Start implementation agents only after planning/security are ready

```bash
./scripts/start-astra-agents.sh --model kimi-k2.5:cloud --phase implementation
```

This starts:

- Rust Backend Agent
- Frontend UI Agent
- AI Orchestration Agent
- Voice/Audio Agent
- Screen Vision Agent

For a targeted task, prefer one agent at a time:

```bash
./scripts/start-astra-agents.sh --model kimi-k2.5:cloud --agent rust
```

## 5. Start validation agents

```bash
./scripts/start-astra-agents.sh --model kimi-k2.5:cloud --phase validation
```

## 6. Manual instruction to paste in each Claude session

```txt
Read your prompt file in astra-dev-council/agents, adopt that role, then read astra-dev-council/TASK.md and all required council files before acting. Work on main, respect FILE_LOCKS.md, update AGENT_BOARD.md and CHANGELOG_AGENTIC.md after your turn.
```

## 7. Recommended model usage

For deep coding/architecture:

```bash
--model kimi-k2.5:cloud
```

For review/cross-checking, test a different strong model if available:

```bash
--model qwen3.5:cloud
```

For Astra's live screen-analysis path, keep the dedicated vision model such as `qwen2.5vl:7b`. Do not use the vision model as the main coding architect.
