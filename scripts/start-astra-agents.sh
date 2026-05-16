#!/usr/bin/env bash
set -euo pipefail

# Global model override.
# If empty, each agent uses the enterprise role-based model map below.
MODEL="${OLLAMA_AGENT_MODEL:-}"
MODEL_WAS_EXPLICIT=0

MODE="single"
AGENT=""
PHASE=""
PRINT_ONLY=0
INTERACTIVE=0
PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-acceptEdits}"
YES_FLAG=1

PROJECT_ROOT="$(pwd)"
AGENTS_DIR="astra-dev-council/agents"
LAUNCHER_DIR=".astra-dev-council-launchers"

declare -A AGENTS=(
  [architect]="01_architect_agent.md"
  [product]="02_product_agent.md"
  [rust]="03_rust_backend_agent.md"
  [frontend]="04_frontend_ui_agent.md"
  [orchestration]="05_ai_orchestration_agent.md"
  [voice]="06_voice_audio_agent.md"
  [vision]="07_screen_vision_agent.md"
  [security]="08_security_agent.md"
  [qa]="09_qa_agent.md"
  [release]="10_release_manager_agent.md"
)

# Enterprise-grade default model assignment.
# Use --model <name> to override this map for all launched agents.
declare -A AGENT_MODELS=(
  [architect]="ministral-3:8b" #"ministral-3:8b"
  [product]="ministral-3:8b"
  [security]="ministral-3:8b"
  [qa]="ministral-3:8b"
  [release]="ministral-3:8b"

  [rust]="qwen3-coder-next:cloud"
  [frontend]="qwen3-coder-next:cloud"
  [orchestration]="ministral-3:8b"
  [voice]="qwen3-coder-next:cloud"
  [vision]="ministral-3:8b"
)

declare -A AGENT_MODEL_NOTES=(
  [architect]="Strategic architecture, enterprise planning, system boundaries, technical governance."
  [product]="Product value, roadmap prioritization, scope control."
  [security]="Security policy, permissions, risk gates, destructive-action prevention."
  [qa]="QA/regression review. Optional fallback: qwen3.6:35b if ministral-3:8b is unavailable."
  [release]="Final release review, merge readiness, changelog and rollback notes."

  [rust]="Rust/Tauri implementation, backend orchestration, tool execution, state machines."
  [frontend]="React/TypeScript UI implementation, panels, state, UX integration."
  [orchestration]="AI routing and planner reasoning. Can be switched to qwen3-coder-next:cloud when implementation-heavy."
  [voice]="Voice/audio implementation, STT/TTS/VAD/barge-in pipeline work."
  [vision]="Reasoning over screen-vision outputs. Runtime visual model remains qwen2.5vl:7b where Astra needs raw image understanding."
)

usage() {
  cat <<EOF
Astra Development Council launcher v13

This version uses role-based model assignment and defaults Claude Code to acceptEdits so authorized council files can be written in headless mode.

Default enterprise model map:
  Architect Agent        -> ministral-3:8b
  Product Agent          -> ministral-3:8b
  Security Agent         -> ministral-3:8b
  QA Agent               -> ministral-3:8b
  Release Manager        -> ministral-3:8b

  Rust Backend Agent     -> qwen3-coder-next:cloud
  Frontend UI Agent      -> qwen3-coder-next:cloud
  AI Orchestration Agent -> ministral-3:8b
  Voice/Audio Agent      -> qwen3-coder-next:cloud
  Screen Vision Agent    -> ministral-3:8b
                            runtime vision note: qwen2.5vl:7b for raw image analysis inside Astra

Options:
  --model <name>              Override the model for every launched agent.
                              Example: --model kimi-k2.5:cloud
  --agent <name>              architect, product, rust, frontend, orchestration, voice, vision, security, qa, release
  --all                       Start all 10 agents
  --phase <name>              planning, implementation, validation
  --interactive               Open Claude Code interactively, with the role injected as system prompt.
                              Without this flag, the launcher uses headless -p mode.
  --permission-mode <mode>    Claude Code permission mode. Default: acceptEdits. Use default/plan/bypassPermissions only intentionally.
  --no-yes                    Do not pass Ollama --yes.
  --print                     Print commands instead of launching windows
  --help                      Show this help

Recommended role-based launch:
  ./scripts/start-astra-agents.sh --phase planning --interactive
  ./scripts/start-astra-agents.sh --phase implementation --interactive
  ./scripts/start-astra-agents.sh --phase validation --interactive

Global override launch:
  ./scripts/start-astra-agents.sh --model kimi-k2.5:cloud --phase planning --interactive

Ollama cloud models may require subscription/login:
  ollama signin
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; MODEL_WAS_EXPLICIT=1; shift 2 ;;
    --agent) MODE="single"; AGENT="$2"; shift 2 ;;
    --all) MODE="all"; shift ;;
    --phase) MODE="phase"; PHASE="$2"; shift 2 ;;
    --interactive) INTERACTIVE=1; shift ;;
    --permission-mode) PERMISSION_MODE="$2"; shift 2 ;;
    --no-yes) YES_FLAG=0; shift ;;
    --print) PRINT_ONLY=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -d "$AGENTS_DIR" ]]; then
  echo "Missing $AGENTS_DIR."
  echo "Run this script from the Astra project root after copying astra-dev-council into the repo."
  exit 1
fi

agent_prompt_path() {
  local name="$1"
  local file="${AGENTS[$name]:-}"
  if [[ -z "$file" ]]; then
    echo "Unknown agent: $name" >&2
    exit 1
  fi
  echo "$AGENTS_DIR/$file"
}

agent_model() {
  local name="$1"

  if [[ -n "$MODEL" ]]; then
    echo "$MODEL"
    return
  fi

  local default_model="${AGENT_MODELS[$name]:-}"
  if [[ -z "$default_model" ]]; then
    echo "ministral-3:8b"
    return
  fi

  echo "$default_model"
}

agent_model_source() {
  if [[ -n "$MODEL" ]]; then
    if [[ "$MODEL_WAS_EXPLICIT" == "1" ]]; then
      echo "global --model override"
    else
      echo "OLLAMA_AGENT_MODEL override"
    fi
  else
    echo "role-based default"
  fi
}

to_windows_path() {
  local path="$1"
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -w "$path"
  else
    echo "$path"
  fi
}

write_agent_files() {
  local name="$1"
  local prompt="$2"
  local selected_model="$3"
  local selected_model_source="$4"
  local model_note="${AGENT_MODEL_NOTES[$name]:-}"

  mkdir -p "$LAUNCHER_DIR"

  local root_win
  root_win="$(to_windows_path "$PROJECT_ROOT")"

  local ps1="$LAUNCHER_DIR/start-$name.ps1"
  local ps1_win
  ps1_win="$(to_windows_path "$PROJECT_ROOT/$ps1")"

  local role_txt="$LAUNCHER_DIR/role-$name.md"
  local task_txt="$LAUNCHER_DIR/task-$name.txt"

  local role_txt_win
  role_txt_win="$(to_windows_path "$PROJECT_ROOT/$role_txt")"

  local task_txt_win
  task_txt_win="$(to_windows_path "$PROJECT_ROOT/$task_txt")"

  local agent_title="Astra $name agent"

  cat > "$role_txt" <<EOF
You are running as: $agent_title.

Model assignment:
- Active model: $selected_model
- Model source: $selected_model_source
- Role model note: $model_note

First, read and follow this role file:
$prompt

Then read:
- astra-dev-council/TASK.md
- astra-dev-council/ACTIVE_PLAN.md
- astra-dev-council/AGENT_BOARD.md
- astra-dev-council/FILE_LOCKS.md
- astra-dev-council/DECISIONS.md
- astra-dev-council/CHANGELOG_AGENTIC.md
- astra-dev-council/SECURITY_REVIEW.md
- astra-dev-council/QA_REPORT.md

Adopt only your assigned role.
Respect file locks.
Do not perform destructive git commands.
Do not create branches or worktrees.
Work on main only.
Keep changes enterprise-grade, stable, reversible, and aligned with the current Astra architecture.

Important:
- If the current model is a global override, still respect your agent role and scope.
- If your role requires implementation, make small, reversible changes and document them.
- If your role is governance/review, do not implement application code unless the current task explicitly authorizes it.
EOF

  cat > "$task_txt" <<EOF
Read $prompt, adopt that role, then read astra-dev-council/TASK.md and all required council files before acting.

You are running in HEADLESS execution mode with Claude Code permission mode configured by the launcher.

Do not ask the human for permission to modify authorized council governance files.
Do not stop after describing what you would do.

Execute your authorized council-governance work now:

1. Read your role file and all required council files.
2. Create or update your own plan file under astra-dev-council/plans/.
3. Update AGENT_BOARD.md with your current status, blockers, handoffs, and completion state.
4. Update CHANGELOG_AGENTIC.md with the council files modified and the rationale.
5. Update FILE_LOCKS.md only if you take, update, or release a real lock.
6. Update your role-specific governance file if authorized by your prompt:
   - Architect: ACTIVE_PLAN.md and DECISIONS.md when planning/architecture-related.
   - Product: DECISIONS.md and AGENT_BOARD.md when product-scope-related.
   - Security: SECURITY_REVIEW.md and DECISIONS.md when safety/risk-related.
   - QA: QA_REPORT.md when validation/regression-related.
   - Release: CHANGELOG_AGENTIC.md, DECISIONS.md, AGENT_BOARD.md, and lock-resolution notes when release-related.
7. Do not modify application source code under src/ or src-tauri/ unless ACTIVE_PLAN.md explicitly authorizes an implementation phase for your role.
8. Do not perform destructive git commands, dependency installation, branch/worktree creation, or writes outside your authorized scope.

Required output:
- Actually perform the allowed council document updates.
- Then provide a short final report listing files modified, decisions made, blockers, and next handoff.
EOF

  cat > "$ps1" <<EOF
\$ErrorActionPreference = "Stop"
\$host.UI.RawUI.WindowTitle = "$agent_title"

Set-Location -LiteralPath @'
$root_win
'@

Write-Host ""
Write-Host "$agent_title" -ForegroundColor Cyan
Write-Host "Model: $selected_model" -ForegroundColor DarkCyan
Write-Host "Model source: $selected_model_source" -ForegroundColor DarkGray
Write-Host "Model note: $model_note" -ForegroundColor DarkGray
Write-Host "Permission mode: $PERMISSION_MODE" -ForegroundColor DarkGray
Write-Host "Prompt file: $prompt" -ForegroundColor Yellow
Write-Host "Role injection file: $role_txt_win" -ForegroundColor DarkGray
Write-Host "Task file: $task_txt_win" -ForegroundColor DarkGray
Write-Host ""

\$ollamaArgs = @("launch", "claude", "--model", "$selected_model")

if ($([[ "$YES_FLAG" == "1" ]] && echo "\$true" || echo "\$false")) {
  \$ollamaArgs += "--yes"
}

\$ollamaArgs += "--"

if ($([[ "$INTERACTIVE" == "1" ]] && echo "\$true" || echo "\$false")) {
  \$ollamaArgs += "--append-system-prompt"
  \$ollamaArgs += (Get-Content -LiteralPath "$role_txt_win" -Raw)

  if ("$PERMISSION_MODE" -ne "") {
    \$ollamaArgs += "--permission-mode"
    \$ollamaArgs += "$PERMISSION_MODE"
  }
}
else {
  \$ollamaArgs += "-p"
  \$ollamaArgs += (Get-Content -LiteralPath "$task_txt_win" -Raw)
  \$ollamaArgs += "--append-system-prompt"
  \$ollamaArgs += (Get-Content -LiteralPath "$role_txt_win" -Raw)

  if ("$PERMISSION_MODE" -ne "") {
    \$ollamaArgs += "--permission-mode"
    \$ollamaArgs += "$PERMISSION_MODE"
  }
}

& ollama @ollamaArgs
EOF

  echo "$ps1_win"
}

launch_window() {
  local name="$1"
  local prompt
  prompt="$(agent_prompt_path "$name")"

  local selected_model
  selected_model="$(agent_model "$name")"

  local selected_model_source
  selected_model_source="$(agent_model_source)"

  local ps1_win
  ps1_win="$(write_agent_files "$name" "$prompt" "$selected_model" "$selected_model_source")"

  if [[ "$PRINT_ONLY" == "1" ]]; then
    echo "[$name]"
    echo "model: $selected_model ($selected_model_source)"
    echo "wt.exe -w new new-tab --title \"Astra $name agent\" --suppressApplicationTitle -- powershell.exe -NoExit -NoProfile -ExecutionPolicy Bypass -File \"$ps1_win\""
    echo
    return
  fi

  if command -v wt.exe >/dev/null 2>&1; then
    wt.exe -w new new-tab --title "Astra $name agent" --suppressApplicationTitle -- powershell.exe -NoExit -NoProfile -ExecutionPolicy Bypass -File "$ps1_win" >/dev/null 2>&1 &
  elif command -v gnome-terminal >/dev/null 2>&1; then
    local bash_cmd="cd \"$PROJECT_ROOT\"; powershell.exe -NoExit -NoProfile -ExecutionPolicy Bypass -File \"$ps1_win\""
    gnome-terminal --title="Astra $name agent" -- bash -lc "$bash_cmd; exec bash" &
  else
    echo "No supported terminal launcher found."
    echo "Run manually:"
    echo "powershell.exe -NoExit -NoProfile -ExecutionPolicy Bypass -File \"$ps1_win\""
  fi

  sleep 1
}

start_phase() {
  local phase="$1"
  case "$phase" in
    planning)
      for a in architect product security qa; do launch_window "$a"; done
      ;;
    implementation)
      for a in rust frontend orchestration voice vision; do launch_window "$a"; done
      ;;
    validation)
      for a in qa security release; do launch_window "$a"; done
      ;;
    *)
      echo "Unknown phase: $PHASE"
      usage
      exit 1
      ;;
  esac
}

case "$MODE" in
  single)
    [[ -n "$AGENT" ]] || { usage; exit 1; }
    launch_window "$AGENT"
    ;;
  all)
    for a in architect product security qa rust frontend orchestration voice vision release; do
      launch_window "$a"
    done
    ;;
  phase)
    [[ -n "$PHASE" ]] || { usage; exit 1; }
    start_phase "$PHASE"
    ;;
esac
