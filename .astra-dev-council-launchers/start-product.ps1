$ErrorActionPreference = "Stop"
$host.UI.RawUI.WindowTitle = "Astra product agent"

Set-Location -LiteralPath @'
C:\Users\Simone\personal_ai\personal_ai
'@

Write-Host ""
Write-Host "Astra product agent" -ForegroundColor Cyan
Write-Host "Model: ministral-3:8b" -ForegroundColor DarkCyan
Write-Host "Model source: role-based default" -ForegroundColor DarkGray
Write-Host "Model note: Product value, roadmap prioritization, scope control." -ForegroundColor DarkGray
Write-Host "Permission mode: acceptEdits" -ForegroundColor DarkGray
Write-Host "Prompt file: astra-dev-council/agents/02_product_agent.md" -ForegroundColor Yellow
Write-Host "Role injection file: C:\Users\Simone\personal_ai\personal_ai\.astra-dev-council-launchers\role-product.md" -ForegroundColor DarkGray
Write-Host "Task file: C:\Users\Simone\personal_ai\personal_ai\.astra-dev-council-launchers\task-product.txt" -ForegroundColor DarkGray
Write-Host ""

$ollamaArgs = @("launch", "claude", "--model", "ministral-3:8b")

if ($true) {
  $ollamaArgs += "--yes"
}

$ollamaArgs += "--"

if ($false) {
  $ollamaArgs += "--append-system-prompt"
  $ollamaArgs += (Get-Content -LiteralPath "C:\Users\Simone\personal_ai\personal_ai\.astra-dev-council-launchers\role-product.md" -Raw)

  if ("acceptEdits" -ne "") {
    $ollamaArgs += "--permission-mode"
    $ollamaArgs += "acceptEdits"
  }
}
else {
  $ollamaArgs += "-p"
  $ollamaArgs += (Get-Content -LiteralPath "C:\Users\Simone\personal_ai\personal_ai\.astra-dev-council-launchers\task-product.txt" -Raw)
  $ollamaArgs += "--append-system-prompt"
  $ollamaArgs += (Get-Content -LiteralPath "C:\Users\Simone\personal_ai\personal_ai\.astra-dev-council-launchers\role-product.md" -Raw)

  if ("acceptEdits" -ne "") {
    $ollamaArgs += "--permission-mode"
    $ollamaArgs += "acceptEdits"
  }
}

& ollama @ollamaArgs
