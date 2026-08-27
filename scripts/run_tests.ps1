<#
.SYNOPSIS
  Native Windows twin of scripts/run_tests.sh — hermetic, per-file parallel
  test runner with NO bash / Git Bash required.

.DESCRIPTION
  Runs scripts/run_tests_parallel.py under the project venv with the same
  CI-parity environment as run_tests.sh:
    * Per-file isolation (each test file = one fresh `python -m pytest <file>`)
    * TZ=UTC, LANG=C.UTF-8, PYTHONHASHSEED=0 (deterministic)
    * Hermetic env — only the variables listed below are passed to the
      test subprocesses, so no credential / API-key env vars leak in.
    * Auto-detects the venv interpreter (`.venv`, `venv`, then
      `$HOME\.argus\argus-agent\venv`) and fails fast with the install
      command when pytest (a [dev] extra) is missing.

.EXAMPLE
  pwsh scripts/run_tests.ps1                     # full suite
  pwsh scripts/run_tests.ps1 -j 4                # cap parallelism
  pwsh scripts/run_tests.ps1 tests/agent/        # discover only here
  pwsh scripts/run_tests.ps1 tests/foo.py -- --tb=long
#>
# NOTE: no [CmdletBinding()]/param() on purpose — every positional argument
# must flow through $args straight into run_tests_parallel.py (including
# pytest flags like -q / -k / --tb).
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

# ── Locate the project venv (accept .venv and venv layouts) ─────────────────
$python = $null
foreach ($candidate in @(
    (Join-Path $repoRoot ".venv"),
    (Join-Path $repoRoot "venv"),
    (Join-Path $env:USERPROFILE ".argus\argus-agent\venv"))) {
  $py = Join-Path $candidate "Scripts\python.exe"
  if (Test-Path $py) { $python = $py; break }
  $py = Join-Path $candidate "bin\python"
  if (Test-Path $py) { $python = $py; break }
}
if (-not $python) {
  Write-Error @"
no virtualenv found in $repoRoot\.venv or $repoRoot\venv
  create one first with:  uv sync --extra all --extra dev
"@
}

# ── Dev/test dependencies present? ──────────────────────────────────────────
# pytest is a [dev] extra, not a core dependency, so a bare `uv sync` won't
# have it. Fail fast with the exact fix instead of a confusing import error
# deep inside the parallel runner.
& $python -c "import pytest" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Error @"
pytest is not installed in $python
  install the dev/test extras:
    uv sync --extra all --extra dev
  or: uv pip install -e ".[all,dev]"
"@
}

# ── Hermetic env: only what CI needs (no credential vars leak) ─────────────
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $python
$psi.WorkingDirectory = $repoRoot
$psi.UseShellExecute = $false
$psi.Environment.Clear()
$psi.Environment.Add("PATH", $env:PATH)
if ($env:USERPROFILE) { $psi.Environment.Add("HOME", $env:USERPROFILE) }
# Windows os.path.expanduser IGNORES HOME — it needs USERPROFILE (or
# HOMEDRIVE+HOMEPATH). Without them, `Path("~").expanduser()` raises
# "Could not determine home directory". Keep all three.
if ($env:USERPROFILE) { $psi.Environment.Add("USERPROFILE", $env:USERPROFILE) }
if ($env:HOMEDRIVE)   { $psi.Environment.Add("HOMEDRIVE", $env:HOMEDRIVE) }
if ($env:HOMEPATH)    { $psi.Environment.Add("HOMEPATH", $env:HOMEPATH) }
if ($env:SystemRoot)   { $psi.Environment.Add("SystemRoot", $env:SystemRoot) }
if ($env:TEMP)         { $psi.Environment.Add("TEMP", $env:TEMP) }
if ($env:TMP)          { $psi.Environment.Add("TMP", $env:TMP) }
if ($env:PATHEXT)      { $psi.Environment.Add("PATHEXT", $env:PATHEXT) }
if ($env:COMSPEC)      { $psi.Environment.Add("COMSPEC", $env:COMSPEC) }
$psi.Environment.Add("TZ", "UTC")
$psi.Environment.Add("LANG", "C.UTF-8")
$psi.Environment.Add("LC_ALL", "C.UTF-8")
# Windows Python ignores LANG/LC_ALL (uses the console code page, e.g. GBK),
# which crashes on non-ASCII pytest output (✓/✗). Force UTF-8 explicitly.
$psi.Environment.Add("PYTHONUTF8", "1")
$psi.Environment.Add("PYTHONHASHSEED", "0")
$psi.Environment.Add("PYTHONDONTWRITEBYTECODE", "1")
if ($env:ARGUS_RUN_SLOW_PET_TESTS) {
  $psi.Environment.Add("ARGUS_RUN_SLOW_PET_TESTS", $env:ARGUS_RUN_SLOW_PET_TESTS)
}
$liveGuard = Join-Path $env:USERPROFILE ".argus\pytest_live_guard.py"
if (Test-Path $liveGuard) {
  $psi.Environment.Add("PYTHONPATH", (Join-Path $env:USERPROFILE ".argus"))
  $psi.Environment.Add("PYTEST_PLUGINS", "pytest_live_guard")
}

# Runner + passthrough args (quote args that contain spaces).
$runner = Join-Path $PSScriptRoot "run_tests_parallel.py"
$argParts = @('"' + $runner + '"')
foreach ($a in $args) {
  if ($a -match '\s') { $argParts += ('"' + $a + '"') } else { $argParts += $a }
}
$psi.Arguments = $argParts -join ' '

Write-Host "▶ running per-file parallel test suite via run_tests_parallel.py"
Write-Host "  (TZ=UTC LANG=C.UTF-8 PYTHONHASHSEED=0; clean env)"

$proc = [System.Diagnostics.Process]::Start($psi)
$proc.WaitForExit()
exit $proc.ExitCode
