param(
    [switch]$CheckLibraries,
    [switch]$ProbeOnly,
    [switch]$Trace,
    [switch]$SparseApiOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$localRoot = Join-Path $projectRoot '.local-zluda'
$runtime = Get-Content -LiteralPath (Join-Path $localRoot 'runtime.json') -Raw | ConvertFrom-Json
$savedEnvironment = @{}
$variables = @('HIP_PATH', 'PATH', 'HIP_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING', 'ZLUDA_LOG_DIR')
foreach ($name in $variables) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    $env:HIP_PATH = $runtime.hip_path
    $env:PATH = "$($runtime.hip_path)\bin;$env:PATH"
    # hipInfo on this machine: device 0 is gfx1036 integrated graphics;
    # device 1 is the RX 7800 XT (gfx1101), which this SDK targets.
    $env:HIP_VISIBLE_DEVICES = '1'
    $launcherArgs = @('--')
    if ($Trace) {
        $env:CUDA_LAUNCH_BLOCKING = '1'
        $env:ZLUDA_LOG_DIR = Join-Path $projectRoot 'artifacts\zluda\trace'
        $launcherArgs = @('--zluda-trace', '--')
    }
    $launcher = Join-Path $localRoot 'zluda\zluda.exe'
    if ($CheckLibraries) {
        & $launcher @launcherArgs (Join-Path $localRoot 'zluda\cuda_check.exe')
    } else {
        $python = Join-Path $projectRoot '.venv-zluda\Scripts\python.exe'
        $validation = Join-Path $PSScriptRoot 'validate_zluda.py'
        $validationArgs = @('-u', $validation, '--device', 'cuda')
        if ($ProbeOnly) { $validationArgs += '--probe-only' }
        if ($SparseApiOnly) {
            $validationArgs = @('-u', (Join-Path $PSScriptRoot 'probe_zluda_sparse_api.py'))
        }
        & $launcher @launcherArgs $python @validationArgs
    }
    $result = $LASTEXITCODE
} finally {
    foreach ($name in $variables) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
}
exit $result
