# Inicia Efecto Imán sólo si el dashboard local no está respondiendo.
$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
$python = Join-Path $project '.venv\Scripts\python.exe'
$data = Join-Path $project 'data'
$health = 'http://127.0.0.1:8501/_stcore/health'

New-Item -ItemType Directory -Path $data -Force | Out-Null
$startupLog = Join-Path $data 'streamlit_startup.log'

function Write-StartupLog([string] $message) {
    Add-Content -LiteralPath $startupLog -Value "$(Get-Date -Format o) $message"
}

function Test-Dashboard {
    try {
        $response = Invoke-WebRequest -Uri $health -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

if (Test-Dashboard) {
    Write-StartupLog 'Streamlit ya estaba activo en 127.0.0.1:8501.'
    exit 0
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-StartupLog "No se encontró el entorno virtual: $python"
    exit 1
}

try {
    $process = Start-Process -FilePath $python -ArgumentList @(
        '-m', 'streamlit', 'run', 'app.py',
        '--server.address', '127.0.0.1',
        '--server.port', '8501',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ) -WorkingDirectory $project -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput (Join-Path $data 'streamlit_autostart.out.log') `
      -RedirectStandardError (Join-Path $data 'streamlit_autostart.err.log')
    Write-StartupLog "Iniciando Streamlit (PID $($process.Id))."

    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if (Test-Dashboard) {
            Write-StartupLog 'Dashboard disponible en http://127.0.0.1:8501/'
            exit 0
        }
        if ($process.HasExited) {
            Write-StartupLog "Streamlit terminó antes de responder (código $($process.ExitCode))."
            exit 1
        }
        Start-Sleep -Seconds 1
    }
    Write-StartupLog 'Streamlit no respondió en 60 segundos; revisar streamlit_autostart.err.log.'
    exit 1
} catch {
    Write-StartupLog "Error al iniciar Streamlit: $($_.Exception.Message)"
    exit 1
}
