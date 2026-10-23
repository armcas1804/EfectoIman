# Programa el arranque local al iniciar sesión y de lunes a viernes a las 06:15.
$ErrorActionPreference = 'Stop'
$taskName = 'EfectoIman Streamlit'
$script = Join-Path $PSScriptRoot 'start_efecto_iman.ps1'
$powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name

if (-not (Test-Path -LiteralPath $script -PathType Leaf)) {
    throw "No se encontró $script"
}

$action = New-ScheduledTaskAction -Execute $powershell `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`"" `
    -WorkingDirectory $PSScriptRoot
$weekday = New-ScheduledTaskTrigger -Weekly -WeeksInterval 1 `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At '06:15'
$logon = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 2)

Register-ScheduledTask -TaskName $taskName -Action $action `
    -Trigger @($logon, $weekday) -Principal $principal `
    -Settings $settings -Force | Out-Null
Write-Output "Tarea '$taskName' instalada para $user. Próximo arranque laborable: 06:15 (hora de Windows)."
