$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# ── Load .env if present (sets GUARDIAN_DETECTOR and other vars) ────────────
if (Test-Path -LiteralPath '.env') {
    Get-Content '.env' | Where-Object { $_ -match '^\s*[^#]\S*=' } | ForEach-Object {
        $parts = $_ -split '=', 2
        [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), 'Process')
    }
    Write-Host 'Loaded .env'
}

# Start the local PostgreSQL server if the project is configured for it and it is not running yet.
if ($env:GUARDIAN_DB -eq 'postgres' -and $env:GUARDIAN_PG_HOME -and (Test-Path -LiteralPath "$env:GUARDIAN_PG_HOME\pgsql\bin\pg_ctl.exe")) {
    $pgPort = if ($env:GUARDIAN_DB_PORT) { [int]$env:GUARDIAN_DB_PORT } else { 5432 }
    if (-not (Get-NetTCPConnection -LocalPort $pgPort -State Listen -ErrorAction SilentlyContinue)) {
        Write-Host 'Starting PostgreSQL...'
        # No -Wait: pg_ctl leaves the database server running as its child.
        Start-Process -FilePath "$env:GUARDIAN_PG_HOME\pgsql\bin\pg_ctl.exe" -ArgumentList '-D', "`"$env:GUARDIAN_PG_HOME\data`"", '-l', "`"$env:GUARDIAN_PG_HOME\postgres.log`"", 'start' -WindowStyle Hidden
        for ($i = 0; $i -lt 30 -and -not (Get-NetTCPConnection -LocalPort $pgPort -State Listen -ErrorAction SilentlyContinue); $i++) { Start-Sleep -Seconds 1 }
    }
}

if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed. Install Python first.' }
}
$guardianPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'

# Install AI/ML requirements (torch, ultralytics, opencv, etc.)
& $guardianPython -m pip install -r requirements-ai.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

& $guardianPython manage.py migrate
if ($LASTEXITCODE -ne 0) { throw 'Database migration failed.' }
& $guardianPython manage.py seed_demo
Write-Host 'First run: in another terminal run .venv/Scripts/python manage.py createsuperuser'
Write-Host 'Open http://127.0.0.1:8000 after creating your account.'
$env:DJANGO_DEBUG = '1'
& $guardianPython manage.py runserver 127.0.0.1:8000 --noreload
