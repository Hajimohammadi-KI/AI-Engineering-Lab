Write-Host "=== Python Virtual Environment Builder ===" -ForegroundColor Cyan

$projectPath = Read-Host "Enter the FULL path where the environment should be created (e.g. C:\Users\Elahe\Desktop\MyProject)"

# Trim spaces
$projectPath = $projectPath.Trim()

# Create folder if missing
if (!(Test-Path $projectPath)) {
    Write-Host "Path does not exist. Creating it..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $projectPath | Out-Null
}

Set-Location $projectPath

$envName = Read-Host "Enter the virtual environment name (e.g. venv)"
$envName = $envName.Trim()

Write-Host "Creating virtual environment '$envName' ..." -ForegroundColor Yellow
python -m venv $envName

$activatePath = Join-Path $projectPath "$envName\Scripts\Activate.ps1"
& $activatePath

python -m pip install --upgrade pip

Write-Host "✅ Virtual environment '$envName' created successfully!" -ForegroundColor Green
Write-Host "Location: $projectPath\$envName"
