# Start Backend Server for Online Proctoring System
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Online Proctoring - Backend Server" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if virtual environment exists
if (-Not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "⚠️  Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✅ Virtual environment created`n" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Check if requirements are installed
Write-Host "📦 Checking dependencies..." -ForegroundColor Yellow
$pipList = pip list

if ($pipList -notmatch "flask") {
    Write-Host "⚠️  Dependencies not installed. Installing with correct order..." -ForegroundColor Yellow
    
    # Install protobuf FIRST to avoid version conflicts
    pip install protobuf==3.20.3
    
    # Then install everything else
    pip install -r requirements.txt
    Write-Host "✅ Dependencies installed`n" -ForegroundColor Green
} else {
    # Check for protobuf version issues
    $protobufVersion = pip show protobuf 2>&1 | Select-String "Version:"
    if ($protobufVersion -match "5\.") {
        Write-Host "⚠️  Protobuf version conflict detected. Fixing..." -ForegroundColor Yellow
        pip uninstall -y protobuf tensorflow mediapipe keras-facenet
        pip install protobuf==3.20.3
        pip install tensorflow==2.15.0 keras-facenet==0.3.2 mediapipe==0.10.9
        Write-Host "✅ Protobuf conflict resolved`n" -ForegroundColor Green
    } else {
        Write-Host "✅ Dependencies verified`n" -ForegroundColor Green
    }
}

# Start server
Write-Host "🚀 Starting backend server..." -ForegroundColor Green
Write-Host "   Server will run on http://localhost:5000" -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop`n" -ForegroundColor Yellow
python server.py
