# Complete Fix for Backend Import Errors
# Fixes: Protobuf errors, MediaPipe import errors, TensorFlow issues

Write-Host "`n🔧 Fixing All Backend Dependencies...`n" -ForegroundColor Cyan

# Step 1: Uninstall conflicting packages
Write-Host "Step 1/5: Removing all conflicting packages..." -ForegroundColor Yellow
pip uninstall -y protobuf tensorflow mediapipe keras-facenet 2>&1 | Out-Null
Write-Host "  ✓ Conflicts removed" -ForegroundColor Green

# Step 2: Install protobuf first (CRITICAL - must be first!)
Write-Host "Step 2/5: Installing protobuf 3.20.3..." -ForegroundColor Yellow
pip install protobuf==3.20.3 --quiet
Write-Host "  ✓ Protobuf installed" -ForegroundColor Green

# Step 3: Install TensorFlow
Write-Host "Step 3/5: Installing TensorFlow 2.15.0..." -ForegroundColor Yellow
pip install tensorflow==2.15.0 --quiet
Write-Host "  ✓ TensorFlow installed" -ForegroundColor Green

# Step 4: Install FaceNet and MediaPipe
Write-Host "Step 4/5: Installing FaceNet and MediaPipe..." -ForegroundColor Yellow
pip install keras-facenet==0.3.2 --quiet
pip install mediapipe==0.10.9 --quiet
Write-Host "  ✓ AI models installed" -ForegroundColor Green

# Step 5: Verify installation
Write-Host "Step 5/5: Verifying installation..." -ForegroundColor Yellow
$protobufCheck = pip show protobuf 2>&1 | Select-String "Version: 3.20"
$tensorflowCheck = pip show tensorflow 2>&1 | Select-String "Version: 2.15"
$mediapipeCheck = pip show mediapipe 2>&1 | Select-String "Version: 0.10"

if ($protobufCheck -and $tensorflowCheck -and $mediapipeCheck) {
    Write-Host "  ✓ All packages verified`n" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "✅ Fix Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "`nNow run: " -NoNewline -ForegroundColor White
    Write-Host "python server.py" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "  ⚠ Verification failed. Check manually.`n" -ForegroundColor Red
    Write-Host "Run: pip list | Select-String 'protobuf|tensorflow|mediapipe'" -ForegroundColor Yellow
}
