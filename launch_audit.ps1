$base = "C:\Users\Adaptive Network\Documents\Lung Cancer\lung-nodule-fusion-xai"
$p = Start-Process -FilePath "$base\.venv\Scripts\python.exe" -ArgumentList "_audit_missing.py" `
    -WorkingDirectory $base `
    -RedirectStandardOutput "$base\_audit_out.log" `
    -RedirectStandardError "$base\_audit_err.log" `
    -NoNewWindow -PassThru
Write-Output $p.Id
