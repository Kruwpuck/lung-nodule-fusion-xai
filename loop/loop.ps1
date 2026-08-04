<#
Driver loop eskalasi untuk .claude/Plan_escalation_loop.md, versi Windows.

PLAN aslinya mengandaikan VPS Ubuntu dengan tmux dan heredoc bash. Mesin ini
Windows tanpa tmux, dan sekaligus GPU host, jadi pemisahan VPS-otak dan
GPU-host di PLAN tidak berlaku di sini.

Driver ini sengaja bodoh. Ia tidak menilai kemajuan dan tidak membaca hasil
eksperimen. Ia hanya mesin status yang membaca STATE.json, memilih model dan
prompt sesuai level, memanggil claude sekali, lalu membaca STATE.json lagi.
Agent yang memutuskan naik level; driver hanya memaksa naik kalau agent tidak
memperbarui STATE.json sama sekali, yang berarti proses menggantung atau mati.

Pakai:
  powershell -ExecutionPolicy Bypass -File loop\loop.ps1
  powershell -ExecutionPolicy Bypass -File loop\loop.ps1 -DryRun
  powershell -ExecutionPolicy Bypass -File loop\loop.ps1 -MaxIter 5

Berhenti sendiri saat status jadi done atau blocked, saat level mencapai 6,
atau saat iterasi mencapai MaxIter. Ctrl+C aman: STATE.json selalu ditulis
utuh oleh Set-LoopState, tidak pernah sebagian.
#>

param(
    [string]$RunId   = "2026-08-04-run01",
    [int]$MaxIter    = 15,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root        = Split-Path -Parent $PSScriptRoot
$RunDir      = Join-Path $Root "runs\$RunId"
$Transcripts = Join-Path $RunDir "transcripts"
$EventsPath  = Join-Path $RunDir "events.jsonl"
$LedgerPath  = Join-Path $RunDir "ledger.md"
$StatePath   = Join-Path $Root "STATE.json"
$HandoffDir  = Join-Path $Root "handoff"

Set-Location $Root

function Get-NowIso {
    (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}

function Get-LoopState {
    Get-Content $StatePath -Raw | ConvertFrom-Json
}

function Set-LoopState([hashtable]$Patch) {
    $s = Get-LoopState
    foreach ($k in $Patch.Keys) {
        $s.$k = $Patch[$k]
    }
    $s.diperbarui = Get-NowIso
    ($s | ConvertTo-Json) | Set-Content -Path $StatePath -Encoding utf8
}

function Add-LedgerLine([int]$Level, [string]$Aksi, [string]$Hasil, [string]$Sha) {
    Add-Content -Path $LedgerPath -Encoding utf8 -Value ""
    Add-Content -Path $LedgerPath -Encoding utf8 -Value "## $(Get-NowIso) | lvl $Level | $Aksi | $Hasil | $Sha"
}

# Salin handoff yang ada ke transcripts sebelum iterasi berikutnya menimpanya.
# FASE 7 PLAN: setiap versi STUCK/BRIEF/RESEARCH adalah data penelitian.
function Save-HandoffSnapshot([int]$Iterasi) {
    foreach ($name in @("STUCK", "BRIEF", "RESEARCH", "BLOCKED")) {
        $src = Join-Path $HandoffDir "$name.md"
        if (Test-Path $src) {
            $dst = Join-Path $Transcripts "$Iterasi-$name.md"
            if (-not (Test-Path $dst)) {
                Copy-Item -Path $src -Destination $dst
            }
        }
    }
}

# --- pemeriksaan awal -------------------------------------------------------

if (-not (Test-Path $StatePath))  { throw "STATE.json tidak ada di $Root" }
if (-not (Test-Path $LedgerPath)) { throw "ledger tidak ada di $LedgerPath" }
if (-not (Test-Path (Join-Path $HandoffDir "GOAL.md"))) { throw "handoff/GOAL.md tidak ada" }
if (-not (Test-Path $Transcripts)) { New-Item -ItemType Directory -Path $Transcripts | Out-Null }

# PATH shell interaktif bisa basi kalau sesinya dibuka sebelum claude dipasang,
# jadi jangan hanya mengandalkan Get-Command.
$ClaudeExe = (Get-Command claude -ErrorAction SilentlyContinue).Source
if (-not $ClaudeExe) {
    $fallback = Join-Path $env:USERPROFILE ".local\bin\claude.exe"
    if (Test-Path $fallback) { $ClaudeExe = $fallback }
}
if (-not $ClaudeExe) { throw "claude CLI tidak ketemu di PATH maupun di ~\.local\bin" }
Write-Host "claude : $ClaudeExe"

# Kode yang kotor membuat "commit hijau terakhir" jadi patokan yang tidak
# berarti, dan membuat agent bisa ikut meng-commit perubahan orang lain.
#
# Pembukuan loop sendiri dikecualikan: STATE.json, ledger, handoff, dan
# artifacts justru berubah sebagai bagian kerja normal, dan menahannya berarti
# loop tidak pernah bisa dijalankan dua kali.
$bookkeeping = '^\s*.. (STATE\.json|runs/|handoff/|research/|artifacts/)'
$dirty = git status --porcelain | Where-Object { $_ -notmatch $bookkeeping }
if ($dirty -and -not $DryRun) {
    Write-Host "Kode kotor. Commit atau stash dulu sebelum loop mulai:" -ForegroundColor Yellow
    $dirty | ForEach-Object { Write-Host "  $_" }
    throw "batal: pohon kerja tidak bersih"
}

Write-Host "run    : $RunId"
Write-Host "root   : $Root"
Write-Host "maxiter: $MaxIter"

# --- loop utama -------------------------------------------------------------

while ($true) {
    $st    = Get-LoopState
    $lvl   = [int]$st.level
    $iter  = [int]$st.iterasi

    if ($st.status -ne "running") {
        Write-Host "status = $($st.status). berhenti." -ForegroundColor Cyan
        break
    }

    if ($iter -ge $MaxIter) {
        Add-LedgerLine $lvl "BLOCKED" "MAX_ITER $MaxIter tercapai tanpa memenuhi kriteria selesai" "-"
        Set-LoopState @{ status = "blocked"; menunggu_manusia = $true }
        Write-Host "MAX_ITER $MaxIter tercapai. status = blocked." -ForegroundColor Yellow
        Write-Host "Baca $LedgerPath lalu putuskan: naikkan MaxIter, atau ubah handoff/GOAL.md."
        break
    }

    if ($lvl -ge 6) {
        Add-LedgerLine $lvl "BLOCKED" "level 6 tercapai, loop berhenti sesuai FASE 5" "-"
        Set-LoopState @{ status = "blocked"; menunggu_manusia = $true }
        Write-Host "Level 6. Loop berhenti. Ini keluaran yang benar, bukan kegagalan." -ForegroundColor Yellow
        break
    }

    if ($lvl -le 3) {
        $model  = "sonnet"
        $effort = "low"
        $promptFile = "loop\prompt_exec.txt"
        $tools  = "Read,Write,Edit,Bash,WebSearch,Skill"
        $aksi   = "EXEC"
    }
    elseif ($lvl -eq 4) {
        $model  = "opus"
        $effort = "medium"
        $promptFile = "loop\prompt_lvl4.txt"
        $tools  = "Read,Write,Bash,WebSearch,WebFetch,Skill"
        $aksi   = "BRIEF"
    }
    else {
        $model  = "opus"
        $effort = "medium"
        $promptFile = "loop\prompt_lvl5.txt"
        $tools  = "Read,Write,Bash,WebSearch,WebFetch,Task,Skill"
        $aksi   = "RESEARCH"
    }

    Write-Host ""
    Write-Host "--- iterasi $iter | lvl $lvl | $model effort=$effort | $aksi ---" -ForegroundColor Green

    if ($DryRun) {
        Write-Host "DryRun: claude -p (isi $promptFile) --model $model --effort $effort --allowedTools $tools"
        break
    }

    Save-HandoffSnapshot $iter
    $stampBefore = $st.diperbarui
    $prompt = Get-Content (Join-Path $Root $promptFile) -Raw

    & $ClaudeExe -p $prompt `
        --model $model `
        --effort $effort `
        --allowedTools $tools `
        --permission-mode acceptEdits `
        --output-format stream-json `
        --verbose | Tee-Object -FilePath $EventsPath -Append

    $after = Get-LoopState

    # Agent tidak menyentuh STATE.json sama sekali. Berarti ia mati, kehabisan
    # giliran, atau mengabaikan instruksi. Perlakukan sebagai kegagalan dan
    # naikkan level, kalau tidak loop akan mengulang iterasi yang sama
    # selamanya dengan model dan prompt yang sama.
    if ($after.diperbarui -eq $stampBefore) {
        $newLvl = $lvl + 1
        Add-LedgerLine $lvl $aksi "agent keluar tanpa memperbarui STATE.json; level dinaikkan paksa ke $newLvl" "-"
        Set-LoopState @{ iterasi = $iter + 1; level = $newLvl; gagal_beruntun = [int]$after.gagal_beruntun + 1 }
        Write-Host "STATE.json tidak berubah. Level dinaikkan paksa ke $newLvl." -ForegroundColor Yellow
        continue
    }

    if (Test-Path (Join-Path $HandoffDir "BLOCKED.md")) {
        Set-LoopState @{ status = "blocked"; menunggu_manusia = $true }
        Write-Host ""
        Write-Host "BLOCKED.md ditulis. Loop berhenti total." -ForegroundColor Magenta
        Write-Host "Gerbang G2: buka handoff/BLOCKED.md, salin blok PROMPT SIAP TEMPEL ke"
        Write-Host "claude.ai dengan Research aktif, simpan jawabannya ke handoff/RESEARCH.md,"
        Write-Host "arsipkan ke research/$RunId-<slug>.md, setel level 1 dan gagal_beruntun 0"
        Write-Host "di STATE.json, lalu jalankan loop ini lagi."
        [console]::beep(800, 400)
        break
    }

    $now = Get-LoopState
    Write-Host "-> iterasi $($now.iterasi) | lvl $($now.level) | gagal $($now.gagal_beruntun) | status $($now.status)" -ForegroundColor DarkGray
}

$final = Get-LoopState
Write-Host ""
Write-Host "selesai. status=$($final.status) iterasi=$($final.iterasi) level=$($final.level)"
if ($final.status -eq "done") {
    Write-Host "Gerbang G4: status done ditulis agent, bukan bukti. Verifikasi sendiri kelima"
    Write-Host "kriteria di handoff/GOAL.md sebelum mempercayainya." -ForegroundColor Yellow
}
