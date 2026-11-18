[CmdletBinding()]
param(
    [switch]$DownloadGallery,
    [ValidateRange(1, 64)][int]$BankBatchSize = 1
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

function Invoke-Checked {
    param([string]$File, [string[]]$Arguments)
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$File exited with code $LASTEXITCODE."
    }
}

function Assert-Hash {
    param([string]$Path, [string]$Hash)
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Hash) {
        throw "SHA-256 mismatch: $Path. Move the damaged file away and run setup again."
    }
}

function Get-Download {
    param([string]$Uri, [string]$Path, [string]$Hash)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        $partial = "$Path.part"
        New-Item -ItemType Directory -Force (Split-Path -Parent $Path) | Out-Null
        Write-Host "Downloading $(Split-Path -Leaf $Path)"
        Invoke-Checked 'curl.exe' @('--fail', '--location', '--retry', '2',
            '--show-error', '--output', $partial, $Uri)
        Assert-Hash $partial $Hash
        Move-Item -LiteralPath $partial -Destination $Path
    }
    else {
        Assert-Hash $Path $Hash
    }
}

function Install-Alignment {
    param([string]$Runtime, [string]$Python)
    $mamba = Join-Path $Runtime 'Library\bin\micromamba.exe'
    $prefix = Join-Path $Runtime 'align'
    $alignmentPython = Join-Path $prefix 'python.exe'
    if (-not (Test-Path -LiteralPath $alignmentPython -PathType Leaf)) {
        $archive = Join-Path $Runtime 'micromamba.tar.bz2'
        if (-not (Test-Path -LiteralPath $mamba -PathType Leaf)) {
            Get-Download 'https://micro.mamba.pm/api/micromamba/win-64/2.10.0' $archive `
                '6283f25f8142606d86e81a43c7103b5d4f1d5c2027e3f9d7326eff23b33930d8'
            Invoke-Checked 'tar.exe' @('-xf', $archive, '-C', $Runtime, 'Library/bin/micromamba.exe')
        }
        Invoke-Checked $mamba @('create', '--yes', '--root-prefix', (Join-Path $Runtime 'mamba'),
            '--prefix', $prefix, '--override-channels', '-c', 'conda-forge',
            'python=3.11', 'dlib=20.0.1=cpu_py311*', 'libblas=*=*openblas', 'pillow', 'scipy')
    }

    $previousPath = $env:Path
    try {
        $env:Path = "$(Join-Path $prefix 'Library\bin');$prefix;$previousPath"
        Invoke-Checked $alignmentPython @('-c', 'import dlib, PIL, scipy')
        Write-Host 'Face alignment imports OK'
    }
    finally {
        $env:Path = $previousPath
    }

    $predictor = 'checkpoints\shape_predictor_68_face_landmarks.dat'
    $hash = 'fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f'
    if (-not (Test-Path -LiteralPath $predictor -PathType Leaf)) {
        $compressed = "$predictor.bz2"
        $pending = "$predictor.part"
        Get-Download 'https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2' $compressed `
            '7d6637b8f34ddb0c1363e09a4628acb34314019ec3566fd66b80c04dda6980f5'
        Invoke-Checked $Python @('-c', 'import bz2, shutil, sys; shutil.copyfileobj(bz2.open(sys.argv[1], ''rb''), open(sys.argv[2], ''wb''))', $compressed, $pending)
        Assert-Hash $pending $hash
        Move-Item -LiteralPath $pending -Destination $predictor
    }
    else {
        Assert-Hash $predictor $hash
    }
}

$previousTorchHome = $env:TORCH_HOME
Push-Location -LiteralPath $root
try {
    $bank = 'data\banks\vggface2_hq_4605.h5'
    $gallery = 'data\VGGFace2-HQ\aligned'
    if (-not (Test-Path -LiteralPath $bank -PathType Leaf) -and
        -not (Test-Path -LiteralPath $gallery -PathType Container) -and -not $DownloadGallery) {
        throw "Copy the existing bank to $bank, prepare $gallery, or run setup-windows.cmd -DownloadGallery. See README-LAPTOP.ko.md."
    }

    $python = Join-Path $root '.venv\Scripts\python.exe'
    $runtime = Join-Path $root '.runtime'
    $env:TORCH_HOME = Join-Path $runtime 'torch'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        $command = Get-Command python.exe -ErrorAction SilentlyContinue
        $probe = 'import sys; print(sys.version_info[:2] == (3, 11) and sys.maxsize > 2**32)'
        if ($command -and $command.Source -notlike '*\WindowsApps\*' -and
            (& $command.Source -c $probe) -eq 'True') {
            Invoke-Checked $command.Source @('-m', 'venv', '.venv')
        }
        else {
            Invoke-Checked 'py.exe' @('-3.11', '-m', 'venv', '.venv')
        }
    }

    Invoke-Checked $python @('-m', 'pip', 'install', '--index-url',
        'https://download.pytorch.org/whl/cu128', 'torch==2.7.1+cu128', 'torchvision==0.22.1+cu128')
    Invoke-Checked $python @('-m', 'pip', 'install', '-r', 'requirements-wsl.txt')
    Invoke-Checked $python @('scripts/check_env.py')

    $stylegan = 'checkpoints\ffhq.pkl'
    $e4e = 'checkpoints\e4e.pt'
    $arcface = 'checkpoints\arcface.pth'
    $farl = 'checkpoints\FaRL.pth'
    Get-Download 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl' $stylegan `
        'a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c'
    Get-Download 'https://drive.usercontent.google.com/download?id=1cUv_reLE6k3604or78EranS7XzuVMWeO&export=download&confirm=t' $e4e `
        '2ace1d9a8c05c10a399bcd500b8dda118f759ff1aac89dbdab7435f2136a0999'
    Get-Download 'https://drive.usercontent.google.com/download?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn&export=download&confirm=t' $arcface `
        'a035c768259b98ab1ce0e646312f48b9e1e218197a0f80ac6765e88f8b6ddf28'
    Get-Download 'https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth' $farl `
        'ab3534c4c2377a853926b99456fc4a10140ea092213b9c8d32458e06ec0961b7'
    Get-Download 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth' `
        (Join-Path $env:TORCH_HOME 'hub\checkpoints\alexnet-owt-7be5be79.pth') `
        '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'
    Install-Alignment $runtime $python

    $encoders = @('--e4e', $e4e, '--arcface', $arcface, '--farl', $farl)
    if (-not (Test-Path -LiteralPath $bank -PathType Leaf)) {
        if ($DownloadGallery) {
            Invoke-Checked $python @('-u', 'scripts/prepare_vggface2hq.py', '--output',
                $gallery, '--limit', '4605')
        }
        Invoke-Checked $python (@('-u', 'scripts/build_feature_bank.py', '--images', $gallery,
            '--output', $bank, '--max-identities', '4605', '--batch-size', "$BankBatchSize",
            '--device', 'cuda') + $encoders)
    }
    Invoke-Checked $python (@('-u', 'scripts/validate_inputs.py', '--stylegan', $stylegan,
        '--feature-bank', $bank, '--expected-identities', '4605', '--lpips',
        '--trusted-inputs', '--device', 'cuda') + $encoders)
    Write-Host 'Setup complete. Run start-windows.cmd, then open http://127.0.0.1:8000/.'
}
finally {
    $env:TORCH_HOME = $previousTorchHome
    Pop-Location
}
