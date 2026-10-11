# Copia el código del Avance 2 sobre tu clon de Endo-STTN.
# Uso (desde PowerShell):
#   powershell -ExecutionPolicy Bypass -File "D:\Git\Equipo19\Avance2_DeepEstimator_EndoSTTN\instalar.ps1"
# Otra carpeta de Endo-STTN:
#   powershell -ExecutionPolicy Bypass -File ...\instalar.ps1 -EndoSttn "E:\Endo-STTN"
param([string]$EndoSttn = "D:\Git\Endo-STTN")
$ErrorActionPreference = "Stop"
$aqui = $PSScriptRoot

if (-not (Test-Path (Join-Path $EndoSttn "train.py"))) {
    throw "No encuentro Endo-STTN en '$EndoSttn'. Clónalo con: git clone https://github.com/endomapper/Endo-STTN.git $EndoSttn"
}

# Respaldo del trainer original de Endo-STTN (solo si todavía es el original)
$trainer = Join-Path $EndoSttn "core\trainer.py"
$respaldo = Join-Path $EndoSttn "core\trainer_original.py"
if (-not (Test-Path $respaldo) -and -not (Select-String -Path $trainer -Pattern "depth_loss" -Quiet)) {
    Copy-Item $trainer $respaldo
    Write-Host "Respaldo: core\trainer_original.py"
}

New-Item -ItemType Directory -Force (Join-Path $EndoSttn "configs") | Out-Null
Copy-Item (Join-Path $aqui "endo_sttn\core\*") (Join-Path $EndoSttn "core") -Force
Copy-Item (Join-Path $aqui "endo_sttn\configs\*") (Join-Path $EndoSttn "configs") -Force
Copy-Item (Join-Path $aqui "scripts\*") $EndoSttn -Force
Copy-Item (Join-Path $aqui "notebooks\Equipo19_EndoSTTN_Profundidad_local.ipynb") $EndoSttn -Force

Write-Host ""
Write-Host "Copiado a $EndoSttn :"
Get-ChildItem (Join-Path $aqui "endo_sttn\core"), (Join-Path $aqui "endo_sttn\configs") -File |
    ForEach-Object { Write-Host "  $($_.Directory.Name)\$($_.Name)" }
Get-ChildItem (Join-Path $aqui "scripts") -File | ForEach-Object { Write-Host "  $($_.Name)" }
Write-Host "  Equipo19_EndoSTTN_Profundidad_local.ipynb"
Write-Host ""
Write-Host "Siguiente paso:  cd $EndoSttn ; python test_depth_loss.py"
