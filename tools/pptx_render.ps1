# Render a PPTX with PowerPoint itself: one PNG per slide (+ optional PDF).
#
# usage: powershell -ExecutionPolicy Bypass -File pptx_render.ps1 -Pptx <file.pptx> -OutDir <dir> [-Pdf]
param(
  [Parameter(Mandatory = $true)][string]$Pptx,
  [Parameter(Mandatory = $true)][string]$OutDir,
  [switch]$Pdf
)
# relative paths are relative to this script's folder; no wildcard expansion ([...] in folder names)
function Full([string]$p) {
  if ([System.IO.Path]::IsPathRooted($p)) { return [System.IO.Path]::GetFullPath($p) }
  return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $p))
}
$Pptx = Full $Pptx
$OutDir = Full $OutDir
[System.IO.Directory]::CreateDirectory($OutDir) | Out-Null
Get-ChildItem -LiteralPath $OutDir -Filter 'p*.png' | Remove-Item -Force

$app = New-Object -ComObject PowerPoint.Application
try {
  # ReadOnly, Untitled, WithWindow=false
  $pres = $app.Presentations.Open($Pptx, $true, $false, $false)
  $i = 0
  foreach ($s in $pres.Slides) {
    $i++
    $s.Export((Join-Path $OutDir ('p{0:D2}.png' -f $i)), 'PNG', 1280, 720)
  }
  if ($Pdf) {
    $pdfPath = [System.IO.Path]::ChangeExtension($Pptx, '.pdf')
    $pres.SaveAs($pdfPath, 32)   # ppSaveAsPDF
    Write-Output "pdf: $pdfPath"
  }
  $pres.Close()
  Write-Output "exported $i slides to $OutDir"
}
finally {
  $app.Quit()
  [System.Runtime.Interopservices.Marshal]::ReleaseComObject($app) | Out-Null
}
