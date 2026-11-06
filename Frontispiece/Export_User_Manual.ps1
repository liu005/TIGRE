param()

$ErrorActionPreference = 'Stop'
$source = Join-Path $PSScriptRoot 'TIGRE_User_Manual.md'
$pdf = Join-Path $PSScriptRoot 'TIGRE_User_Manual.pdf'
$html = Join-Path $PSScriptRoot 'TIGRE_User_Manual.html'
$manualText = Get-Content -LiteralPath $source -Raw
$version = [regex]::Match($manualText, '\*\*Manual version: ([^*]+)\*\*').Groups[1].Value
$updated = [regex]::Match($manualText, '\*\*Updated: ([^*]+)\*\*').Groups[1].Value
if (-not $version -or -not $updated) { throw "Missing version or date in $source" }

Push-Location $PSScriptRoot
try {
    & pandoc $source '--from=gfm' '--standalone' '--toc' '--toc-depth=3' `
        '--metadata=title:TIGRE User Manual' '-o' $html
    if ($LASTEXITCODE -ne 0) { throw 'HTML manual conversion failed' }

    & pandoc $source '--from=gfm' '--pdf-engine=xelatex' '--toc' '--toc-depth=3' `
        '--number-sections' '--metadata=title:TIGRE User Manual' `
        ('--metadata=date:Manual version ' + $version + ' - ' + $updated) `
        ('--lua-filter=' + (Join-Path $PSScriptRoot 'manual-tables.lua')) `
        ('--include-in-header=' + (Join-Path $PSScriptRoot 'manual-header.tex')) `
        '-V' 'papersize=a4' '-V' 'geometry:margin=23mm' '-V' 'fontsize=11pt' `
        '-V' 'colorlinks=true' '-o' $pdf
    if ($LASTEXITCODE -ne 0) { throw 'PDF manual conversion failed' }
    Write-Output "Created $html and $pdf"
}
finally {
    Pop-Location
}
