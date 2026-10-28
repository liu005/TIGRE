# Generate both manuals with wrapping tables and code blocks. Requires Pandoc/XeLaTeX.
$ErrorActionPreference = 'Stop'
foreach ($manual in @('TIGRE_CT_Simulation_Builder_Python', 'TIGRE_CT_Simulation_Builder_MATLAB')) {
    $edition = if ($manual -eq 'TIGRE_CT_Simulation_Builder_Python') { 'Python User Manual' } else { 'MATLAB User Manual' }
    $source = Join-Path $PSScriptRoot ($manual + '.md')
    $target = Join-Path $PSScriptRoot ($manual + '.pdf')
    $arguments = @(
        $source, '--from=gfm', '--pdf-engine=xelatex', '--toc', '--toc-depth=2', '--number-sections',
        '--metadata=title:TIGRE CT Simulation Builder',
        ('--metadata=subtitle:' + $edition),
        '--metadata=author:TIGRE package reference 3.0.0',
        '--metadata=date:Manual version 1.1 - 28 October 2026',
        ('--lua-filter=' + (Join-Path $PSScriptRoot 'pdf_support/manual-tables.lua')),
        ('--include-in-header=' + (Join-Path $PSScriptRoot 'pdf_support/manual-header.tex')),
        '-V', 'papersize=a4', '-V', 'classoption=titlepage', '-V', 'geometry:margin=20mm', '-V', 'fontsize=10pt',
        '-V', 'colorlinks=true', '-V', 'monofont=Consolas', '-o', $target
    )
    & pandoc @arguments
    if ($LASTEXITCODE -ne 0) { throw "PDF conversion failed for $manual" }
    Write-Output "Created $target"
}
