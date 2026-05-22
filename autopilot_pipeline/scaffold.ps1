$base = $PSScriptRoot

$dirs = @(
    "workflows", "agents", "workers", "contracts", "orchestration",
    "renderer", "infrastructure", "tools", "logs", "observability",
    "config\prompts", "data\vectorstore", "data\assets",
    "outputs\audio", "outputs\video", "outputs\visual"
)

$d = $dirs[0]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[1]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[2]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[3]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[4]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[5]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[6]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[7]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[8]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[9]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[10]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[11]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[12]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[13]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[14]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
$d = $dirs[15]; New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null

Set-Content -Path "$base\workflows\__init__.py" -Value ""
Set-Content -Path "$base\agents\__init__.py" -Value ""
Set-Content -Path "$base\workers\__init__.py" -Value ""
Set-Content -Path "$base\contracts\__init__.py" -Value ""
Set-Content -Path "$base\orchestration\__init__.py" -Value ""
Set-Content -Path "$base\renderer\__init__.py" -Value ""
Set-Content -Path "$base\infrastructure\__init__.py" -Value ""
Set-Content -Path "$base\tools\__init__.py" -Value ""
Set-Content -Path "$base\observability\__init__.py" -Value ""

Write-Host "Scaffold complete."
Get-ChildItem -Directory $base | Select-Object Name
