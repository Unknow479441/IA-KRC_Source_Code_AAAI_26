@echo off
:: Install SC2 and add the custom maps

:: Set SC2PATH to your actual StarCraft II installation path
set SC2PATH=D:\StarCraft II
echo SC2PATH is set to %SC2PATH%

:: Check if StarCraft II is installed
if not exist "%SC2PATH%" (
    echo StarCraftII is not found in the expected location.
    echo Please make sure StarCraft II is installed in: %SC2PATH%
    pause
    exit /b 1
) else (
    echo StarCraftII is already installed.
)

:: Set up maps directory
set MAP_DIR=%SC2PATH%\Maps
echo MAP_DIR is set to %MAP_DIR%

if not exist "%MAP_DIR%" (
    mkdir "%MAP_DIR%"
)

:: Download and extract SMAC maps
powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip' -OutFile 'SMAC_Maps.zip'}"
powershell -Command "& {Expand-Archive -Path 'SMAC_Maps.zip' -DestinationPath '%MAP_DIR%' -Force}"

:: Copy empty passive map
copy "src\envs\starcraft2\maps\SMAC_Maps\empty_passive.SC2Map" "%MAP_DIR%"

:: Copy 8piece map
copy "src\envs\starcraft2\maps\SMAC_Maps\8piece.SC2Map" "%MAP_DIR%"

:: Clean up
del SMAC_Maps.zip

echo StarCraft II and SMAC are installed.
pause