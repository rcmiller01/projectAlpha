@echo off
REM Autopilot Bootloader Service Management Script
REM For Windows Task Scheduler or manual service management

set SCRIPT_DIR=%~dp0
set PYTHON_SCRIPT=%SCRIPT_DIR%autopilot_bootloader.py
set CONFIG_FILE=%SCRIPT_DIR%bootloader_config.json
set LOG_FILE=%SCRIPT_DIR%bootloader_service.log
set PID_FILE=%SCRIPT_DIR%bootloader_service.pid

if "%1"=="start" goto START
if "%1"=="stop" goto STOP
if "%1"=="restart" goto RESTART
if "%1"=="status" goto STATUS
if "%1"=="install" goto INSTALL
if "%1"=="uninstall" goto UNINSTALL

echo Usage: %0 {start^|stop^|restart^|status^|install^|uninstall}
exit /b 1

:START
echo Starting Autopilot Bootloader Service...
if exist "%PID_FILE%" (
    echo Service may already be running. Check status first.
    goto STATUS
)

start /b python "%PYTHON_SCRIPT%" --config "%CONFIG_FILE%" > "%LOG_FILE%" 2>&1
echo Service started. Check %LOG_FILE% for output.
goto END

:STOP
echo Stopping Autopilot Bootloader Service...
python "%PYTHON_SCRIPT%" --stop
if exist "%PID_FILE%" del "%PID_FILE%"
echo Service stopped.
goto END

:RESTART
call "%0" stop
timeout /t 5 /nobreak > nul
call "%0" start
goto END

:STATUS
echo Checking Autopilot Bootloader Service status...
python "%PYTHON_SCRIPT%" --status
goto END

:INSTALL
echo Installing Autopilot Bootloader as Windows Task...
schtasks /create /tn "AutopilotBootloader" /tr "\"%PYTHON_SCRIPT%\" --config \"%CONFIG_FILE%\"" /sc onstart /ru SYSTEM /f
if %errorlevel%==0 (
    echo Task scheduled successfully. Run 'schtasks /run /tn AutopilotBootloader' to start.
) else (
    echo Failed to schedule task. Run as Administrator.
)
goto END

:UNINSTALL
echo Uninstalling Autopilot Bootloader Windows Task...
schtasks /delete /tn "AutopilotBootloader" /f
if %errorlevel%==0 (
    echo Task removed successfully.
) else (
    echo Failed to remove task or task not found.
)
goto END

:END
echo Done.
