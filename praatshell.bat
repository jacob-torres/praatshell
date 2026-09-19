@echo off
rem Start praatshell from PowerShell or the command prompt.
cd /d "%~dp0"
python -u -m praatshell %*
