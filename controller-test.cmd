@echo off
call "%~dp0scripts\windows\controller-test.cmd" %*
exit /b %ERRORLEVEL%
