@echo off
echo ========================================
echo Smart AI Agent Environment Setup
echo ========================================
echo.

echo Current OPENAI_API_KEY status:
if defined OPENAI_API_KEY (
    echo Found system OPENAI_API_KEY: %OPENAI_API_KEY:~0,10%...
    echo This might be your old key.
) else (
    echo No system OPENAI_API_KEY found.
)

echo.
echo Options:
echo 1. Clear system OPENAI_API_KEY
echo 2. Set new OPENAI_API_KEY
echo 3. Run Python setup script
echo 4. Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Clearing system OPENAI_API_KEY...
    set OPENAI_API_KEY=
    echo OPENAI_API_KEY cleared from current session.
    echo Note: You may need to restart your terminal for permanent changes.
    pause
    goto :eof
)

if "%choice%"=="2" (
    echo.
    set /p new_key="Enter your new OpenAI API key: "
    set OPENAI_API_KEY=%new_key%
    echo OPENAI_API_KEY set for current session.
    echo Note: This is only for the current terminal session.
    echo For permanent changes, use the .env file or system environment variables.
    pause
    goto :eof
)

if "%choice%"=="3" (
    echo Running Python setup script...
    python setup_env.py
    pause
    goto :eof
)

if "%choice%"=="4" (
    echo Goodbye!
    goto :eof
)

echo Invalid choice. Please try again.
pause 