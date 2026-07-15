@echo OFF

REM Set the path to the main Anaconda installation directory.
SET CONDA_ROOT=C:\Users\CHINMAY\anaconda3

REM Activate the target conda environment.
CALL "%CONDA_ROOT%\condabin\conda.bat" activate llm-gpu
IF %errorlevel% neq 0 (
    echo [WRAPPER-ERROR] Failed to activate conda environment.
    exit /b 1
)

REM Execute the python script, passing along all arguments.
python src\main\resources\scripts\init_model.py %*

REM Capture the exit code and deactivate before exiting.
SET SCRIPT_EXIT_CODE=%errorlevel%
CALL "%CONDA_ROOT%\condabin\conda.bat" deactivate
exit /b %SCRIPT_EXIT_CODE%