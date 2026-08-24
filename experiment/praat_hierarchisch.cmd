@echo off
rem Praat met het hierarchische woord-model (model_hierarchisch.pt), naast praat.cmd
rem (die het char-model gebruikt) - zo kun je ze naast elkaar vergelijken.
"%~dp0..\.venv\Scripts\python.exe" "%~dp0praat_hierarchisch.py" %*
