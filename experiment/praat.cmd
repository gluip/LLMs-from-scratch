@echo off
rem Praat met het model zonder het venv-pad te hoeven typen: gewoon `praat` in deze map.
rem %~dp0 is de map van dit bestand, dus dit werkt vanuit elke werkdirectory.
"%~dp0..\.venv\Scripts\python.exe" "%~dp0praat.py" %*
