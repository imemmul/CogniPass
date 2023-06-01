#!/usr/bin/python3.9

import pam
from engine import run_facial, speak, run_background_admin
import subprocess
import time

facial_result = run_facial()
time.sleep(1)

def shutdown_machine(countdown):
    speak(f"I am turning of the machine in {countdown} seconds.")
    for _ in range(countdown, 0, -1):
        speak(str(_))
        time.sleep(1)
    subprocess.run(['sudo', 'shutdown', '-h', 'now'])

if facial_result:
    print(f"Auth successful") # this is reached
    speak("Background Detection has been started.")
    run_background_admin()
    
else:
    shutdown_machine(5)
    speak("Bye")



