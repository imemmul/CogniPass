#!/usr/bin/python3.9

import pam
from engine import run_facial, speak, check_presence
import subprocess
import time
import os
import threading

os.environ['DISPLAY'] = ':0'

facial_result = run_facial(silent=False)


def shutdown_machine(countdown, text):
    speak(f"{text}, I am turning of the machine in {countdown} seconds.")
    for _ in range(countdown, 0, -1):
        speak(str(_))
        time.sleep(1)
    speak("Bye")
    time.sleep(0.5)
    subprocess.run(['sudo', 'shutdown', '-h', 'now'])

if facial_result:
    print(f"Auth successful") # this is reached
    time.sleep(2)
    if check_presence():
        shutdown_machine(3, "You are not present")
    
else:
    time.sleep(2)
    shutdown_machine(5, "You are not admin.")



