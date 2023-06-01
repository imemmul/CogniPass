#!/usr/bin/python3.9

import pam
from engine import run_facial, speak, check_presence
import subprocess
import time
import os
import threading

facial_result = run_facial(silent=True)


def shutdown_machine(countdown):
    speak(f"I am turning of the machine in {countdown} seconds.")
    for _ in range(countdown, 0, -1):
        speak(str(_))
        time.sleep(1)
    speak("Bye")
    subprocess.run(['sudo', 'shutdown', '-h', 'now'])

if facial_result:
    print(f"Auth successful") # this is reached
    time.sleep(2)
    check_presence()
    
else:
    time.sleep(2)
    shutdown_machine(1)



