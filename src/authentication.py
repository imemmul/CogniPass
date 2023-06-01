#!/usr/bin/python3.9

import pam
from engine import run_facial, speak
import subprocess
import time

login = False

def shutdown_machine():
    subprocess.run(['sudo', 'shutdown', '-h', 'now'])

facial_result = run_facial()
if facial_result:
    print(f"Auth successful") # this is reached
    login = pam.authenticate("emir", "422453", service='login')
    if login:
        speak("Hand Tracking processes is starting..")
    else:
        speak("Something Failed")
    
else:
    speak("I am turning of the machine in 5 seconds.")
    for _ in range(5, 0, -1):
        speak(str(_))
    speak("Bye")
    shutdown_machine()



