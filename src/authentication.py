#!/usr/bin/python3.9

import pam
from engine import run

facial_result = run()
if facial_result:
    print(f"Auth successful")
    p = pam.authenticate("emir", "422453", service='login')


