#!/usr/bin/python3.9

from recognition import FaceComparision
import csv
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import logging
import subprocess
import pam

PRESENCE_CHECK_INTERVAL = 20

os.environ['DISPLAY'] = ':0'


def load_csv(csv_file):
    with open(csv_file, 'r') as input:
            reader = csv.reader(input)
            next(input)
            for line in reader:
                access_key_id = line[0]
                secret_access_key = line[1]
    return access_key_id, secret_access_key

target_folder = "/home/emir/Desktop/dev/CogniPass/images/target/"
source_folder = "/home/emir/Desktop/dev/CogniPass/images/source/"
csv_file = "/home/emir/Desktop/dev/authentication/cs350_accessKeys.csv"
id, secret = load_csv(csv_file)
source_file = source_folder + "source.jpeg"
target_file = target_folder + "target.jpeg"
fc = FaceComparision(access_key_id=id, secret_access_key=secret)


# FIXME below directories should be absolute path
# TODO optimize this function with running camera in background and start prefetching images
class FaceTracker():
    def __init__(self) -> None:
        pass
    def run(self, command):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)
        counter = 0
        while cap.isOpened():
            success, image = cap.read()

            # Flip the image horizontally for a later selfie-view display
            # Also convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image.flags.writeable = False
            
            # Get the result
            results = face_mesh.process(image)
            
            # To improve performance
            image.flags.writeable = True
            
            # Convert the color space from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])       
                    
                    # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)

                    # Convert it to the NumPy array
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # The camera matrix
                    focal_length = 1 * img_w

                    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                            [0, focal_length, img_w / 2],
                                            [0, 0, 1]])

                    # The Distance Matrix
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360

                    if command == 0:
                        folder = source_folder
                        print(f"i in source {counter}")
                        if counter == 4:
                            concat_images(command=0)
                            cap.release()
                            break
                    elif command == 1:
                        folder = target_folder
                        if counter == 4:
                            concat_images(command=1)
                            cap.release()
                            break
                    # See where the user's head tilting
                    if y < -10:
                        pass
                    elif y > 10:
                        pass
                    elif x < -10:
                        pass
                    elif x > 10:
                        pass
                    else:
                        cv2.imwrite(filename=f"{folder}forward_{counter}.jpeg", img=image)
                        counter+= 1

        cap.release()

def authentication(command):
   face_tracker = FaceTracker()
   face_tracker.run(command)


def delete_aut(folder):
    for file in os.listdir(folder):
        dir = folder + file
        os.remove(dir)

def concat_images(command):
    if command == 0:
        folder = source_folder
        sorted_list = sorted(os.listdir(folder))
        list_2d = [[cv2.imread(folder+sorted_list[0]), cv2.imread(folder+sorted_list[1])], [cv2.imread(folder+sorted_list[2]), cv2.imread(folder+sorted_list[3])]]
        img = cv2.vconcat([cv2.hconcat(list_h) 
                            for list_h in list_2d])
        cv2.imwrite(folder+"source.jpeg", img=img)
        fc.upload_file(folder+"source.jpeg", "source.jpeg")
        delete_aut(folder)
    elif command == 1:
        folder = target_folder
        sorted_list = sorted(os.listdir(folder))
        list_2d = [[cv2.imread(folder+sorted_list[0]), cv2.imread(folder+sorted_list[1])], [cv2.imread(folder+sorted_list[2]), cv2.imread(folder+sorted_list[3])]]
        img = cv2.vconcat([cv2.hconcat(list_h) 
                            for list_h in list_2d])
        cv2.imwrite(folder+"target.jpeg", img=img)
        fc.upload_file(folder+"target.jpeg", "target.jpeg")
        delete_aut(folder)
       
def check_user(folder):
    # for local storage
    if len(os.listdir(folder)) == 0:
        print("no user found registering")
        return False
    return True

def speak(text):
    subprocess.call(['espeak', text])

def register_user(silent):
    if not silent:
        speak("I am registering you please wait.")
    authentication(0)

def login_user(silent):
    if not silent:
        speak("I am logging in please wait.")
    authentication(1)
    
def run_facial(silent):
    if not fc.get_len_db() > 0:
        # time.sleep(2)
        register_user(silent)
        login_user(silent)
    else:
        # time.sleep(2)
        login_user(silent)
        source_img = fc.get_file('source.jpeg')
        target_img = fc.get_file('target.jpeg')
        if fc.log_in(source_img, target_img):
            if not silent:
                time.sleep(3)
                speak("Welcome, admin")
            logging.basicConfig(filename='/home/emir/Desktop/dev/CogniPass/scripts/logfile.log', level=logging.DEBUG)
            logging.debug('Script executed successfully.')
            return True
        else:
            if not silent:
                time.sleep(3)
                speak("Please get away from this computer.")
            logging.basicConfig(filename='/home/emir/Desktop/dev/CogniPass/scripts/logfile.log', level=logging.DEBUG)
            logging.debug('Script executed successfully.')
            return False

def unlock_screen():
    # Create a PAM instance
    # FIXME doesnot unlocks
    p = pam.pam()

    # Authenticate the user
    authenticated = p.authenticate("emir", "422453")

    if authenticated:
        print("User authenticated successfully.")
        # Unlock the screen by changing the PAM session
        session_opened = p.open_session()
        if session_opened:
            print("Screen unlocked successfully.")
            return True
        else:
            print("Failed to unlock the screen.")
            return False
    else:
        print("Authentication failed.")
        return False


def check_presence():
    """
        Doesn't lock the screen so transformed into shutdown operation
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    last_presence_time = time.time()
    locked = False
    while True:
        ret, frame = cap.read()
        # time.sleep(5) # usually my machine works in low fps (10-11) thats why waiting 5 second to change frame
        
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face = face_mesh.process(frame_rgb)
        
        if face.multi_face_landmarks: # face detected due to limitations of service script it deprecated
            last_presence_time = time.time()
            # TODO UNLOCK
            if locked:
                speak("Welcome back. You need to enter password.")
                locked = False

        if time.time() - last_presence_time > PRESENCE_CHECK_INTERVAL and not locked:
            # speak("I am locking the machine.") # Deprecated
            time.sleep(1)
            # Below code deprecated
            # cmd = subprocess.run(["xdg-screensaver", "lock"]) # Deprecated
            # if cmd.returncode == 0:
            #     logging.basicConfig(filename='/home/emir/Desktop/dev/CogniPass/scripts/logfile.log', level=logging.DEBUG)
            #     logging.debug('Script executed successfully.')
            #     locked = True
                
            # else:
            #     logging.basicConfig(filename='/home/emir/Desktop/dev/CogniPass/scripts/logfile.log', level=logging.DEBUG)
            #     logging.debug(f'Script executed usuccessfully. {cmd}')
            #     locked = False
            locked = True
            return True

    cap.release()
    cv2.destroyAllWindows()