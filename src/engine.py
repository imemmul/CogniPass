#!/usr/bin/python3.9

from recognition import FaceComparision
import csv
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys
import logging
import subprocess
import pyautogui
pyautogui.FAILSAFE = False

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



# class Capture():
#     """
#     TODO This class need to capture the 4 jpeg file of face with head pose estimation and concat them into one jpeg called target_file. 
#     """
#     def __init__(self) -> None:
#         mp_face_mesh = mp.solutions.face_mesh
#         face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#         cap = cv2.VideoCapture(0)
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

def run_facial():
    count_running = 0
    time.sleep(2)
    print("In loop")
    if not fc.get_len_db() > 0:
        speak("I am registering you please wait.")
        authentication(0)
    else:
        speak("I am logging in please wait.")
        authentication(1)
        source_img = fc.get_file('source.jpeg')
        target_img = fc.get_file('target.jpeg')
        if fc.log_in(source_img, target_img):
            speak("Hello, Emir")
            logging.basicConfig(filename='/home/emir/Desktop/dev/CogniPass/scripts/logfile.log', level=logging.DEBUG)
            logging.debug('Script executed successfully.')
            return True
        else:
            speak("Please get away from this computer.")
            logging.basicConfig(filename='/home/emir/Desktop/dev/CogniPass/scripts/logfile.log', level=logging.DEBUG)
            logging.debug('Script executed successfully.')
            return False

def run_hand():
    cap = cv2.VideoCapture(0)   # capture video '0' one cam
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # hiegh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hand_detector = mp.solutions.hands.Hands()  # detect hand
    screen_width, screen_height = pyautogui.size()
    index_y = 0

    '''Smoothen the movement of mouse to stop at the exact position of,
    our hand movement without any shake in the movement of the mouse'''
    smoothening = 6
    plocx, plocy = 0, 0
    clocx, clocy = 0, 0 

    while True:
        _, frame = cap.read()   # read data from cap
        '''Flip the frame or screen since the camera shows the mirror image,
        of our hand and moves in opposite direction so we need to flip the screen'''
        frame = cv2.flip(frame, 1)
        # shape gives frame height and width using shape 
        frame_height, frame_width = screen_height, screen_width
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # detect on rgb frame color
        output = hand_detector.process(frame)
        hands = output.multi_hand_landmarks # hand landmark
        
        if hands:
            for hand in hands:
                
                landmarks = hand.landmark
                
                for id, landmark in enumerate(landmarks):   # add counter
                    # show the landmarks on kernel in x and y axis
                    # x and y axis is multiplies by the height and width to get the x and y axis on the frames
                    x = int(landmark.x*frame_width)
                    y = int(landmark.y*frame_height)
                    # print(x,y)
                    # Index finger tip point number is 8
                    # and draw a boundary to the point a circle
                    if id == 8:
                        cv2.circle(img=frame, center=(x,y), radius=15, color=(0, 255, 255))
                        # pyautogui.moveTo(x,y)
                        index_x = (screen_width/frame_width)*x
                        index_y = (screen_height/frame_height)*y
                        # co-ordinates need to be changed 
                        # smoothining varies with the change in the smoothening factor
                        clocx = plocx + (index_x - plocx) /smoothening
                        clocy = plocy + (index_y - plocy) /smoothening
                        pyautogui.moveTo(clocx, clocy)
                        plocx, plocy = clocx, clocy
                    
                    # thumb tip point number is 4

                    if id == 4:
                        cv2.circle(img=frame, center=(x,y), radius=15, color=(0, 255, 255))
                        thumb_x = (screen_width/frame_width)*x
                        thumb_y = (screen_height/frame_height)*y
                        print('distance : ', abs(index_y - thumb_y))
                        if abs(index_y - thumb_y) < 70:
                            print('click')
                            pyautogui.click()
                            pyautogui.sleep(1)
        # cv2.imshow('Virtual Mouse', frame)  # show image
        cv2.waitKey(1)  # waits for key infinitely