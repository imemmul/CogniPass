from recognition import FaceComparision
import csv
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from os import system
# class Capture():
#     """
#     TODO This class need to capture the 4 jpeg file of face with head pose estimation and concat them into one jpeg called target_file. 
#     """
#     def __init__(self) -> None:
#         mp_face_mesh = mp.solutions.face_mesh
#         face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#         cap = cv2.VideoCapture(0)
source_folder = "../images/source/"
target_folder = "../images/target/"
# TODO optimize this function with running camera in background and start prefetching images
class FaceTracker():
    def __init__(self) -> None:
        mp_face_mesh = mp.solutions.face_mesh
        self.mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        try:
            self.cap = cv2.VideoCapture(1)
        except:
            self.cap = cv2.VideoCapture(1)
        self.counter = 0
    def run(self, command):
        self.counter = 0
        while self.cap.isOpened():
            _, image = self.cap.read()
            time.sleep(0.2)
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        print("I am in cam")
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

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

                # The distortion parameters
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
                z = angles[2] * 360
                if command == 0:
                    folder = source_folder
                    print(f"i in source {self.counter}")
                    if self.counter == 4:
                        concat_images(command=0)
                        self.cap.release()
                        break
                elif command == 1:
                    folder = target_folder
                    if self.counter == 4:
                        concat_images(command=1)
                        self.cap.release()
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
                    cv2.imwrite(filename=f"{folder}forward_{self.counter}.jpeg", img=image)
                    self.counter+= 1
                
                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
                
                cv2.line(image, p1, p2, (255, 0, 0), 3)

                # Add the text on the image


            self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec)


class HandTracker():
    def __init__(self) -> None:
        pass

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
        fc.upload_file(folder+"source.jpeg")
        delete_aut(folder)
    elif command == 1:
        folder = target_folder
        sorted_list = sorted(os.listdir(folder))
        list_2d = [[cv2.imread(folder+sorted_list[0]), cv2.imread(folder+sorted_list[1])], [cv2.imread(folder+sorted_list[2]), cv2.imread(folder+sorted_list[3])]]
        img = cv2.vconcat([cv2.hconcat(list_h) 
                            for list_h in list_2d])
        cv2.imwrite(folder+"target.jpeg", img=img)
        fc.upload_file(folder+"target.jpeg")
        delete_aut(folder)
       
def check_user(folder):
    if len(os.listdir(folder)) == 0:
        print("no user found registering")
        return False
    return True

def speak(text):
    system(f'say {text}')

def load_csv(csv_file):
    with open(csv_file, 'r') as input:
            reader = csv.reader(input)
            next(input)
            for line in reader:
                access_key_id = line[0]
                secret_access_key = line[1]
    return access_key_id, secret_access_key

from test import test_register
if __name__ == "__main__":
    csv_file = "/Users/emirulurak/Desktop/dev/ozu/cs350/cs350_accessKeys.csv"
    id, secret = load_csv(csv_file)
    source_file = source_folder + "source.jpeg"
    target_file = target_folder + "target.jpeg"
    fc = FaceComparision(access_key_id=id, secret_access_key=secret)
    while True:
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
                speak("Hello Emir")
            else:
                speak("Please get away from this computer.")
        
    # test_register(fc)