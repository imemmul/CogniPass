from recognition import FaceComparision
import csv
import cv2
import mediapipe as mp
import numpy as np
import time
import os
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
def save_sources(command):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


    cap = cv2.VideoCapture(0)
    i = 0
    while cap.isOpened():
        success, image = cap.read()
        time.sleep(0.2)
        start = time.time()

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
                    print(f"i in source {i}")
                    if i == 4:
                        concat_images(command=0)
                        cap.release()
                        break
                elif command == 1:
                    folder = target_folder
                    print(f"i in target {i}")
                    if i == 4:
                        concat_images(command=1)
                        cap.release()
                        break
                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                    # cv2.imwrite(filename=folder+"left.jpeg", img=image)
                elif y > 10:
                    text = "Looking Right"
                    # cv2.imwrite(filename=folder+"right.jpeg", img=image)
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    # cv2.imwrite(filename=folder+"up.jpeg", img=image)
                    text = "Looking Up"
                else:
                    cv2.imwrite(filename=f"{folder}forward_{i}.jpeg", img=image)
                    i+= 1
                    text = "Forward"
                
                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
                
                cv2.line(image, p1, p2, (255, 0, 0), 3)

                # Add the text on the image
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            #print("FPS: ", fps)

            cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)


        # cv2.imshow('Head Pose Estimation', image)
    cap.release()


def concat_images(command):
    if command == 0:
        folder = source_folder
        list_2d = [[cv2.imread(folder+os.listdir(folder)[0]), cv2.imread(folder+os.listdir(folder)[1])], [cv2.imread(folder+os.listdir(folder)[2]), cv2.imread(folder+os.listdir(folder)[3])]]
        img = cv2.vconcat([cv2.hconcat(list_h) 
                            for list_h in list_2d])
        cv2.imwrite(folder+"source.jpeg", img=img)
    elif command == 1:
        folder = target_folder
        list_2d = [[cv2.imread(folder+os.listdir(folder)[0]), cv2.imread(folder+os.listdir(folder)[1])], [cv2.imread(folder+os.listdir(folder)[2]), cv2.imread(folder+os.listdir(folder)[3])]]
        img = cv2.vconcat([cv2.hconcat(list_h) 
                            for list_h in list_2d])
        cv2.imwrite(folder+"target.jpeg", img=img)

if __name__ == "__main__":
    csv_file = "/Users/emirulurak/Desktop/dev/ozu/cs350/cs350_accessKeys.csv"
    # concat_images()
    run = True
    while run:
        command = input("Sign in or Sign up: ")
        if command == "r":
            print(f"Please show your face to register")
            save_sources(command=0)
            print("Signed up")
        elif command == "l":
            print(f"loggin in show your face")
            save_sources(command=1)
            with open(csv_file, 'r') as input:
                reader = csv.reader(input)
                next(input)
                for line in reader:
                    access_key_id = line[0]
                    secret_access_key = line[1]
            fc = FaceComparision(access_key_id=access_key_id, secret_access_key=secret_access_key)
            source_file = "../images/source/source.jpeg"
            target_file = "../images/target/target.jpeg"
            fc.run(source_file=source_file, target_file=target_file)
            run = False
            break