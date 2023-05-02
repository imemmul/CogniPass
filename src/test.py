from recognition import FaceComparision
from os import system
import cv2
import os
import time
# def concat_images(folder):
#     sorted_list = sorted(os.listdir(folder))
#     print(sorted_list[2])
#     list_2d = [[cv2.imread(folder+sorted_list[3]), cv2.imread(folder+sorted_list[4])], [cv2.imread(folder+sorted_list[5]), cv2.imread(folder+sorted_list[6])]]
#     img = cv2.vconcat([cv2.hconcat(list_h) 
#                         for list_h in list_2d])
#     cv2.imwrite(folder+"target_1.jpeg", img=img)

def speak(text):
    system(f'say {text}')

def test_register(fc:FaceComparision):
    ex_source = "../example_images/source/source.jpeg"
    ex_target = "../example_images/target/target_1.jpeg"
    # fc.upload_file(ex_source, 'source.jpeg')
    # time.sleep(1)
    print(fc.get_len_db())
    print(fc.is_object_contains('source.jpeg'))
    if (fc.run(source_file=ex_source, target_file=ex_target)) > 50:
        speak("You are logged in. Hello Emir")
    else:
        speak("Please get away from this computer or I am calling Emir.")

def test_login():
    pass