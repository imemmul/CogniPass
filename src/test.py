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
import cv2 
import mediapipe as mp

# service env:
    
#     environ({'LANG': 'en_US.UTF-8', 'LC_ADDRESS': 'tr_TR.UTF-8', 'LC_IDENTIFICATION': 'tr_TR.UTF-8', 'LC_MEASUREMENT': 'tr_TR.UTF-8', 'LC_MONETARY': 'tr_TR.UTF-8', 'LC_NAME': 'tr_TR.UTF-8', 'LC_NUMERIC': 'tr_TR.UTF-8', 'LC_PAPER': 'tr_TR.UTF-8', 'LC_TELEPHONE': 'tr_TR.UTF-8', 'LC_TIME': 'tr_TR.UTF-8', 'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin', 'INVOCATION_ID': 'fadd1f935c424376bc84e49596497785', 'JOURNAL_STREAM': '8:92560', 'PYTHONPATH': '/home/emir/.local/lib/python3.9/site-packages', 'DISPLAY': ':0', 'QT_QPA_PLATFORM_PLUGIN_PATH': '/home/emir/.local/lib/python3.9/site-packages/cv2/qt/plugins', 'QT_QPA_FONTDIR': '/home/emir/.local/lib/python3.9/site-packages/cv2/qt/fonts', 'LD_LIBRARY_PATH': '/home/emir/.local/lib/python3.9/site-packages/cv2/../../lib64:'})
# environ({'SHELL': '/bin/bash', 'CONDA_EXE': '/home/emir/anaconda3/bin/conda', '_CE_M': '', 'LC_ADDRESS': 'tr_TR.UTF-8', 'LC_NAME': 'tr_TR.UTF-8', 'LC_MONETARY': 'tr_TR.UTF-8', 'PWD': '/home/emir/Desktop/dev/CogniPass/src', 'LOGNAME': 'emir', 'XDG_SESSION_TYPE': 'tty', 'MOTD_SHOWN': 'pam', 'HOME': '/home/emir', 'LC_PAPER': 'tr_TR.UTF-8', 'LANG': 'C.UTF-8', 'LS_COLORS': 'rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.zst=01;31:*.tzst=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.wim=01;31:*.swm=01;31:*.dwm=01;31:*.esd=01;31:*.jpg=01;35:*.jpeg=01;35:*.mjpg=01;35:*.mjpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:', 'SSH_CONNECTION': '192.168.1.2 52603 192.168.1.16 22', 'LESSCLOSE': '/usr/bin/lesspipe %s %s', 'XDG_SESSION_CLASS': 'user', 'LC_IDENTIFICATION': 'tr_TR.UTF-8', 'TERM': 'xterm-256color', '_CE_CONDA': '', 'LESSOPEN': '| /usr/bin/lesspipe %s', 'USER': 'emir', 'CONDA_SHLVL': '0', 'SHLVL': '1', 'LC_TELEPHONE': 'tr_TR.UTF-8', 'LC_MEASUREMENT': 'tr_TR.UTF-8', 'XDG_SESSION_ID': '18', 'CONDA_PYTHON_EXE': '/home/emir/anaconda3/bin/python', 'LC_CTYPE': 'C.UTF-8', 'XDG_RUNTIME_DIR': '/run/user/1000', 'SSH_CLIENT': '192.168.1.2 52603 22', 'LC_TIME': 'tr_TR.UTF-8', 'XDG_DATA_DIRS': '/usr/local/share:/usr/share:/var/lib/snapd/desktop', 'PATH': '/home/emir/.local/bin:/home/emir/.cargo/bin:/home/emir/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin', 'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/1000/bus', 'SSH_TTY': '/dev/pts/1', 'LC_NUMERIC': 'tr_TR.UTF-8', 'OLDPWD': '/home/emir', '_': '/usr/bin/python3.9', 'QT_QPA_PLATFORM_PLUGIN_PATH': '/home/emir/.local/lib/python3.9/site-packages/cv2/qt/plugins', 'QT_QPA_FONTDIR': '/home/emir/.local/lib/python3.9/site-packages/cv2/qt/fonts', 'LD_LIBRARY_PATH': '/home/emir/.local/lib/python3.9/site-packages/cv2/../../lib64:'})

def test_cv():
    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands 
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    while True:
        success,img = cap.read()
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handlandmark in results.multi_hand_landmarks:
                for id,lm in enumerate(handlandmark.landmark):
                    print(id,lm) # lm produce output in ratio format
                    h,w,_ = img.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    cv2.circle(img,(cx,cy),4,(0,0,255),cv2.FILLED)
                    
                mpDraw.draw_landmarks(img,handlandmark,mpHands.HAND_CONNECTIONS)

        cv2.imshow('Image',img)
        if cv2.waitKey(1) & 0xff==ord('q'):
            break

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

def test_hand():
    pass

import subprocess
if __name__ == "__main__":
    print(f"Env {os.environ}")
