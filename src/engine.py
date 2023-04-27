from recognition import FaceComparision
import csv
import cv2

class Capture():
    """
    TODO This class need to capture the 4 jpeg file of face with head pose estimation and concat them into one jpeg called target_file. 
    """
    def __init__(self) -> None:
        pass



if __name__ == "__main__":
    csv_file = "/Users/emirulurak/Desktop/dev/ozu/cs350/cs350_accessKeys.csv"
    with open(csv_file, 'r') as input:
        reader = csv.reader(input)
        next(input)
        for line in reader:
            access_key_id = line[0]
            secret_access_key = line[1]
    fc = FaceComparision(access_key_id=access_key_id, secret_access_key=secret_access_key)
    source_file = "../images/source.jpeg"
    target_file = "../images/target.jpeg" 
    fc.run(source_file=source_file, target_file=target_file)