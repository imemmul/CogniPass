# TODO Research on how to use AWSRekogniton
# TODO How to run python file on startup in linux

import boto3
import csv
import pandas as pd

class FaceComparision():
    
    def __init__(self, access_key_id, secret_access_key) -> None:
        self.client = boto3.client('rekognition', region_name='eu-central-1', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    def run(self, source_file, target_file):
        source_bytes = open(source_file, 'rb')
        target_bytes = open(target_file, 'rb')
        # imageSource = base64.decodebytes(base64.b64encode(open(sourceFile,'rb')))
        # imageTarget = base64.decodebytes(base64.b64encode(open(targetFile,'rb')))
        response = self.client.compare_faces(
                SimilarityThreshold=70,
                SourceImage={'Bytes': source_bytes.read()},
                TargetImage={'Bytes': target_bytes.read()})    
        # TODO this will fixed
        # print(f"number of faces {len(response['FaceMatches'])}")
        source_bytes.close()
        target_bytes.close()
        if len(response['FaceMatches']) > 0:
            print(f"Welcome again, given image is valid with validity of {response['FaceMatches'][0]['Face']['Confidence']}")
            return response['FaceMatches'][0]['Face']['Confidence']
        else:
            print(f"You are not Emir.")
            return 0