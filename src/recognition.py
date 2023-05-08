# TODO Research on how to use AWSRekogniton
# TODO How to run python file on startup in linux

import boto3
import csv
import pandas as pd
import io
import matplotlib.image as mpimg
from PIL import Image
from io import BytesIO
import PIL
class FaceComparision():
    
    def __init__(self, access_key_id, secret_access_key) -> None:
        self.bucket_client = boto3.client('s3', region_name='eu-central-1', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
        self.client = boto3.client('rekognition', region_name='eu-central-1', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    def run(self, source_file, target_file):
        source_bytes = source_file
        target_bytes = target_file
        # imageSource = base64.decodebytes(base64.b64encode(open(sourceFile,'rb')))
        # imageTarget = base64.decodebytes(base64.b64encode(open(targetFile,'rb')))
        response = self.client.compare_faces(
                SimilarityThreshold=70,
                SourceImage={'Bytes': source_bytes},
                TargetImage={'Bytes': target_bytes})    
        # TODO this will fixed
        # print(f"number of faces {len(response['FaceMatches'])}")
        if len(response['FaceMatches']) > 0:
            print(f"Welcome again, given image is valid with validity of {response['FaceMatches'][0]['Face']['Confidence']}")
            self.delete_file('target.jpeg')
            return response['FaceMatches'][0]['Face']['Confidence']
        else:
            print(f"You are not Emir.")
            self.delete_file('target.jpeg')
            return 0
    def upload_file(self, file, file_name):
        self.bucket_client.upload_file(file, 'myawsdatabase', file_name)
    
    def delete_file(self, filename):
        response = self.bucket_client.delete_objects(
        Bucket='myawsdatabase', Delete={"Objects": [{"Key": filename}]})
    
    def get_file(self, filename):
        file_byte_string = self.bucket_client.get_object(Bucket='myawsdatabase', Key=filename)['Body'].read()
        return file_byte_string

    def get_len_db(self):
        response = self.bucket_client.list_objects_v2(Bucket='myawsdatabase')
        files = response.get("Contents")
        print(files)
        if files == None:
            return 0
        else:
            return len(files)
    
    def is_object_contains(self, filename):
        response = self.bucket_client.list_objects_v2(Bucket='myawsdatabase')
        files = response.get("Contents")
        for dictionary in files:
            if filename in dictionary.values():
                return True
        return False
    def log_in(self, source_file, target_file):
        if self.run(source_file, target_file) > 50:
            return True
        return False