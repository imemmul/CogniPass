# TODO Research on how to use AWSRekogniton
# TODO How to run python file on startup in linux

import boto3
import base64
from io import BytesIO
access_id = ''
access_key = ''


def compare_faces():
    sourceFile='../images/source.jpg'
    targetFile='../images/target.jpg'
    client=boto3.client('rekognition', region_name='eu-central-1', aws_access_key_id=access_id, aws_secret_access_key=access_key)
   
    # imageSource = base64.decodebytes(base64.b64encode(open(sourceFile,'rb')))
    # imageTarget = base64.decodebytes(base64.b64encode(open(targetFile,'rb')))
    with open(sourceFile, 'rb') as image_source:
        with open(targetFile, 'rb') as image_target:
            response = client.compare_faces(
                SimilarityThreshold=70,
                SourceImage={'Bytes': image_source.read()},
                TargetImage={'Bytes': image_target.read()}
            )
    print(response)
    for faceMatch in response['FaceMatches']:
        position = faceMatch['Face']['BoundingBox']
        confidence = str(faceMatch['Face']['Confidence'])
        print('The face at ' +
               str(position['Left']) + ' ' +
               str(position['Top']) +
               ' matches with ' + confidence + '% confidence')

    image_source.close()
    image_target.close() 

if __name__ == "__main__":
    compare_faces()