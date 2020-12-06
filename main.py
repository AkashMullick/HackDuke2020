
import os, io, cv2
from google.cloud import vision
from absl import app, flags
from absl.flags import FLAGS
import math

import time
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "hackduke-2020-297720-1aecc6f0b1fc.json"

client = vision.ImageAnnotatorClient()


flags.DEFINE_string('video', '0', 'path to input video, set to 0 for webcam')
flags.DEFINE_integer('capRate', 100, 'every amount of frames to send an API call')
flags.DEFINE_boolean('info', True, 'print info on detections')
flags.DEFINE_boolean('output', True, 'write detections to CSV')
flags.DEFINE_boolean('Display', True, 'Display individual frames (Debug)')


def main(_argv):

   # cap = cv2.VideoCapture(FLAGS.video)

    df = {
        'Roll':[],
        'Pitch':[],
        'Yaw':[],
        'Joy':[],
        'Sorrow':[],
        'Anger':[],
        'Surprise':[],
        'Engagement':[],
        'Time':[]
    }

    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')
    start = time.perf_counter()
    frameNum = 0
    #Main capture loop
    while True:
        
        #Get frame
        #ret, frame = cap.read()
        ret = True
        frame = cv2.imread("Student-Studying.jpeg")

        if not ret:
            print('Video has ended or failed')
            break

        cv2.namedWindow("Current Frame", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Current Frame", frame)

        #every 100 frames make API call
        if frameNum % FLAGS.capRate == 0:

            #store capture
            cv2.imwrite('./tmp.png', frame)

            #Use Vision API to get predictions
            with io.open('./tmp.png', 'rb') as imageFile:
                content = imageFile.read()
            image = vision.Image(content=content)
            
            #Perform Face detection
            response = client.face_detection(image=image)
            faces  = response.face_annotations

            print("Face Features:")
            YawOffset =  13.23
            PitchOffset = 14.85
#######
#Positive corrilation: Joy, neutral headpose,
#Negative Corrilation: Neutral emotion
#Camera Axis (Different from face)
#    Y-
#    ^
#    |
#   Z(x)--> X+
#
#
#Roll: Around Z CW - No adjustment
#Yaw: Around y CW -Students look down on camera and above for notifications
#Pitch: Around x CCW Students Look left & rigt on screen (Small rotation)
#
#
            

            for face in faces:
                if face.detection_confidence > 0.6:
                    df['Roll'].append(face.roll_angle)
                    df['Pitch'].append(face.tilt_angle)
                    df['Yaw'].append(face.pan_angle)
                    df['Joy'].append(likelihood_name[face.joy_likelihood])
                    df['Sorrow'].append(likelihood_name[face.sorrow_likelihood])
                    df['Anger'].append(likelihood_name[face.anger_likelihood])
                    df['Surprise'].append(likelihood_name[face.surprise_likelihood])

                    

                    Y = (1.0-min(1.0,(abs(face.pan_angle-(face.pan_angle-YawOffset)/2.0)/(abs(face.pan_angle)-YawOffset))))*120.0
                    P= (1.0-min(1.0, (abs(face.tilt_angle-(face.tilt_angle-PitchOffset)/2.0)/(abs(face.tilt_angle)-PitchOffset))))*235.0
                    JoyNormal=(face.joy_likelihood/5.0)*275.3
                    Emotions = (1.0-min(1.0, (math.log(((5*4.0-(face.joy_likelihood+face.sorrow_likelihood+face.anger_likelihood+face.surprise_likelihood))/20.0)*135.5+0.75)/(100.0*math.log(1.0038)))))*55.0
                    Engagement = Y + P + JoyNormal+Emotions

                    df['Engagement'].append(Engagement)
                    df['Time'].append((time.perf_counter()-start))

                    #Save CSV
                    if FLAGS.output:
                        dff = pd.DataFrame.from_dict(df)
                        dff.to_csv('SavedResponse.csv')

                    #Print
                    if FLAGS.info:
                        print ("Roll: ", face.roll_angle)
                        print ("Pitch: ", face.tilt_angle)
                        print("Yaw: ", face.pan_angle)
                        print("Joy: ", face.joy_likelihood)
                        print("Sorrow: ", face.sorrow_likelihood)
                        print("Anger: ", face.anger_likelihood)
                        print("Surprise: ", face.surprise_likelihood)
                        print("Engagement Score: ", Engagement)
            if response.error.message:
                raise Exception(
                    '{}\nFor more info on error messages, check: '
                    'https://cloud.google.com/apis/design/errors'.format(
                        response.error.message))

        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frameNum  += 1
    
        






if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass