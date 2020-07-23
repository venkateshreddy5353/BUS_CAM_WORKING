
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import dlib
from scipy.spatial import distance
from imutils import face_utils
import cv2
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
imageHub = imagezmq.ImageHub()
def sendImagesToWeb():
    
        # start looping over all the frames
    while True:
        # receive RPi name and frame from the RPi and acknowledge
        # the receipt
        (rpiName, frame) = imageHub.recv_image()
        imageHub.send_reply(b'OK')
        frame1=frame[:,:,[0,1,2]]
        frame2=frame[:,:,[3,4,5]]
        print("[INFO] receiving data from {}...".format(rpiName))

        

        # if a device is not in the last active dictionary then it means
        # that its a newly connected device
        
            
        # record the last active time for the device from which we just
        # received a frame
        #lastActive[rpiName] = datetime.now()

        # resize the frame to have a maximum width of 400 pixels, then
        # grab the frame dimensions and construct a blob
        #frame1 = imutils.resize(frame1, width=400)

            

        #frameDict[rpiName] = frame1
        jpg = cv2.imencode('.jpg', frame1)[1]
        jpg2=cv2.imencode('.jpg',frame2)[1]

        yield b'--frame1\r\nContent-Type:image/jpeg\r\n\r\n'+jpg.tobytes()+b'\r\n'+b'--frame2\r\nContent-Type:image/jpeg\r\n\r\n'+jpg2.tobytes()+b'\r\n'
        #yield b'--frame2\r\nContent-Type:image/jpeg\r\n\r\n'+jpg.tobytes()+b'\r\n'
        #cv2.imshow("result",frame1)
        # detect any kepresses
        
@Request.application
def application(request):
    return Response(sendImagesToWeb(), mimetype='multipart/x-mixed-replace; boundary=frame1'+'multipart/x-mixed-replace; boundary=frame2')
    


if __name__ == '__main__':
    run_simple('127.0.0.1', 4000, application)

# do a bit of cleanup
cv2.destroyAllWindows()
