# USAGE
# python server.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --montageW 2 --montageH 2

# import the necessary packages
from imutils import build_montages
from datetime import datetime
import numpy as np
import dlib
from scipy.spatial import distance
from imutils import face_utils
import imagezmq
import argparse
import imutils
import cv2
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
#Slope of line
def Slope(a,b,c,d):
    return (d - b)/(c - a)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-mW", "--montageW", required=True, type=int,
    help="montage frame width")
ap.add_argument("-mH", "--montageH", required=True, type=int,
    help="montage frame height")
args = vars(ap.parse_args())


def sendImagesToWeb():
    thresh = 0.25
    frame_check = 20
    belt = False
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    flag=0
        # initialize the ImageHub object
    imageHub = imagezmq.ImageHub()

    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # initialize the consider set (class labels we care about and want
    # to count), the object count dictionary, and the frame  dictionary
    CONSIDER = set(["dog", "person", "car"])
    objCount = {obj: 0 for obj in CONSIDER}
    frameDict = {}

    # initialize the dictionary which will contain  information regarding
    # when a device was last active, then store the last time the check
    # was made was now
    lastActive = {}
    lastActiveCheck = datetime.now()

    # stores the estimated number of Pis, active checking period, and
    # calculates the duration seconds to wait before making a check to
    # see if a device was active
    ESTIMATED_NUM_PIS = 4
    ACTIVE_CHECK_PERIOD = 10
    ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

    # assign montage width and height so we can view all incoming frames
    # in a single "dashboard"
    mW = args["montageW"]
    mH = args["montageH"]
    print("[INFO] detecting: {}...".format(", ".join(obj for obj in
        CONSIDER)))

        # start looping over all the frames
    while True:
        # receive RPi name and frame from the RPi and acknowledge
        # the receipt
        (rpiName, frame) = imageHub.recv_image()
        frame1=frame[:,:,[0,1,2]]
        frame2=frame[:,:,[3,4,5]]
        imageHub.send_reply(b'OK')
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        

        # if a device is not in the last active dictionary then it means
        # that its a newly connected device
        if rpiName not in lastActive.keys():
            print("[INFO] receiving data from {}...".format(rpiName))

        # record the last active time for the device from which we just
        # received a frame
        lastActive[rpiName] = datetime.now()

        # resize the frame to have a maximum width of 400 pixels, then
        # grab the frame dimensions and construct a blob
        frame1 = imutils.resize(frame1, width=300)
        frame2 = imutils.resize(frame2, width=300)
        #gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        subjects = detect(frame2, 0)
        for subject in subjects:
            shape = predict(frame2, subject)
            shape = face_utils.shape_to_np(shape)#converting to NumPy Array
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame2, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame2, [rightEyeHull], -1, (0, 255, 0), 1)
            blur = cv2.blur(frame2, (1, 1))
            # Converting Image To Edges
            edges = cv2.Canny(blur, 50, 400)
            # Previous Line Slope
            ps = 0

            # Previous Line Co-ordinates
            px1, py1, px2, py2 = 0, 0, 0, 0
            # Extracting Lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/270, 30, maxLineGap = 20, minLineLength = 170)

            # extract the confidence (i.e., probability) associated with
            # the prediction
            if ear < thresh:
                flag += 1
                print (flag)
                if flag >= frame_check:
                    cv2.putText(frame2, "****************ALERT!****************", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame2, "****************ALERT!****************", (10,325),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print ("Drowsy")
            else:
                flag = 0
            # If "lines" Is Not Empty
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]# Co-ordinates Of Current Line
                    s = Slope(x1,y1,x2,y2)# Slope Of Current Line
                    # If Current Line's Slope Is Greater Than 0.7 And Less Than 2
                    if ((abs(s) > 0.7) and (abs (s) < 2)):
                        # And Previous Line's Slope Is Within 0.7 To 2
                        if((abs(ps) > 0.7) and (abs(ps) < 2)):
                            # And Both The Lines Are Not Too Far From Each Other
                            if(((abs(x1 - px1) > 5) and (abs(x2 - px2) > 5)) or ((abs(y1 - py1) > 5) and (abs(y2 - py2) > 5))):
                                # Plot The Lines On "beltframe"
                                cv2.line(frame2, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.line(frame2, (px1, py1), (px2, py2), (0, 0, 255), 3)
                                print ("Belt Detected")
                                belt = True
                    # Otherwise Current Slope Becomes Previous Slope (ps) And Current Line Becomes Previous Line (px1, py1, px2, py2)            
                    ps = s
                    px1, py1, px2, py2 = line[0]
            if belt == False:
                print("No Seatbelt detected")
        
        (h, w) = frame1.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame1, (300, 300)),
            0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # reset the object count for each object in the CONSIDER set
        objCount = {obj: 0 for obj in CONSIDER}

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # detections
                idx = int(detections[0, 0, i, 1])

                # check to see if the predicted class is in the set of
                # classes that need to be considered
                if CLASSES[idx] in CONSIDER:
                    # increment the count of the particular object
                    # detected in the frame
                    objCount[CLASSES[idx]] += 1

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the bounding box around the detected object on
                    # the frame
                    cv2.rectangle(frame1, (startX, startY), (endX, endY),
                        (255, 0, 0), 2)

        # draw the sending device name on the frame
        cv2.putText(frame1, rpiName, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # draw the object count on the frame1
        label = ", ".join("{}: {}".format(obj, count) for (obj, count) in
            objCount.items())
        cv2.putText(frame1, label, (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)

        # update the new frame in the frame dictionary
        frameDict[rpiName] = [frame1,frame2]
        #frame=np.concatenate([frame1,frame2])
        #jpg = cv2.imencode('.jpg', frame1)[1]
        #jpg2= cv2.imencode('.jpg',frame2)[1]
        framem=np.concatenate([frame1,frame2],axis=0)
        frames= cv2.imencode('.jpg',framem)[1]
        #frames=cv2.vconcat([jpg,jpg2])
        #frames=np.hstack((jpg,jpg2))
        cv2.imshow("main",framem)
        #cv2.imshow("frame2",frame2)
        
        yield b'--frames\r\nContent-Type:image/jpeg\r\n\r\n'+frames.tobytes()+b'\r\n'
       
        

        # build a montage using images in the frame dictionary
        #montages = build_montages(frameDict.values(), (w, h), (mW, mH))

        # display the montage(s) on the screen
        #for (i, montage) in enumerate(montages):
            #cv2.imshow("Home pet location monitor ({})".format(i),montage)

        # detect any kepresses
        key = cv2.waitKey(1) & 0xFF

        # if current time *minus* last time when the active device check
        # was made is greater than the threshold set then do a check
        if (datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
            # loop over all previously active devices
            for (rpiName, ts) in list(lastActive.items()):
                # remove the RPi from the last active and frame
                # dictionaries if the device hasn't been active recently
                if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
                    print("[INFO] lost connection to {}".format(rpiName))
                    lastActive.pop(rpiName)
                    frameDict.pop(rpiName)

            # set the last active check time as current time
            lastActiveCheck = datetime.now()

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
@Request.application
def application(request):
    return Response(sendImagesToWeb(), mimetype='multipart/x-mixed-replace; boundary=frames')


if __name__ == '__main__':
    run_simple('127.0.0.1', 4000, application)
    

# do a bit of cleanup
cv2.destroyAllWindows()