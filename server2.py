
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

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
flag=0
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

	


def sendImagesToWeb():
	
		# initialize the ImageHub object
	imageHub = imagezmq.ImageHub()
	

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
	


		# start looping over all the frames
	while True:
		# receive RPi name and frame from the RPi and acknowledge
		# the receipt
		(rpiName, frame) = imageHub.recv_image()
		imageHub.send_reply(b'OK')
		

		# if a device is not in the last active dictionary then it means
		# that its a newly connected device
		if rpiName not in lastActive.keys():
			print("[INFO] receiving data from {}...".format(rpiName))

		# record the last active time for the device from which we just
		# received a frame
		lastActive[rpiName] = datetime.now()

		# resize the frame to have a maximum width of 400 pixels, then
		# grab the frame dimensions and construct a blob
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		subjects = detect(gray, 0)
			

		# loop over the detections
		for subject in subjects:
			shape = predict(gray, subject)
			shape = face_utils.shape_to_np(shape)#converting to NumPy Array
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			# extract the confidence (i.e., probability) associated with
			# the prediction
			if ear < thresh:
				flag += 1
				print (flag)
				if flag >= frame_check:
				    cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				    cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				    print ("Drowsy")
			else:
				flag = 0

		frameDict[rpiName] = frame
		jpg = cv2.imencode('.jpg', frame)[1]
		yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+jpg.tostring()+b'\r\n'
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
    return Response(sendImagesToWeb(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    run_simple('127.0.0.1', 4000, application)

# do a bit of cleanup
cv2.destroyAllWindows()
