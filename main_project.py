import mediapipe as mp
import cv2
import numpy as np

def cangle(a, b, c):
	a = np.array(a)
	b = np.array(b)
	c = np.array(c)
	radians = np.arctan2(c[1] - b[1], c[0] - b[0] ) - np.arctan2(a[1] - b[1], a[0] - b[0])
	angle = np.abs(radians * 180.0 / np.pi) 
	if angle >180.0:
		angle =360 - angle
	return angle

mp_drawing = mp.solutions.drawing_utils #it gives all drawing utilities visualizing poses
mp_pose = mp.solutions.pose #importing pode estimstion models

cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
	while cap.isOpened(): 
		ret, frame = cap.read()
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image.flags.writeable = False
		
		# make detection
		results = pose.process(image)
		#reclor back to BGR
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		# extract landmarks
		left_count=-1
		right_count=-1
		try:
			landmarks = results.pose_landmarks.landmark
			left_shoulder =[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose. PoseLandmark.LEFT_SHOULDER.value].y] # 11 
			left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y] #13
			left_wrist =[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] #15
			left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y] # 23
			
			right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
			right_elbow =[landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
			right_wrist =[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
			right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y] 
			
			left_elbow_angle = cangle(left_shoulder, left_elbow, left_wrist)
			left_shoulder_angle = cangle(left_hip, left_shoulder, left_elbow)
			#calculate angle for right elbow and right angle 
			right_elbow_angle = cangle(right_shoulder, right_elbow, right_wrist)
			right_shoulder_angle = cangle(right_hip, right_shoulder, right_elbow)
			#visualize angle for left side
			cv2.putText(image, str(int(left_shoulder_angle)),tuple(np.multiply(left_shoulder, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
			cv2.putText (image, str(int(left_elbow_angle)),tuple(np.multiply(left_elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
			
			cv2.putText(image, str(int(left_shoulder_angle)), tuple(np.multiply(left_shoulder, [640,480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
			cv2.putText(image, str(int(left_elbow_angle)), tuple(np.multiply(left_elbow, [640,480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2,cv2.LINE_AA)
			
			#visualize angle for right side
			cv2.putText(image, str(int(right_shoulder_angle)), tuple(np.multiply(right_shoulder, [640,480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,cv2.LINE_AA)
			cv2.putText(image, str(int(right_elbow_angle)), tuple(np.multiply(right_elbow, [640,480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2,cv2.LINE_AA)
			if left_shoulder_angle<=50 or left_shoulder_angle>110: 
				left_stage="up"
				left_count = 0
			if left_shoulder_angle>75 and left_shoulder_angle <110 and left_elbow_angle>160 and left_stage== "up": # to count only when the angle
				right_stage="down"
				right_count=1
				print("rSTOP THE BUS")
			if left_shoulder_angle>100:
					left_count=0
			if right_shoulder_angle<=50 or right_shoulder_angle>110: 
				right_stage="up"
				#right_count = 0
			if right_shoulder_angle>75 and right_shoulder_angle <110 and right_elbow_angle>160 and right_stage=="up": 
				left_stage="down"
				left_count=1
				print("lSTOP THE BUS")
		except :
			print("")
		cv2.rectangle(image, (0,0), (130,50), (200,100,16), -1) 	#put data
		cv2.putText(image, 'STOP:', (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
		if left_count==1 or right_count==1: 
			cv2.putText(image, 'YES', (63, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA) 
		else:
			cv2.putText(image, 'NO', (63, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color= (255, 0, 0), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color= (255, 0, 0), thickness=1, circle_radius=1))
				 #color in BGR format here mp_drawing.DrawingSpec 
		cv2.imshow("mediapipe feed", image)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
