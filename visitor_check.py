import face_recognition
import cv2
from datetime import datetime
import numpy as np
import json

''' 
recgonize and display unique faces of visitors entering a space

@author   Jürgen Buchiger
@version  1.0 26 May 2023
'''

man_text = ("face detection and recognition algorithm for visitor recognition\n")
	
print(man_text)

# array to store all recognized faces
faces = []

# width and height of ouput frame
width = 1280
height = 960
thumbs_per_height = 4

# to save pseudo console lines
console = [
	"face detection and recognition algorithm for visitor recognition",
	"starting ai assisted facial recognition engine",
	"setting up face encoding database",
	"maximum face distance for unique visitor: 0.6",
	"processing every 4th frame",
	"starting camera",
	"rerouting camera stream",
	"stream downsample = 2",
	"dlib deep learning algorithm running..."
]





thumb_size = int(height/thumbs_per_height)

def analyse(process_nth=4, resize=2, max_distance=0.6, output_path='.', thumbs_per_height=0.25):
	video_capture = cv2.VideoCapture(0)
	framecount = 0
	frames_processed = 0
	contacts = 0
	start_time = datetime.now()
	
	font = cv2.FONT_HERSHEY_DUPLEX
	
	known_face_encodings = []

	while True:
		# Grab a single frame of video and get dimensions
		ret, fullframe = video_capture.read()
		h, w, ch = fullframe.shape

		# Only process every nth frame of video to save time
		if framecount%process_nth == 0:
			
			# reduce image size 
			h, w, ch = fullframe.shape
			frame = cv2.resize(fullframe, (int(w/resize), int(h/resize)))
		
			# Find all the faces and face encodings in the current frame of video
			face_locations = face_recognition.face_locations(frame)
			face_encodings = face_recognition.face_encodings(frame, face_locations)
		
			for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
				if len(known_face_encodings) > 0:
					# See if the face is a match for the known face(s)					
					face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
					best_match_index = np.argmin(face_distances)
					if face_distances[best_match_index] > max_distance:    # we do not know this face
						known_face_encodings.append(face_encoding)
						# extract face with some extra space
						if top*resize < 20:
							top = 0
							bottom = bottom*resize+20
							left = left*resize-20
							right = right*resize+20
						elif left*resize < 20:
							top = top*resize-20
							bottom = bottom*resize+20
							left = 0
							right = right*resize+20						
						else:
							top = top*resize-20
							bottom = bottom*resize+20
							left = left*resize-20
							right = right*resize+20
						he, wi, cha = fullframe[top:bottom, left:right].shape
						face_image = np.zeros((he,wi,cha), dtype=np.uint8)
						face_image[:,:] = fullframe[top:bottom, left:right]
						if face_image.size != 0:
							cv2.imwrite(output_path+"/contacts/{}-ID{:05}_face.png".format(start_time.strftime("%Y-%m-%dT%H_%M_%S"), contacts), face_image)
						faces.append({ 'id': contacts, 'time': datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), 'image': face_image })
						console.append("{} new visitor detected, id: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(faces)-1))
						contacts += 1
				else:
					known_face_encodings.append(face_encoding)
					# extract face with some extra space
					if top*resize < 20:
						top = 0
						bottom = bottom*resize+20
						left = left*resize-20
						right = right*resize+20
					elif left*resize < 20:
						top = top*resize-20
						bottom = bottom*resize+20
						left = 0
						right = right*resize+20						
					else:
						top = top*resize-20
						bottom = bottom*resize+20
						left = left*resize-20
						right = right*resize+20
					he, wi, cha = fullframe[top:bottom, left:right].shape
					face_image = np.zeros((he,wi,cha), dtype=np.uint8)
					face_image[:,:] = fullframe[top:bottom, left:right]
					if face_image.size != 0:
						cv2.imwrite(output_path+"/contacts/{}-ID{:05}_face.png".format(start_time.strftime("%Y-%m-%dT%H_%M_%S"), contacts), face_image)
					faces.append({ 'id': contacts, 'time': datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), 'image': face_image })
					contacts += 1

			frames_processed += 1

		# draw everything to a new frame
		newframe = np.zeros((height, width, 3), dtype=np.uint8)
		
		i=0
		for i in range(8):
			if len(faces)-i > 0:
				f = cv2.resize(faces[len(faces)-i-1]['image'], (thumb_size, thumb_size))
				newframe[thumb_size*(i%4):thumb_size*((i%4)+1), thumb_size*int(i/4):thumb_size*int(i/4+1)] = f
	
		# draw a box around the face in video stream
		for (top, right, bottom, left) in face_locations:		
			cv2.rectangle(fullframe, (left*resize, top*resize), (right*resize, bottom*resize + 14), (0, 0, 255), 2)
		
		# add the video stream to output frame
		vidframe = cv2.resize(fullframe, (width - 2*thumb_size, int(9*(width - 2*thumb_size)/16)))
		hn,wn,chn = vidframe.shape
		newframe[0:hn, 2*thumb_size:] = vidframe
		
		# write statistics
		time = datetime.now() - start_time
		print(time.seconds)
		texts = [
			"frame #{}".format(framecount),
			"unique_contacts={}".format(contacts),
			"contacts_per_second={:.1f}".format(contacts/(1+time.seconds)),
			"contact_chances_per_week={:.0f}".format(10*60*18*7*contacts/(1+time.seconds))
		]
		for i in range(len(texts)):
			cv2.putText(newframe, texts[i], (2*thumb_size+15, int(9*(width - 2*thumb_size)/16)+20+15*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
			
		for i in range(0, 9):
			cv2.putText(newframe, console[-i], (2*thumb_size+15, int(9*(width - 2*thumb_size)/16)+20+15*(i+4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))			
		
		
		print("frame #{}, unique_contacts={}, contacts_per_second={:.1f}, contact_chances_per_week={:.0f}	   ".format(framecount, contacts, contacts/(1+time.seconds), 10*60*18*7*contacts/(1+time.seconds)), end='\r')
		cv2.imshow('Visitor Check', newframe)	
		# Hit 'q' on the keyboard to quit!
		key = cv2.waitKey(1)
		if key > 0:
			if key == ord('q'):
				break
		
	print("")
	print("processed {} frames".format(frames_processed))
	print("found {} unique contact".format(contacts))

	#f = open(output_path+"/contacts.json","wt")
	#json.dump(faces, f)
	

	# Release handle to the webcam
	video_capture.release()
	cv2.destroyAllWindows()
