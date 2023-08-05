import cv2
import numpy as np
import mediapipe as mp
import time
import math

class CHECKOUT_STUDYING :
    def __init__(self, time) :
        # hand detection
        self.mp_hands = mp.solutions.hands
        self.middle_finger_mcp_list = []
        self.no_detection_time = None
        self.not_moved_time = None
        self.previous_dot_distance = None
        
        # face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.eye_closed_time = None
        
        self.user_score = {}
        
        # set time
        self.time = time
        
    def pythagorean_theorem_dot_dist(selft, coordinate_list):
        dot_distance = math.sqrt((coordinate_list[0].x - coordinate_list[1].x)**2 + (coordinate_list[0].y - coordinate_list[1].y)**2 + (coordinate_list[0].z - coordinate_list[1].z)**2)
        return dot_distance
    
    def face_checkout(self, image) :
        with self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5) as face_detection :
            not_studying = False
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections :
                for detection in results.detections:
                    if detection.location_data:
                        box = detection.location_data.relative_bounding_box
                        ih, iw, _ = image.shape
                        x, y, w, h = int(box.xmin * iw), int(box.ymin * ih), int(box.width * iw), int(box.height * ih)
                        # Crop the face with margins
                        face_image = image[y:y+h, x:x+w]
                        face_image = cv2.resize(face_image, dsize=(500, 500), interpolation=cv2.INTER_AREA)
                        
                        with self.mp_face_mesh.FaceMesh(
                            max_num_faces=1,
                            refine_landmarks=True) as face_mesh :

                            face_image.flags.writeable = False
                            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                            results = face_mesh.process(face_image)
    
                            face_image.flags.writeable = True
                            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                            if results.multi_face_landmarks :
                                LEFT_TOP_EYELID = 159
                                LEFT_BOTTOM_EYELID = 145
                            
                                RIGHT_TOP_EYELID = 386
                                RIGHT_BOTTOM_EYELID = 374
                                
                                for face_landmarks in results.multi_face_landmarks:
                                    left_top_eyelid_coordinate = face_landmarks.landmark[LEFT_TOP_EYELID]
                                    left_bottom_eyelid_coordinate = face_landmarks.landmark[LEFT_BOTTOM_EYELID]
                                    
                                    right_top_eyelid_coordinate = face_landmarks.landmark[RIGHT_TOP_EYELID]
                                    right_bottom_eyelid_coordinate = face_landmarks.landmark[RIGHT_BOTTOM_EYELID]
                                    
                                    left_distance = self.pythagorean_theorem_dot_dist([left_top_eyelid_coordinate, left_bottom_eyelid_coordinate])
                                    right_distance = self.pythagorean_theorem_dot_dist([right_top_eyelid_coordinate, right_bottom_eyelid_coordinate])
                                    
                                    if left_distance <= 0.04 and right_distance <= 0.045 :
                                        if self.eye_closed_time is None :
                                            not_studying = True
                                        elif time.time() - self.eye_closed_time > self.time :
                                            not_studying = True
                                            self.eye_closed_time = None
                            else :
                                not_studying = True
            else :
                not_studying = True
    
        return not_studying                                    
    def start_detection(self, image_path : str) :

        image = cv2.imread(image_path)

        return self.face_checkout(image)