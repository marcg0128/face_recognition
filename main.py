import cv2
import mysql.connector as mysql
import os
from deepface import DeepFace
import json
import time



class FaceRecognition:
    def __init__(self):
        self.conn = mysql.connect(
            host="localhost",
            port="3000",
            user="root",
            passwd="root",
            database="face_recognition"
        )
        self.cursor = self.conn.cursor()




    def start_camera(self):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = frame[y:y+h, x:x+w]
                self.save_face(roi_gray)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def save_face(self, img):
        number = 0
        try:
            os.listdir('stored_faces')
        except FileNotFoundError:
            os.mkdir('stored_faces')


        file = os.listdir('stored_faces')[-1]
        print(file)
        if file:
            filename = file.split('.')[0]
            try:
                number = int(filename[4:])+1
            except ValueError:
                pass


        fn = 'stored_faces/face' + str(number) + '.jpg'
        cv2.imwrite(fn, img)
        self.face_exist(fn)




    def face_exist(self, fn):
        dfs = DeepFace.find(img_path=fn, db_path="stored_faces", enforce_detection=False)
        print(dfs)













recognition = FaceRecognition()
recognition.start_camera()