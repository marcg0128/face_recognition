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


        for filename in os.listdir('stored_faces'):
            if filename:
                filename = filename.split('.')[0]
                try:
                    number = int(filename[17:])
                except ValueError:
                    pass
        number += 1

        fn = 'stored_faces/face' + str(number) + '.jpg'
        cv2.imwrite(fn, img)

        if self.face_exist(fn):
            print('Face already exists')
        else:

            embedding = DeepFace.represent(fn)

            embedding_json = json.dumps(embedding[0])

            self.cursor.execute("""
                INSERT INTO faces VALUES (%s, %s)
            """, (fn, embedding_json))
            self.conn.commit()


    def get_all_faces_emb(self):
        self.cursor.execute("""
            SELECT * FROM faces
        """)
        return self.cursor.fetchall()

    def face_exist(self, fn):
        emb = DeepFace.represent(fn)
        for embedding in self.get_all_faces_emb():

            embedding_json = json.dumps(emb[0])
            if embedding_json == embedding[1]:
                return True

        return False













recognition = FaceRecognition()
recognition.start_camera()