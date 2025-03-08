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
        try:
            os.listdir('stored_faces')
        except FileNotFoundError:
            os.mkdir('stored_faces')

        number = 0
        for filename in os.listdir('stored_faces'):
            if filename.startswith('face') and filename.endswith('.jpg'):
                try:
                    n = int(filename[4:])
                    print(n)
                    if n >= number:
                        number = n + 1
                        print(number)
                except ValueError:
                    pass

        fn = 'stored_faces/face' + str(number) + '.jpg'
        name = 'face' + str(number)
        cv2.imwrite(fn, img)


        if self.face_exist(fn):
            print('Face already exists')
        else:

            embedding = DeepFace.represent(fn, enforce_detection=False)

            embedding_json = json.dumps(embedding[0])

            #self.cursor.execute("""
            #    INSERT INTO faces VALUES (%s, %s)
            #""", (name, embedding_json))
            #self.conn.commit()


    def get_all_faces_emb(self):
        self.cursor.execute("""
            SELECT * FROM faces
        """)
        return self.cursor.fetchall()

    def face_exist(self, fn):
        emb = DeepFace.represent(fn, enforce_detection=False)
        for embedding in self.get_all_faces_emb():

            embedding_json = json.dumps(emb[0])
            if embedding_json == embedding[1]:
                return True

        return False













recognition = FaceRecognition()
recognition.start_camera()