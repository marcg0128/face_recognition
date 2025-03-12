import random

import cv2
import mysql.connector as mysql
import os
from deepface import DeepFace
import json
import time
import numpy as np



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

        if not os.path.exists('stored_faces'):
            os.mkdir('stored_faces')

        if not os.path.exists('temp'):
            os.mkdir('temp')

        self.last_check = time.time()

    def start_camera(self):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            faces = face_cascade.detectMultiScale(frame, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('s'):
                self.save_face()



    def capture_multiple_faces(self, samples=5):
        """
        Capture multiple captured_faces for a better analysis
        :param samples: Number of samples to capture
        """

        captured_faces = []
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)

        while len(captured_faces) < samples:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for  (x, y, w, h) in faces:
                roi_frame = frame[y:y + h, x:x + w]



                path = f'temp/face{len(captured_faces)}.jpg'
                cv2.imwrite(path, cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))

                captured_faces.append(path)
                print(f"Face {len(captured_faces)} captured")


            cv2.imshow('Face Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return captured_faces



    def create_avg_embedding(self, img_path):
        embeddings = []

        for path in img_path:  # 5 Bilder analysierenCapture multiple faces for a better analysis
            emb = DeepFace.represent(path, enforce_detection=False, model_name='Facenet512')
            if emb:
                embeddings.append(np.array(emb[0]['embedding']))  # Speichern als NumPy-Array

        if len(embeddings) == 0:
            print("Kein gültiges Embedding erhalten.")
            return None

        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding.tolist()


    def save_face(self):
        faces_images = self.capture_multiple_faces()

        if faces_images:
            avg_embedding = self.create_avg_embedding(faces_images)
            if avg_embedding:
                embedding_json = json.dumps(avg_embedding)
                name = str(random.randint(0, 1000000))

                self.cursor.execute("""
                    INSERT INTO faces (name, embedding) VALUES (%s, %s)
                """, (name, embedding_json,))

                self.conn.commit()

            for img in faces_images:
                os.remove(img)


    def get_all_faces_emb(self):
        self.cursor.execute("""
            SELECT * FROM faces
        """)
        return self.cursor.fetchall()


    def face_exist(self, fn):
        pass
        #emb = DeepFace.represent(fn, enforce_detection=False, model_name='Facenet512')

        #if not emb:
        #    return False

        #for embedding in self.get_all_faces_emb():
        #    if json.dumps(emb[0]) == embedding[1]:  # embedding[1] ist der gespeicherte JSON-Wert
        #        return True

        #return False













recognition = FaceRecognition()
recognition.start_camera()