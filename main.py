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

        while cap.isOpened():
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if time.time() - self.last_check > 3:  # alle 3 Sekunden ein Gesicht speichern
                for (x, y, w, h) in faces:
                    if w * h > 10000:
                        roi_gray = frame[y:y + h, x:x + w]
                        self.save_face(roi_gray)

                self.last_check = time.time()

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

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
                time.sleep(1)


            cv2.imshow('Face Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def create_avg_embedding(self, img_path, samples=5):
        embeddings = []

        for _ in range(samples):  # 5 Bilder analysieren
            emb = DeepFace.represent(img_path, enforce_detection=False, model_name='Facenet512')
            if emb:
                embeddings.append(np.array(emb[0]['embedding']))  # Speichern als NumPy-Array

        if len(embeddings) == 0:
            print("Kein gültiges Embedding erhalten.")
            return None

        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding.tolist()


    def save_face(self, img):
        if not os.path.exists('stored_faces'):
            os.mkdir('stored_faces')

        files = os.listdir('stored_faces')
        number = int(files[-1].split('.')[0].split('face')[1]) + 1 if files else 0

        path = f'stored_faces/face{number}.jpg'

        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if self.face_exist(path):
            print('Face already exists')
            os.remove(path)
        else:
            avg_embedding = self.create_avg_embedding(path)  # Durchschnitt berechnen

            if avg_embedding:
                embedding_json = json.dumps(avg_embedding)

                self.cursor.execute("INSERT INTO faces VALUES (%s, %s)",
                                    (f"face{number}", embedding_json))
                self.conn.commit()
                print(f"Gesicht gespeichert: {path}")


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