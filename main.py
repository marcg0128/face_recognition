import random

import cv2
import mysql.connector as mysql
import os
from deepface import DeepFace
import json
import time
import numpy as np
from scipy.spatial.distance import cosine
import tkinter as tk
from tkinter import simpledialog



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

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
        """)

        if not os.path.exists('frame'):
            os.mkdir('frame')

        if not os.path.exists('temp'):
            os.mkdir('temp')

        self.root = tk.Tk()
        self.root.withdraw()  # Hauptfenster verstecken



    def start_camera(self):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            faces = face_cascade.detectMultiScale(frame, 1.1, 4)
            cv2.imwrite('frame/frame.jpg', frame)

            exist, name, similarity = self.face_exist()

            print(faces)


            for (x, y, w, h) in faces:
                if exist:
                    cv2.putText(frame, f"{name.capitalize()}: {similarity*100:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                else:
                    cv2.putText(frame, "Gesicht nicht erkannt", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


                    if len(faces) == 1:
                        time.sleep(2)
                        print('Gesicht wird automatisch gespeichert...')
                        self.save_face()





            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Quitting...')
                break
            elif cv2.waitKey(1) & 0xFF == ord('s'):
                print('Saving Face...')
                cv2.putText(frame, "Gesischt wird gespeichert...", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
                name = simpledialog.askstring("Name", "Bitte gib dein Namen ein:")
                if name:
                    self.show_message(f'{name.capitalize()} wird gespeichert...')

                    self.cursor.execute("""
                                            INSERT INTO faces (name, embedding) VALUES (%s, %s)
                                        """, (name, embedding_json,))

                    self.conn.commit()

                    time.sleep(2)
                    self.clear_message()


                else:
                    self.show_message('Name nicht eingegeben')
                    time.sleep(2)
                    self.clear_message()

            for img in faces_images:
                os.remove(img)

        self.start_camera()


    def get_all_faces_emb(self):
        self.cursor.execute("""
            SELECT * FROM faces
        """)
        return self.cursor.fetchall()


    def face_exist(self, file_path='frame/frame.jpg', treshold=0.55):
        emb = DeepFace.represent(file_path, enforce_detection=False, model_name='Facenet512')

        if not emb:
            return False

        new_embedding = np.array(emb[0]['embedding'])

        for id, name, stored_emb in self.get_all_faces_emb():
            stored_embedding = np.array(json.loads(stored_emb))

            similarity = 1 - cosine(new_embedding, stored_embedding)

            if similarity > treshold:
                print(f'Face recognized: {id} with similarity {similarity:.2f}')
                return True, name, similarity

        return False, None, None

    def show_message(self, message):
        """ Zeigt eine Nachricht im Fenster an """
        self.root.deiconify()  # Fenster sichtbar machen
        self.label = tk.Label(self.root, text=message, font=("Arial", 12))
        self.label.pack(pady=20)
        self.root.update()  # UI aktualisieren

    def clear_message(self):
        """ Schließt das Fenster nach der Nachricht """
        self.label.destroy()
        self.root.withdraw()  # Fenster wieder verstecken


recognition = FaceRecognition()
recognition.start_camera()
recognition.root.mainloop()