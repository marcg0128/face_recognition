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
                roi_gray = frame[y:y+h, x:x+w]
                cv2.imwrite('temp/temp.jpg',roi_gray)
                self.mark_face('temp/temp.jpg', frame, x, y, w, h)

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    self.save_face(roi_gray)
                    print('Face saved')

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def save_face(self, img):
        number = 0
        try:
            os.listdir('stored_faces')
            os.listdir('temp')
        except FileNotFoundError:
            os.mkdir('stored_faces')
            os.mkdir('temp')

        if len(os.listdir('stored_faces')) == 0:
            number = 0
        else:
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




    def mark_face(self, fn, frame, x, y, w, h):
        if len(os.listdir('stored_faces')) >= 1 :
            dfs = DeepFace.find(img_path=fn, db_path="stored_faces", enforce_detection=False)
            if dfs:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (96, 201, 91), 2)
                return True
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                return False
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            return False













recognition = FaceRecognition()
recognition.start_camera()