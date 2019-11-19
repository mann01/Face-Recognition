#input for faces
import cv2
import sqlite3
import numpy as np

def Insertorupdate(id,name):
    conn=sqlite3.connect("FaceDatabase.db")
    cmd="select id,name from people where id="+str(id)
    cursor=conn.execute(cmd)
    isrecordexist=0
    for row in cursor:
        isrecordexist=1
    if(isrecordexist==1):
        cmd = "update people set name=' " + str(name) + " ' WHERE id=" + str(id)
    else:

        #cmd="insert into people(id,name,age) values("+str(id)+","+str(name)+")"
        cmd = "INSERT INTO people(id,name) values(" + str(id) + ",' " + str(name) + " ' )"
        conn.execute(cmd)
        conn.commit()
        conn.close()

face_classifier=cv2.CascadeClassifier('C:/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
id=input('enter a id')
name=input('Enter your name')
#Age=int(input('enter age'))
Insertorupdate(id,name)

def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    #scaling factor ,neighbouring

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_faces = img[y:y+h,x:x+w]
    return cropped_faces

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret,frame = cap.read()
    if face_extractor(frame) is not None:
        count = count+1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        file_name_path = 'C:/Users/Sony/PycharmProjects/face_reg sqlite/faces2/user'+str(id)+'-'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)
        print("Reading images:",5*count)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print('face not found,Sorry :) ')
        pass
    if cv2.waitKey(1)==13 or count==20:
        break
cap.release()
cv2.destroyAllWindows()
print('Collecting samples')
