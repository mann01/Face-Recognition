import cv2
import numpy as np
import sqlite3
from os import listdir #for contacting to directory files
from os.path import isfile,join

def getProfile(id):
    conn=sqlite3.connect("FaceDatabase.db")
    cmd="select * from people where id="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

data_path='C:/Users/Sony/PycharmProjects/face_reg sqlite/faces2/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
#f hi me store hoga f(onlyfile) is file name
#print(onlyfiles) output list of filenames
training_data,labels = [],[]

for i,files in enumerate(onlyfiles):
    image_path=data_path+onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    training_data.append(np.asarray(images,dtype=np.uint8))
    labels.append(onlyfiles[i][4:5])
labels=np.asarray(labels,dtype=np.int32)
model =cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(training_data),np.asarray(labels))
#print(labels)
print("Trained succesfully")
#till now we have trained a face


face_classifier=cv2.CascadeClassifier('C:/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 30, 130), 2)
        # paramet img,starting points ending points colour and width
        roi = gray[y:y + h, x:x + w]  # this area cropped
        roi = cv2.resize(roi, (200, 200))
         # print(x,y,w,h)


        id,r =  model.predict(roi)
        #print(r)
        profile=getProfile(id)
        if profile !=None:

            cv2.putText(frame, str(profile[2]), (x+8,y+h+60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, str(profile[1]), (x, y + h+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face cropper', frame)

  #     if result[1] < 500:
 #          confidence = int(100*(1-(result[1])/300))
  #         display_string=str(confidence)+'% Confidence it is'
 #          cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

  #     if confidence >75:
  #         cv2.putText(image,"Unlocked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
  #         cv2.imshow('Face cropper',image)

  #     else:
  #         cv2.putText(image,"Locked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
  #         cv2.imshow('Face cropper', image)


    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()

