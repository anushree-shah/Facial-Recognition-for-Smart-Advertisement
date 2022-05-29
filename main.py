import cv2


def faceBox(faceNet, frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame,1.0,(300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()                                            #print(detection.shape) #shape is (1, 1, 200, 7)
    bboxs=[]                                                                 #list for storing boundry boxes
    for i in range(detection.shape[2]):                                      #loop through shape 
        confidence=detection[0,0,i,2]                                        #extracting confidence value
        if confidence>0.7:                                                   #threshold = 70% confidence 
            x1=int(detection[0,0,i,3]*frameWidth)                            #create rectangle to frame detected face 
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame, bboxs



#reading models for face detection
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

#reading models for age and gender detection
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel, ageProto)
genderNet=cv2.dnn.readNet(genderModel, genderProto)

#deciding parameters 
ageList = ['0-4', '5-8', ' 8-15', '15-20', '20-30', '30-45', '45-60', '60-100']
genderList = [ 'Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

video=cv2.VideoCapture(0)

padding = 20

while True:
    ret,frame=video.read()
    frame, bboxs = faceBox(faceNet,frame)               #calling function faceBox
    for bbox in bboxs:
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face,0.1,(227,227), MODEL_MEAN_VALUES, swapRB= False)

        genderNet.setInput(blob)
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]
        label="{},{}".format(gender,age)
        cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1)
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2,cv2.LINE_AA)
    cv2.imshow("Age-Gender",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
print("Age:"+age)
print("Gender:"+gender)

#using sample dataset
import csv                          
file = open('dataset.csv')          #extarcting from csv file
type(file)
csvreader = csv.reader(file)
rows = []
for row in csvreader:
        rows.append(row)

print("Suggested ads:")
for row in rows:
    if(row[0]==gender):
        if(row[1]==age):
            print(row[2])

video.release()
cv2.destroyAllWindows()
