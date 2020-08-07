import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import socket
import threading
import time



gesturecon = 0.0
detectcon = 0.0
isObjectCount = 0
interval = 5

curObject = ""
prevObject = ""
prev=""
recognizedthreat = ""
mdetectobject = ""
gesturevalue=""
detectobject = "N/A"

isThreatdetected = False
isObjectdetected = False
isfinaloutput = False

isongoingobject = False
isongoinggesture = False
iswaitgest = False

def tensorproc():
    #Output
    global detectobject
    global mdetectobject

    global detectcon
    global isObjectdetected
    global curObject
    global prevObject
    global gesturevalue
    global recognizedthreat
    global isThreatdetected
    global isongoingobject
    global isongoinggesture
    global iswaitgest
    iswaitgest = False
    isongoinggesture = False
    isongoingobject = False
  
    sys.path.append("..")  # This is needed since the notebook is stored in the object_detection folder.
    
    from utils import label_map_util
    from utils import visualization_utils as vis_util

# CALLING PATHS
    MODEL_NAME = 'inference_graph'  # Name of the directory containing the object detection module we're using
    CWD_PATH = os.getcwd()  # Grab path to current working directory
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')  # Path to frozen detection graph .pb file, which contains the model that is used for object detection.
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')   # Path to label map file
    NUM_CLASSES = 5   # Number of classes the object detector can identify
    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `1`, we know that this corresponds to `knife`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    
    detection_graph = tf.Graph() # Load the Tensorflow model into memory.
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)


    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')# Number of objects detected

# Initialize VIDEO
    video = cv2.VideoCapture(0)
    ret = video.set(3,1080)
    ret = video.set(4,720)
# SETTING INITIAL TIME
    start = time.time()
    startwaitgest = time.time()
    startinit = time.time()
    startrecog = time.time()
    startconfirm = time.time()

    totalwaitgestcur = 0
    totalinit = 0
    totalrecog = 0
    totalconfirm = 0
    
    while(True):

        ret, frame = video.read()
        frame_expanded = np.expand_dims(frame, axis=0)
       
#CLASSIFICATION/DETECTION
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

#DISLAY SCORE AND BOX
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.50)  # Display Threshold

#!!!!!(VERIFICATION PURPOSE)!!!!!!!!!        
#SETTING INITIAL TIME        
        cur = time.time()
        curinit = cur

# FUNCTIONS
        totalinit = int(curinit - startinit)
        total = int(cur -start)

# FINDING OBJECTS!!
        try:
            objects = []
            for index, value in enumerate(classes[0]):         
                object_dict = {}
 # Locking Threshold
                if scores[0, index] > .90 and totalinit > 20:
                    object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                            scores[0, index]
                    objects.append(object_dict)
                    svalue = str(objects[0]).split(":")
    # detected object is ready to freeze
                    detectobject = svalue[0].strip('b{')
                    detectcon = svalue[1].strip('}')
                    print(objects[0])
                    print(detectobject)
                    print(detectcon)
                    isObjectdetected = True
                    curObject = detectobject
    # verifying........
                    if (curObject != prevObject and isongoingobject == False):
                        prevObject = curObject   
                        startconfirm = time.time() 
                                 
                    else: 
                        totalconfirm = int(cur - startconfirm)    
                        print("Confirming obeject in: ",totalconfirm)
                        total = int(cur - start)
    # displaying.......
                    if(totalconfirm == 1):     
                        position = (1000,80) 
                        displayString(frame,position,"Confirming in .")
                        mdetectobject = ""
                        gesturevalue = ""
                        recognizedthreat = ""
                        startwaitgest = time.time()
                    if(totalconfirm == 2):
                        position = (1000,80) 
                        displayString(frame,position,"Confirming in ..")
                        mdetectobject = detectobject                  
                        start = cur    
                        isongoingobject = True

                        
            if totalinit == 15:
                start = cur
            if totalinit > 15 and isongoingobject == False:
                total = int(cur - start)                  
            if(len(objects) == 0 and frame is not None and totalinit > 15):    
                totalconfirm = int(cur - startconfirm)
                         
                if(isObjectdetected == False) or (isObjectdetected == True) :
                    if(total == 1 and isObjectdetected == False and isongoingobject == False):
                        mdetectobject = ""
                        gesturevalue = ""
                        recognizedthreat = ""
                        startconfirm = cur          
                    if(total > 2 and  isongoingobject == False):
                        startconfirm = time.time()
                        isObjectdetected = False            #key
                        detectobject = ""
                        mdetectobject = "NO THREAT OBJECT RECOGNIZED"
                        
                        if total == 4:
                            startwaitgest = time.time()
                        if(total > 4 and recognizedthreat == ""):
                            totalwaitgestcur = int(cur - startwaitgest)
                           
                            if isThreatdetected == True and isongoinggesture == False:
                                totalwaitgestcur == 1
                                isongoinggesture = True
                            if(totalwaitgestcur == 1):
                                if isongoinggesture == False:
                                    gesturevalue = "NO THREAT GESTURE RECOGNIZED"
                                startrecog = time.time()
                            elif(totalwaitgestcur > 2):
                                totalrecog = int(cur-startrecog)
                                if totalrecog == 1:
                                    if isongoinggesture == False:
                                        recognizedthreat = "NO THREAT GESTURE RECOGNIZED"  
                                if totalrecog > 3:
                                    mdetectobject = ""
                                    gesturevalue = ""
                                    recognizedthreat = ""
                                    start = cur
                                    startconfirm = cur
                                    isThreatdetected = False
                    if(total > 11):

                        start = cur
                        total = 0
                        mdetectobject = ""
                        recognizedthreat = ""
                        isongoinggesture = True
                        gesturevalue = ""
                        isThreatdetected = False
                        isObjectdetected = False
                        isongoinggesture = False
                    print("status total:",total)

                            
                            

 # EXCEPTION =  Reset Out Values !!!!!
        except Exception as e:
            detectobject = ""
            isObjectdetected = False
            isongoingobject = False
            isThreatdetected = False
            print("resett-ing")

# REROUTING
        if(totalconfirm> 3 and isObjectdetected == True and isongoingobject == True):
            if totalconfirm == 4:
                startwaitgest = time.time()
            if isThreatdetected == True and isongoinggesture == False:
                totalwaitgestcur == 3
                isongoinggesture = True
            if(totalwaitgestcur >3 and totalwaitgestcur < 7):
                if isongoinggesture == False:
                    gesturevalue = "NO THREAT GESTURE RECOGNIZED"
                startrecog = time.time()

            elif(totalwaitgestcur > 3):
                totalrecog = int(cur - startrecog)
                if totalrecog == 1:
                    if isThreatdetected == False:
                        recognizedthreat = "OBJECT RECOGNIZED (" + mdetectobject + ")"
                if totalrecog > 3:
                    mdetectobject = ""
                    gesturevalue = ""
                    recognizedthreat = ""
                    startconfirm = cur
                    isongoingobject = False
                    isObjectdetected = False
                    isThreatdetected = False
                    isongoinggesture = False
                    start = cur
            if(total > 15):
                if(isongoingobject == True or isongoinggesture == True):
                    start = cur
                    mdetectobject = ""
                    recognizedthreat = ""
                    gesturevalue = ""
                    isongoinggesture = False
                    isongoingobject = False
                    isObjectdetected = False
            totalwaitgestcur = int(cur - startwaitgest) 
            print("is there a gesture count: ",totalwaitgestcur)
        
# !!!!!!!(DISPLAY VALUES)!!!!!!!!!!!!!
        if(frame is not None):          
            position = (10,80)     
# 1st line
            cv2.putText(
                frame, 
                "Recognized Threat Object:", #text
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, #font size
                (0, 0, 0, 255), #font color
                5 )#font stroke
        
            position = (10,150)
            cv2.putText(
                frame, 
                mdetectobject, #OUT1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                position, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, #font size
                (0, 0, 255, 255), #font color
                7) #font stroke
# 2nd Line
            position = (10,280)    
            cv2.putText(
                frame, 
                "Recognized Threat Gesture:", 
                position,
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, #font size
                (0, 0, 0, 255), #font color
                5) #font stroke
                           
            position = (10,350)
            cv2.putText(
                frame,
                gesturevalue.strip("b'"), #OUT2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                position, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, #font size
                (0, 0, 255, 255), #font color
                5) #font stroke
# 3rd line
            position = (10,480)
            cv2.putText(
                frame,
                "Recognized Threat:",
                position,
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, #font size
                (0, 0, 0, 255), #font color
                5) #font stroke

            position = (10,550)
            cv2.putText(
                frame, 
                recognizedthreat.strip("b'"), #OUT3!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                position, 
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                2.5, #font size
                (0, 0, 255, 255), #font color
                5) #font stroke

        dim = (640,480)
        resized =cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
        cv2.imshow('THREAT DETECTION', resized)
        if cv2.waitKey(1) == ord('q'):
            break
# OUTIDE WHILE
    video.release()
    cv2.destroyAllWindows()

def displayString(mframe,mposition,dmesg):
            cv2.putText(
                mframe, 
                dmesg, #text
                mposition,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, #font size
                (0, 0, 255, 255), #font color
                5) #font stroke
            dim = (640,480)           

def gestproc():

    global gesturecon
    global isfinaloutput
    global gesturevalue
    global recognizedthreat
    global isThreatdetected

# PREPPARING FOR UDP
    isfinaloutput = False
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('127.0.0.1', 8888)
    print('starting up on {} port {}'.format(*server_address))
    sock.bind(server_address)
    sock.listen(1)

    while True:
        print('waiting for a connection')
        connection, client_address = sock.accept()
        try:
            print('connection from', client_address)

# PROGRAM COMMUNICATION
            while True:
                data = connection.recv(30)         
                if data:             
                    testval = '{!r}'.format(data)
                    mdata = testval.split(",")
# PROGRAM CONDITION


                    if mdetectobject == "":
                        gesturevalue = ""
                        recognizedthreat = ""
                    elif isObjectdetected == True and mdetectobject != "NO THREAT OBJECT RECOGNIZED" and mdetectobject != "" and isThreatdetected is False:
                        isThreatdetected = True
                        gesturevalue = mdata[0][2:]
                        if gesturevalue == "Aiming" or gesturevalue == "Throwing":
                            recognizedthreat = str(mdata[0])+' a '+str(mdetectobject)
                        elif gesturevalue == "Swing" and mdetectobject != "grenade":
                            recognizedthreat = str(mdata[0])+'ing a '+str(mdetectobject)
                        elif gesturevalue == "Stabbing" and mdetectobject != "grenade":
                            recognizedthreat = str(mdata[0])+' with a '+str(mdetectobject)

                    elif (detectobject == "N/A" or detectobject == "" or mdetectobject == ""):

                        isThreatdetected = True
                        if "Push" in mdata[0]:
                            gesturevalue = mdata[0]       
                            recognizedthreat = "Pushing"
                            isfinaloutput = True
                            print("Push")
                            
                        elif "Punch" in mdata[0]:
                            gesturevalue = mdata[0]                  
                            recognizedthreat = "Punching"
                            isfinaloutput = True

                           
                        elif "Slapping" in mdata[0]:
                            gesturevalue = mdata[0] 
                            recognizedthreat = "Slapping"
                            isfinaloutput = True

                        else:
                            gesturevalue = ""
                            isThreatdetected = False
                            recognizedthreat = ""
      
        finally:
            connection.close()

class myThread1(threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
      print ("Starting " + self.name)
      tensorproc()
      print ("Exiting " + self.name)

class myThread2(threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
      print ("Starting " + self.name)
      gestproc()
      print ("Exiting " + self.name)

if __name__ == '__main__':
    thread2 = myThread2(2, "Thread-2", 1)
    thread2.start()
    tensorproc()

