import cv2
import time
import imutils
import argparse
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:            
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]
            
            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)

    return boxes, confidences, classIDs

def make_prediction(net, layer_names, labels, image, confidences, threshold):
    height, width = image.shape[:2]
    
    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # Extract bounding boxes, confidences and classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)

    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    return boxes, confidences, classIDs, idxs

def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors):
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            # draw the bounding box and label on the image
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

PROTOTXT = ('/home/skuba/skuba_ws/src/skuba-vision/yolo/hand/hand-yolov3-tiny.cfg') ################################# change path to config file #############################
MODEL = ('/home/skuba/skuba_ws/src/skuba-vision/yolo/hand/hand-yolov3-tiny_last.weights') ################################# change path to weights file #############################
confidence = 0.5
threshold = 0.3
#Initialize Objects and corresponding colors which the model can detect
labels = ["hand"] ############################## put labels name #####################################
colors = np.random.uniform(0, 255, size=(len(labels), 3))
#Loading Model
print('[Status] Loading Model...')
net = cv2.dnn.readNetFromDarknet(PROTOTXT, MODEL)
# Get the ouput layer names
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
cap = cv2.VideoCapture(0)

while True: 
    rec, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, confidence, threshold)
    image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)
    cv2.imshow("cap",image)
    
    if cv2.waitKey(1) == 27: 
        break
        