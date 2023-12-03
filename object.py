from imutils.video import FPS #Framespersecond
import numpy as np
import imutils
import cv2
## Set configuration parameters
use_gpu = True
live_video = False #Set to True for live video feed, False for video file
confidence_level = 0.5  # Minimum confidence level for object detection

# Initialize the FPS counter
fps = FPS().start()

# Variable to check if the video capture was successful
ret = True
# Define the classes for which the model can detect objects
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


# Generate random COLORS for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# Load pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('ssd_files/MobileNetSSD_deploy.prototxt',
                               'ssd_files/MobileNetSSD_deploy.caffemodel')

# Set backend and target to CUDA if GPU is enabled
if use_gpu:
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Open a video stream (live or from a file)
print("[INFO] accessing video stream...")
if live_video:
    vs = cv2.VideoCapture(0)
else:
    vs = cv2.VideoCapture('test-2.mp4')

# Main video processing loop
while ret:
     # Read a frame from the video source
    ret, frame = vs.read()
    # If the frame was successfully captured
    if ret:
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]

        # Prepare the frame for object detection
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Check if the confidence level is above the threshold
            if confidence > confidence_level:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw a bounding box and label around the detected object
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[idx], 1)

        # Resize the frame for display
        frame = imutils.resize(frame,height=400)
        cv2.imshow('Live detection',frame)

        # Break the loop if the 'Esc' key is pressed
        if cv2.waitKey(1)==27:
            break
         # Update the FPS counter
        fps.update()

fps.stop()

print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))