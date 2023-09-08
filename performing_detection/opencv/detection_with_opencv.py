import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_crop_and_weed(image_path):
    # Load the class labels
    labelsPath = 'C:/Users/PRAVEEN/Downloads/Crop_and_weed_detection/Crop_weed_detection_training/obj.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    # Load YOLO weights and configuration
    weightsPath = 'C:/Users/PRAVEEN/Downloads/Crop_and_weed_detection/performing_detection/data/weights/crop_weed_detection.weights'
    configPath = 'C:/Users/PRAVEEN/Downloads/Crop_and_weed_detection/Crop_weed_detection_training/crop_weed.cfg'

    # Color selection for drawing bounding boxes
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image. Check the image path.")

    (H, W) = image.shape[:2]

    # Parameters
    confi = 0.5
    thresh = 0.5
    layer_names = net.getUnconnectedOutLayersNames()

    # Resize the input image to a smaller size (adjust as needed)
    resized_image = cv2.resize(image, (416, 416))

    # Construct a blob from the resized input image
    blob = cv2.dnn.blobFromImage(resized_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform YOLO forward pass
    layerOutputs = net.forward(layer_names)

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each detection
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maximum suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert BGR image to RGB for displaying with Matplotlib
    det = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(det)
    plt.axis('off')
    plt.show()

# Example usage:
image_path = 'C:\\Users\\PRAVEEN\\Downloads\\Crop_and_weed_detection\\performing_detection\\data\images\\crop_4.jpeg'
detect_crop_and_weed(image_path)
