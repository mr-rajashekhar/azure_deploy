import cv2
import argparse
import numpy as np
import sys
from PIL import Image
import io
import time
from flask import Flask, request, jsonify
# Raja : Added for flask

ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required=False,
                help = 'path to input image')
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    for i in net.getUnconnectedOutLayers():
     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes,COLORS):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # cv2.imwrite("images/deteced.jpg",img)


def detectObjects(im_bytes):
    inferencing_start_time = time.time()
    image= Image.open(io.BytesIO(im_bytes))
    # print(type(image_filename))
    nparr = np.frombuffer(im_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    # with open("object_detection_opencv/yolov3.txt", 'r') as f:
    with open("yolov3.txt", 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # net = cv2.dnn.readNet("object_detection_opencv/yolov3.weights", "object_detection_opencv/yolov3.cfg")
    net = cv2.dnn.readNet("yolov3_last_2000.weights", "custom_yolov3.cfg")

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            # print(detection)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.95:
                # print(confidence)
                # print(class_id)
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    text1 = {}
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
        
        # fp = open("object_detection_opencv/yolov3.txt")
        fp = open("yolov3.txt")
        for j, line in enumerate(fp):
            if j == class_ids[i]:
                text_1 = line.strip()
                if text_1 in text1:
                    text1[text_1] += 1
                else:
                    text1[text_1] = 1
        fp.close()

    # cv2.imwrite("images/deteced.jpg",image)
    # print(type(text1))
    inferencing_end_time = time.time()
    text1["inferencing_time"] = str(inferencing_end_time-inferencing_start_time) + "s"
    return text1

# Raja: Added for flask
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            im_bytes = file.read()
            # print(detectObjects(im_bytes))
            # data = {"prediction": detectObjects(im_bytes)}
            return detectObjects(im_bytes)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


# if __name__ == "__main__":
#     with open("test1.jpeg", "rb") as f:
#         im_bytes = f.read()
#     new_image_name = "received.jpeg"
#     with open(new_image_name, "wb") as new_file:
#         new_file.write(im_bytes)
#     pillow_img = Image.open(io.BytesIO(im_bytes)).convert('L')
#     print(detectObjects(im_bytes))

# Raja: Added for flask
if __name__ == "__main__":
    app.run(debug=True)

