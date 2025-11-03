from flask import Flask, render_template, request, jsonify
import cv2, numpy as np, base64, onnxruntime as ort

app = Flask(__name__)

# Load models
face_net = cv2.dnn.readNetFromCaffe("RFB-320.prototxt", "RFB-320.caffemodel")
emotion_model = ort.InferenceSession("emotion-ferplus-8.onnx")
emotions = ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    img_bytes = base64.b64decode(data.split(",")[1])
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    (h, w) = img.shape[:2]

    # Prepare input for face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # Debug: log shape of detections
    print("Detection output shape:", detections.shape)

    # Handle both 3D and 4D outputs
    if len(detections.shape) == 4:
        detections = detections[0, 0, :, :]
    elif len(detections.shape) == 3:
        detections = detections[0, :, :]

    results = []
    for i in range(detections.shape[0]):
        confidence = detections[i, 2]
        if confidence > 0.5:
            box = detections[i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            face = img[y1:y2, x1:x2]

            if face.size > 0:
                # Preprocess face for emotion model
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (64, 64))
                gray = gray[np.newaxis, np.newaxis, :, :].astype(np.float32)

                # Run emotion recognition
                outputs = emotion_model.run(None, {'Input3': gray})
                emotion = emotions[np.argmax(outputs[0])]
                results.append({"emotion": emotion, "confidence": float(confidence)})

    return jsonify(results=results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
