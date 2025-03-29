from flask import Flask, render_template, request, redirect, url_for
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os
import pyresearch

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("last.pt")  # Segmentation model
names = model.model.names  # Class names from the model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'images' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('images')
    if not files:
        return redirect(request.url)

    result_paths = []

    for file in files:
        if file.filename == "":
            continue

        # Save the file in the upload folder
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Read the image & process it
        image = cv2.imread(filepath)
        if image is None:
            continue

        results = model.predict(image)
        annotator = Annotator(image, line_width=2)
        detected_classes = set()  # Classes detected in this specific image

        # Handling Predictions
        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()  # Class IDs
            masks = results[0].masks.xy  # Segmentation masks

            # Annotating the image
            for mask, cls in zip(masks, clss):
                color = colors(int(cls), True)
                txt_color = annotator.get_txt_color(color)
                annotator.box_label(results[0].boxes.xyxy[0], label=names[int(cls)], color=color)
                detected_classes.add(names[int(cls)])

        # Save the annotated image
        output_path = os.path.join(RESULT_FOLDER, 'result_' + file.filename)
        cv2.imwrite(output_path, image)
        result_paths.append({
            'input': filepath,
            'output': output_path,
            'classes': list(detected_classes)  # List of classes for this image
        })

    return render_template('result.html', result_paths=result_paths)

if __name__ == '__main__':
    app.run(debug=True)