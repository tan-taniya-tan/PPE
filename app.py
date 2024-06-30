import argparse
import io
import os
from PIL import Image
import cv2
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Ensure directories exist
UPLOAD_FOLDER = 'static/uploads'
DETECT_FOLDER = 'runs/detect'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(DETECT_FOLDER):
    os.makedirs(DETECT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/hi")
def home():
    return render_template('intropage.html')

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(filepath)

            global imgpath
            imgpath = filepath

            file_extension = f.filename.rsplit('.', 1)[1].lower()
            if file_extension in ['jpg', 'jpeg', 'png']:
                try:
                    print(f"Reading the image from {filepath}")
                    img = cv2.imread(filepath)
                    frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()

                    print("Image successfully read and encoded")
                    image = Image.open(io.BytesIO(frame))
                    yolo = YOLO('best.pt')  # Ensure best.pt is in the correct path

                    # Ensure the model saves to the correct directory
                    print(f"Running YOLO detection on {filepath}")
                    yolo_results = yolo.predict(source=image, save=True, project=DETECT_FOLDER, name='latest_detect')

                    print("YOLO detection complete")

                    # Print the yolo_results to see if detections were made
                    print(f"YOLO Results: {yolo_results}")

                    # Check if the model made any detections and saved the result
                    if yolo_results and yolo_results[0].boxes:
                        print(f"Detections: {yolo_results[0].boxes}")
                    else:
                        print("No detections were made by the YOLO model")

                    # Print contents of the detect folder for debugging
                    print(f"Contents of '{DETECT_FOLDER}': {os.listdir(DETECT_FOLDER)}")

                    # Find the latest subfolder created by YOLO and get the processed image filename
                    folder_path = os.path.join(DETECT_FOLDER, 'latest_detect')
                    print(f"Looking in folder: {folder_path}")
                    
                    if os.path.exists(folder_path):
                        print(f"Subfolder contents: {os.listdir(folder_path)}")
                    else:
                        print(f"Subfolder {folder_path} does not exist")
                        return "No processed images found", 404

                    processed_images = os.listdir(folder_path)

                    if not processed_images:
                        print(f"No processed images found in the folder: {folder_path}")
                        return "No processed images found in the folder", 404

                    processed_image_filename = processed_images[0]
                    print(f"Processed image found: {processed_image_filename}")
                    return redirect(url_for('display', filename=processed_image_filename))
                
                except Exception as e:
                    print(f"An error occurred during the YOLO detection process: {e}")
                    return f"An error occurred: {e}", 500

            else:
                print(f"Unsupported file format: {file_extension}")
                return "Unsupported file format. Please upload JPG, JPEG, or PNG files."

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def display(filename):
    print(f"Displaying image: {filename}")
    return render_template('result.html', filename=filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLO models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(debug=True, host='0.0.0.0', port=args.port)
