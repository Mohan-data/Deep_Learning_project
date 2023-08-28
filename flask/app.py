#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask, render_template, request
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
import numpy as np

image_shape = (128, 128, 2)

app = Flask(__name__)

# Load model architecture from JSON file
json_file = open(r"C:\Users\velur\flask\flask\model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load model weights from H5 file
model.load_weights(r"C:\Users\velur\flask\flask\model.h5")


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get uploaded file from request
        file = request.files["file"]

        # Read and preprocess the image
        img = load_img(BytesIO(file.read()), target_size=image_shape[:2])
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make prediction using the loaded model
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        confidence = prediction[0][predicted_class_index]
        prediction_percentages = prediction[0]*100 
        # Define class labels
        class_labels = {0: "Potato Early blight", 1: "Potato Healthy", 2: "Potato Late blight"}
	

        # Get predicted and actual labels
        predicted_label = class_labels[predicted_class_index]
        for i in range(len(class_labels)):
            class_label = class_labels[i]
            percentage = prediction_percentages[i]
    		#print(f"{class_label}: {percentage:.2f}%")

        # Render the result template with the predicted output
        return render_template("result.html", predicted_label=predicted_label, confidence=confidence)

    # Render the home template for uploading files
    return render_template("home.html")


if __name__ == "__main__":
    app.run()
