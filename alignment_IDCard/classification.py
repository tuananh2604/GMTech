from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def predict_image_class(image_path, model_path):
    # Load the model
    model = load_model(model_path)

    # Define the image size
    img_size = (224, 224)

    # Class labels
    class_labels = ['cccd_new', 'cccd_old', 'cmnd_new']

    # Load the image
    img = load_img(image_path, target_size=img_size)

    # Convert the image to a numpy array
    img = img_to_array(img)

    # Expand the dimensions of the image
    img = np.expand_dims(img, axis=0)

    # Preprocess the image
    img = preprocess_input(img)

    # Predict the class of the image
    preds = model.predict(img)

    # Get the class index
    class_idx = np.argmax(preds[0])

    # Get the class label
    class_label = class_labels[class_idx]

    # Return the predicted class
    return class_label