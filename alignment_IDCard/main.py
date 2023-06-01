import image_processing
import text_extraction
import classification

def main():
    image_path = "C:/Users/hamga/Downloads/1.jpg"

    model_path = './weights/crop.pth'
    classification_path = "./weights/model.h5"
    config_path = 'vietocr.yml'

    cropped_image = image_processing.preprocess_image(model_path, image_path)  # the cropped image is now in memory
    cropped = "./crop/cropped.jpg"
    class_label = classification.predict_image_class(cropped, classification_path)

    if class_label == 'cccd_new':
        detecto_model_path = "./weights/cccd_new_ROIs.pth"
    elif class_label == 'cccd_old':
        detecto_model_path = "./weights/cccd_old_ROIs.pth"
    elif class_label == 'cmnd_new':
        detecto_model_path = "./weights/cccd_new_ROIs.pth"

    text_extraction.extract_text(detecto_model_path, config_path, cropped_image)  # pass the cropped image directly

if __name__ == "__main__":
    main()