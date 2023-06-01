import cv2
import torch
from detecto import core, utils
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from PIL import Image

def process_ocr_results(ocr_results):
    expected_labels = ['ID', 'name', 'DOB', 'sex', 'nation', 'origin_top', 'origin', 'residence1', 'residence2', 'expiry']
    sorted_results = sorted(ocr_results.items(), key=lambda x: expected_labels.index(x[0]))

    combined_results = {}
    for label, text in sorted_results:
        if label == 'origin_top':
            combined_results['Place of origin'] = text
        elif label == 'origin':
            if 'Place of origin' in combined_results:
                combined_results['Place of origin'] += ' ' + text
            else:
                combined_results['Place of origin'] = text
        elif label == 'residence1':
            combined_results['Place of residence'] = text
        elif label == 'residence2':
            if 'Place of residence' in combined_results:
                combined_results['Place of residence'] += ' ' + text
            else:
                combined_results['Place of residence'] = text
        else:
            combined_results[label] = text
    return combined_results

def write_results_to_file(results, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for label, text in results.items():
            f.write(f"{label}: {text}\n")

def extract_text(detecto_model_path, config_path, image):  # the third argument is now an image, not a file path
    detecto_model = core.Model.load(detecto_model_path, ['ID', 'name', 'DOB', 'sex', 'nation', 'origin', 'expiry', 'residence1', 'residence2', 'origin_top'])
    config = Cfg.load_config_from_file(config_path)
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    ocr = Predictor(config)

    # Comment out the following line because you no longer read the image from a file
    # image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    predictions = detecto_model.predict(image)
    
    rois = []
    for label, box, score in zip(predictions[0], predictions[1], predictions[2]):
        if score > 0.5:  # You can adjust the threshold according to your requirements
            x1, y1, x2, y2 = box
            roi = image[int(y1):int(y2), int(x1):int(x2)]
            rois.append((label, roi))
    ocr_results = {}
    for label, roi in rois:
        # Convert the OpenCV ndarray to a PIL Image
        pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    
        # Perform OCR on the PIL Image
        ocr_result = ocr.predict(pil_roi)
        ocr_results[label] = ocr_result
    combined_results = process_ocr_results(ocr_results)
    write_results_to_file(combined_results, 'results.txt')