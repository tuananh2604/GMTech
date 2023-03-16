#import library
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import vietocr
import cv2

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

#Load weights 
config = Cfg.load_config_from_file('C:/Users/caotu/Downloads/alignment_IDCard/vietocr.yml') #Model weight C:\Users\caotu\AppData\Local\Temp\vgg_transformer.pth exsits. Ignore download!

#config['weights'] = 'C:/Users/caotu/AppData/Local/Temp/vgg_transformer.pth'
config['cnn']['pretrained']=False
config['device'] = 'cuda:0'

detector = Predictor(config)

# Load image
img = cv2.imread('C:/Users/caotu/Downloads/alignment_IDCard/crop/crop2.jpg')

# Define the ROIs
rois = [                        # rois = img[y:y+h, x:x+w]
    img[121:146, 185:585],      # Số                    x, y, w, h = 185, 121, 400, 25             
    img[165:188, 120:520],      # Name                  x, y, w, h = 120, 165, 400, 26
    img[187:209, 285:448],      # DOB                   x, y, w, h = 285, 187, 163, 22
    img[203:223, 233:278],      # Sex                   x, y, w, h = 233, 203, 45, 20
    img[207:227, 405:805],      # Nation                x, y, w, h = 405, 207, 400, 20
    img[246:265, 125:525],      # Origin                x, y, w, h = 125, 246, 400, 19
    img[275:295, 58:138],       # Expiry                x, y, w, h = 58, 275, 80, 20
]

# ROIs of Residence
roi1 = img[265:285, 338:738]    # Residence (top)       x1, y1, w1, h1 = 338, 265, 400, 20
roi2 = img[280:305, 130:430]    # Residence (bottom)    x2, y2, w2, h2 = 130, 280, 400, 25

# Convert NumPy array to PIL Image object for each ROI
pil_rois = [Image.fromarray(roi) for roi in rois]
pil_roi1 = Image.fromarray(roi1)
pil_roi2 = Image.fromarray(roi2)

# Detect words in each ROI
for i, roi in enumerate(pil_rois):
    s = detector.predict(roi)
    print(f'{["Số", "Name", "DOB", "Sex", "Nation", "Origin", "Expiry"][i]}: {s}')

s1 = detector.predict(pil_roi1)
s2 = detector.predict(pil_roi2)
result = s1 + ' ' + s2
print(f'{"Residence"}:',result)