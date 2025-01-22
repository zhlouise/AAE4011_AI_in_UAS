import torch
from PIL import Image
import matplotlib.pyplot as plt

model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)

img_path = 'VisDrone Images/0000001_02999_d_0000005.jpg'
img = Image.open(img_path)
plt.imshow(img)
plt.axis('off')

plt.show() # Reasoning, get detection results
results = model(img) # Prediction
results.show() # Show the results
