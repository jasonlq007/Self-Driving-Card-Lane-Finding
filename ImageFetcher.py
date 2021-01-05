# fetch image

import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

url = 'https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg'
r = requests.get(url, stream=True)
img = Image.open(r.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))

# Preprocess image

img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
plt.imshow(img, cmap=plt.get_cmap('gray'))
print(img.shape)

# Reshape reshape

img = img.reshape(1, 32, 32, 1)
# Test image
print("predicted sign: " + str(model.predict_classes(img)))