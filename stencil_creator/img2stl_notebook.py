# %% clean up
%matplotlib inline
import cv2
import numpy as np
from matplotlib import pyplot as plt

image_name = 'web.jpg'
x1,x2 = 580, 680
y1,y2 = 350, 450
img = cv2.imread(image_name, 0)
print('Shape',img.shape)

print('original')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

print('original')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[y1:y2,x1:x2])
plt.show()

kernel = np.ones((5,5),np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
print('eroded and diluted')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[y1:y2,x1:x2])
plt.show()

img = cv2.GaussianBlur(img, (15, 15), 1.0)
ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

print('smoothed')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[y1:y2,x1:x2])
plt.show()

# %%
from stl_tools import numpy2stl

numpy2stl(img, "example.stl", scale=0.05, mask_val=5., solid=True)

# %%
