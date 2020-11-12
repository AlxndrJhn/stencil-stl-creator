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
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from stl_tools import numpy2stl
import cv2
import numpy as np

image_name = 'web.jpg'
zoom_factor = 0.3

A = np.array(Image.open(image_name).convert('LA'))
A = A[:,:,0]

kernel = np.ones((5,5),np.uint8)
A = cv2.morphologyEx(A, cv2.MORPH_OPEN, kernel)

A = cv2.GaussianBlur(A, (15, 15), 1.0)
ret,A = cv2.threshold(A,127,255,cv2.THRESH_BINARY)

A = zoom(A, zoom_factor)
A = 10*(A / A.max())

# %%
edges = cv2.Canny(np.uint8(A),1,9)
edges = 8*(edges / edges.max())

B = A-edges
B = np.clip(B, 0, 10)
numpy2stl(B, "web.stl", scale=1, mask_val=1, solid=True)

# %%
