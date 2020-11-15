# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

image_name = 'web.jpg'
zoom_factor = 0.25

A = np.array(Image.open(image_name).convert('LA'))
A = A[:,:,0]

plt.imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
plt.show()

# kernel = np.ones((5,5),np.uint8)
# A = cv2.morphologyEx(A, cv2.MORPH_OPEN, kernel)

plt.imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
plt.show()

A = cv2.GaussianBlur(A, (15, 15), 1.0)
ret,A = cv2.threshold(A,127,255,cv2.THRESH_BINARY)

plt.imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
plt.show()

A = zoom(A, zoom_factor)
plt.imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
plt.show()

A = 10*(A / A.max())


# %%
import skimage.measure
import visvis as vv

verts, faces = skimage.measure.marching_cubes_classic(np.dstack((np.zeros(A.shape),A,A,np.zeros(A.shape))), 0.0)
#verts, faces, normals, values
# vv.mesh(np.fliplr(verts), faces, normals, values)
# vv.use().Run()

# %%
from stl import mesh

# Create the mesh
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = verts[f[j],:]

# %%
cube.save('testweb.stl')

# %%
