from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
img = np.array(Image.open('one.ppm').convert('L'))
plt.imshow(img)
plt.show()
