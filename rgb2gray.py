from PIL import Image
import matplotlib.pyplot as plt
img = Image.open('Test/Test/directTest.ppm').convert('L')
plt.imshow(img)
plt.show()