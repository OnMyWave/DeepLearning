import sys
sys.path.append("/Users/onmywave/Desktop/Github/DeepLearning/deep-learning-from-scratch/")
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image

#
(x_train, t_train) ,(x_test,t_test) = load_mnist(flatten= True,normalize=True,one_hot_label=True)

#
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_test[0]
img = img.reshape(28,28)
label = t_test[0]

print(label)
print(img.shape)

img_show(img)