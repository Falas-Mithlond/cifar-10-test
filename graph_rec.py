import cv2
import numpy as np
from keras.models import load_model

Label = {0: 'Airplane', 1: 'Automoblie', 2: 'Bird', 3: 'Cat', 4: 'Deer',
         5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'}

img_path = 'D:/picture/deer.jpg'
img_pro = cv2.imread(img_path)
img_resize = cv2.resize(img_pro, (32, 32))
img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
img = img.astype('float32') / 255.0
img = img.reshape(1, 32, 32, 3)

model = load_model('./trained_model/vgg/model_vgg_adadelta_b128_e100.h5')
predict = np.argmax(model.predict(img))
print('Recognize as ：{}'.format(Label[predict]))