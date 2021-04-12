from keras.datasets import cifar10
from keras.utils import np_utils
from keras import optimizers
from keras import losses
from keras.utils.vis_utils import plot_model
from model_vgg import creat_model

import pickle
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
n_classes = 10
EPOCH = 100
OPT = optimizers.Adam()
BATCH_SIZE = 128
LOSS = 'categorical_crossentropy'

# 数据规范化处理
X_train_norm = x_train.astype('float32') / 255.0
X_test_norm = x_test.astype('float32') / 255.0

# 标签编码
Y_train_onehot = np_utils.to_categorical(y_train, n_classes)
Y_test_onehot = np_utils.to_categorical(y_test, n_classes)

#保存模型结构
model = creat_model()
#plot_model(model, './model_constructure/VGG_16/model_VGG_16.png', show_shapes=True)
print(model.summary())

#训练模型
model.compile(loss=LOSS, optimizer=OPT, metrics=['accuracy'])
history = model.fit(X_train_norm, Y_train_onehot, batch_size=BATCH_SIZE, epochs=EPOCH,
                    verbose=2, validation_data=(X_test_norm, Y_test_onehot))

#保存模型和训练过程
model_path = './trained_model/vgg/model_VGG_adam_b128_e100_uniformini.h5'
model.save(model_path)
history_path = './history/vgg/model_VGG_adam_b128_e100_uniformini.history'
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)

# 可视化
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper right')

plt.savefig('./visualise_history/vgg/model_VGG_adam_b128_e100_uniformini.png')
plt.show()
