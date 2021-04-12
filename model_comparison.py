import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import load_model

# 绘图比较各个模型的表现
HISTORY_DIR = '.\\history\\vgg\\'
history = {}
for filename in glob.glob(HISTORY_DIR + '*.history'):
    with open(filename, 'rb') as f:
        history[filename] = pickle.load(f)

for key, val in history.items():
    print(key.replace(HISTORY_DIR, '').rstrip('.history'), val.keys())

def plot_training(history=None, metric='acc', title='Model Accuracy', loc='lower right'):
    model_list = []
    fig = plt.figure(figsize=(10, 8))
    for key, val in history.items():
        model_list.append(key.replace(HISTORY_DIR, '').rstrip('.histroy'))
        plt.plot(val[metric])

    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(model_list, loc=loc)
    # plt.savefig('./visualise_history/comparison/{}.jpg'.format(title))
    plt.show()

plot_training(history)
plot_training(history, metric='loss', title='Model Loss', loc='upper right')
plot_training(history, metric='val_acc', title='Model Accuracy(val)')
plot_training(history, metric='val_loss', title='Model Loss(val)', loc='upper right')


# 验证比较各个训练模型的准确率
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
n_classes = 10
# 数据规范化处理
X_train_norm = x_train.astype('float32') / 255.0
X_test_norm = x_test.astype('float32') / 255.0

# 标签编码
Y_train_onehot = np_utils.to_categorical(y_train, n_classes)
Y_test_onehot = np_utils.to_categorical(y_test, n_classes)

model = {}
MODEL_DIR = '.\\trained_model\\vgg\\'
for filename in glob.glob(MODEL_DIR + '*.h5'):
    model[filename] = load_model(filename)

for key, val in model.items():
    scores = model[key].evaluate(X_test_norm, Y_test_onehot)
    print('{}: {}, {}'.format(key, scores[1]))
