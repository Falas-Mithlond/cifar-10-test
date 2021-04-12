from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D

def LeNet():
    myInput = Input(shape=(32, 32, 3))
    n_classes = 10

    # 第一层卷积
    conv_1 = Conv2D(filters=6, kernel_size=5, strides=1, padding='same', activation='relu',
                    name='conv_1')(myInput)
    maxpool1 = MaxPooling2D(pool_size=2, padding='same', name='maxpooling1')(conv_1)

    # 第二层卷积
    conv_2 = Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu',
                    name='conv_2')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=2, padding='same', name='maxpooling2')(conv_2)

    # 全连接层
    flat = Flatten()(maxpool2)
    allcon_1 = Dense(units=120, activation='relu')(flat)
    allcon_2 = Dense(units=84, activation='relu')(allcon_1)
    output = Dense(units=n_classes, activation='softmax')(allcon_2)

    return Model(inputs=myInput, outputs=output)