from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D

def vgg_16():
    myInput = Input(shape=(32, 32, 3))
    n_classes = 10

    # 第一段卷积层
    conv11 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',
                    name='conv11')(myInput)
    conv12 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',
                    name='conv12')(conv11)
    maxpool1 = MaxPooling2D(pool_size=2, strides=2, name='maxpool1')(conv12)

    # 第二段卷积层
    conv21 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu',
                    name='conv21')(maxpool1)
    conv22 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu',
                    name='conv22')(conv21)
    maxpool2 = MaxPooling2D(pool_size=2, strides=2, name='maxpool2')(conv22)

    # 第三段卷积层
    conv31 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                    name='conv31')(maxpool2)
    conv32 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                    name='conv32')(conv31)
    conv33 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                    name='conv33')(conv32)
    maxpool3 = MaxPooling2D(pool_size=2, strides=2, name='maxpool3')(conv33)

    # 第四段卷积层
    conv41 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                    name='conv41')(maxpool3)
    conv42 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                    name='conv42')(conv41)
    conv43 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                    name='conv43')(conv42)
    maxpool4 = MaxPooling2D(pool_size=2, strides=2, name='maxpool4')(conv43)

    # 第五段卷积层
    conv51 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                    name='conv51')(maxpool4)
    conv52 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                    name='conv52')(conv51)
    conv53 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                    name='conv53')(conv52)
    maxpool5 = MaxPooling2D(pool_size=2, strides=2, name='maxpool5')(conv53)

    # 全连接层
    flat = Flatten(name='Flatten')(maxpool5)
    allcon1 = Dense(units=4096, activation='relu', name='fc1')(flat)
    allcon2 = Dense(units=4096, activation='relu', name='fc2')(allcon1)
    output = Dense(units=n_classes, activation='softmax', name='output')(allcon2)

    return Model(inputs=myInput, outputs=output)