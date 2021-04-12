from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D

def creat_model():
    myInput = Input(shape=(32, 32, 3))
    n_classes = 10

    # 第一段卷积层
    conv_11 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                     kernel_initializer='he_normal', name='conv11')(myInput)
    conv_12 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal',name='conv12')(conv_11)
    maxpool1 = MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpooling1')(conv_12)

    # 第二段卷积层
    conv_21 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                     kernel_initializer='he_normal',name='conv_21')(maxpool1)
    conv_22 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                     kernel_initializer='he_normal',name='conv_22')(conv_21)
    maxpool2 = MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpooling2')(conv_22)

    # 第三段卷积层
    conv_31 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                     kernel_initializer='he_normal',name='conv_31')(maxpool2)
    conv_32 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                     kernel_initializer='he_normal',name='conv_32')(conv_31)
    maxpool3 = MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpooling3')(conv_32)

    # 摊平，dropout，全连接
    x = Flatten()(maxpool3)
    x = Dropout(0.5)(x)
    output = Dense(units=n_classes, activation='softmax', kernel_initializer='he_normal')(x)

    return Model(inputs=myInput, outputs=output)