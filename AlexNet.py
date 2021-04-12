from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D

def AlexNet():
    myInput = Input(shape=(32, 32, 3))
    n_classes = 10

    # 第一层卷积
    conv_1 = Conv2D(filters=96, kernel_size=11, strides=4, padding='same', activation='relu',
                    name='conv_1')(myInput)
    maxpool1 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='maxpooling1')(conv_1)

    # 第二层卷积
    conv_2 = Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu',
                    name='conv_2')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='maxpooling2')(conv_2)

    # 第三层卷积
    conv_3 = Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu',
                    name='conv_3')(maxpool2)

    # 第四层卷积
    conv_4 = Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu',
                    name='conv_4')(conv_3)

    # 第五层卷积
    conv_5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                    name='conv_5')(conv_4)
    maxpool3 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='maxpooling3')(conv_5)

    flat = Flatten()(maxpool3)
    allcon_1 = Dense(units=4096, activation='relu')(flat)
    dropout_1 = Dropout(0.5)(allcon_1)
    allcon_2 = Dense(units=4096, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.5)(allcon_2)
    output = Dense(units=n_classes, activation='softmax')(dropout_2)

    return Model(inputs=myInput, outputs=output)