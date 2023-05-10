from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler


def get_vgg16_default(input_shape=(224, 224, 3), classes=1000):
    inputs = Input(shape=input_shape, name="input_layer")
    
    # Block 1
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", name="block1_conv1")(inputs)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv3")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block5_pool")(x)

    # Classification block
    x = Flatten(name="flatten")(x)
    x = Dense(units=4096, activation="relu", name="fc1")(x)
    x = Dense(units=4096, activation="relu", name="fc2")(x)
    
    outputs = Dense(units=classes, activation="softmax", name="predictions")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="vgg16_default")
    
    return model


def get_vgg16_avg(input_shape=(224, 224, 3), classes=10):
    inputs = Input(shape=input_shape, name="input_layer")
    
    # Block 1
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", name="block1_conv1")(inputs)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv3")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block5_pool")(x)

    # Classification block
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dropout(rate=0.5, name="dropout_1")(x)
    x = Dense(units=128, activation="relu", name="fc1")(x)
    x = Dropout(rate=0.5, name="dropout_2")(x)
    
    outputs = Dense(units=classes, activation="softmax", name="predictions")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="vgg16_avg")
    
    return model