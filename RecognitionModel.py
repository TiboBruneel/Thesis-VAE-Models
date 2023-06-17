import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import GlobalMaxPooling2D, Dense, Lambda, Input


IMG_SIZE = (256, 256)
IMG_SHAPE = IMG_SIZE + (3,)


class Recognition(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # EfficientNet model
        self.efficient_net_model = tf.keras.applications.EfficientNetB0(input_shape=IMG_SHAPE, include_top=False)
        
        self.efficient_net_model.trainable = True
        if self.efficient_net_model.trainable:
            for layer in self.efficient_net_model.layers[:0]:
                layer.trainable =  False

        # Layers                
        self.max_pool = GlobalMaxPooling2D(name="MaxPoolingLayer")
        self.dense_1 = Dense(1024, dtype='float32', name="DenseLayer01")
        self.lambda_1 = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1), name="OutputLayer", dtype='float32')

    def call(self, inputs):
        processed_inputs = keras.applications.efficientnet.preprocess_input(inputs)
        
        x = self.efficient_net_model(processed_inputs)
        x = self.max_pool(x)
        x = self.dense_1(x)
        x = self.lambda_1(x)
        
        return x

    def model(self):
        x = Input(shape=IMG_SHAPE, name="InputLayer")
        return Model(inputs=[x], outputs=self.call(x))
    
    
recognition = Recognition().model()
recognition.summary()
