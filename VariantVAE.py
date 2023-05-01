import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Layer, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose, Input


print(tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices('CPU'))


IMG_SIZE = (256, 256)
IMG_SHAPE = IMG_SIZE + (3,)
BUFFER_SIZE = 4
BATCH_SIZE = 32
LATENT_DIM = 1024


def compute_reconstruction_loss(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_true-y_pred))

def compute_kl_loss(z_mean, z_log_var):
    return tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))

def compute_elbo_loss(reconstruction_loss, kl_loss):
    return reconstruction_loss + kl_loss


# Sampling Class for Encoder network
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

# Encoder Class
class Encoder(Model):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

        # Layers        
        self.conv_1 = Conv2D(32, (3, 3), activation="relu", strides=(2, 2), padding="same", input_shape=IMG_SHAPE, name="Conv2DLayer01")
        self.conv_2 = Conv2D(64, (3, 3), activation="relu", strides=(2, 2), padding="same", name="Conv2DLayer02")
        self.conv_3 = Conv2D(128, (3, 3), activation="relu", strides=(2, 2), padding="same", name="Conv2DLayer03")
        self.conv_4 = Conv2D(256, (3, 3), activation="relu", strides=(2, 2), padding="same", name="Conv2DLayer04")
        self.conv_5 = Conv2D(512, (3, 3), activation="relu", strides=(2, 2), padding="same", name="Conv2DLayer05")
        self.dense_1 = Dense(latent_dim, name="DenseLayer01")
        self.z_mean = Dense(latent_dim, name="z_mean")
        self.z_log_var = Dense(latent_dim, name="z_log_var")
        self.sampling = Sampling(name="Sampling")

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = Flatten()(x)
        x = self.dense_1(x)

        s_mean = []
        s_log_var = []
        for i in range(self.latent_dim):
            sub_x = Dense(8, activation="relu")(x)
            sub_x = Dense(4, activation="relu")(sub_x)
            mean = Dense(1, name=f"s_mean_{i}")(sub_x)
            log_var = Dense(1, name=f"s_log_var_{i}")(sub_x)
            s_mean.append(mean)
            s_log_var.append(log_var)

        c_mean = tf.reshape(tf.stack(s_mean, 1),(-1, self.latent_dim))
        c_log_var = tf.reshape(tf.stack(s_log_var, 1),(-1, self.latent_dim))
        z = Sampling()([c_mean, c_log_var])

        return [c_mean, c_log_var, z]

    def model(self):
        x = Input(shape=IMG_SHAPE, name="InputLayer")
        return Model(inputs=[x], outputs=self.call(x))


# Decoder Class
class Decoder(Model):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

        # Layers
        self.dense_1 = Dense((16 * 16 * 128), activation="relu", input_shape=(latent_dim, ), name="DenseLayer01")
        self.reshape_1 = Reshape((16, 16, 128), name="Reshape")
        self.conv_2d_transpose_1 = Conv2DTranspose(512, (3, 3), activation="relu", strides=(2, 2), padding="same", name="Conv2DTransposeLayer01")
        self.conv_2d_transpose_2 = Conv2DTranspose(256, (3, 3), activation="relu", strides=(2, 2), padding="same", name="Conv2DTransposeLayer02")
        self.conv_2d_transpose_3 = Conv2DTranspose(128, (3, 3), activation="relu", strides=(2, 2), padding="same", name="Conv2DTransposeLayer03")
        self.conv_2d_transpose_4 = Conv2DTranspose(64, (3, 3), activation="relu", strides=(2, 2), padding="same", name="Conv2DTransposeLayer04")
        self.conv_2d_transpose_5 = Conv2DTranspose(3, (3, 3), activation="relu", strides=(1, 1), padding="same", name="Conv2DTransposeLayer05")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.reshape_1(x)
        x = self.conv_2d_transpose_1(x)
        x = self.conv_2d_transpose_2(x)
        x = self.conv_2d_transpose_3(x)
        x = self.conv_2d_transpose_4(x)
        x = self.conv_2d_transpose_5(x)
        
        return x

    def model(self):
        x = Input(shape=(self.latent_dim,), name="InputLayer")
        return Model(inputs=[x], outputs=self.call(x))


# VAE Combined Network Class
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = keras.metrics.Mean(name="val_reconstruction_loss")
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = compute_reconstruction_loss(data, reconstruction)
            kl_loss = compute_kl_loss(z_mean, z_log_var)
            total_loss = compute_elbo_loss(reconstruction_loss, kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        
        reconstruction_loss = compute_reconstruction_loss(data, reconstruction)
        kl_loss = compute_kl_loss(z_mean, z_log_var)
        total_loss = compute_elbo_loss(reconstruction_loss, kl_loss)
        
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        
        return {
            "val_loss": self.val_total_loss_tracker.result(),
            "val_reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "val_kl_loss": self.val_kl_loss_tracker.result(),
        }
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction
    
    def model(self):
        x = Input(shape=IMG_SHAPE, name="InputLayer")
        return Model(inputs=[x], outputs=self.call(x))
    
    
encoder = Encoder(LATENT_DIM).model()
encoder.summary()

decoder = Decoder(LATENT_DIM).model()
decoder.summary()

vae_dup = VAE(encoder, decoder, name="VAE").model()
vae_dup.summary()