import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from keras import Model, Input
import time

IMG_SIZE = (256, 256)
IMG_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 1
AMOUNT_OF_SAMPLES = 20
LATENT_DIM = 1024
POS_RADIUS = 0.1
NEG_RADIUS = 0.3
POS_SAMPLE_TOWARDS_MEAN_PROB=0.95
NEG_SAMPLE_TOWARDS_MEAN_PROB=0.90
IMG_SAVING_PATH = "./SampledTriplets"
VISUALISE = True

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image, channels = 3)
    image = tf.cast(image, tf.float32)
    return image

def resize_image(image, size):
    return tf.image.resize(image, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def normalize(image):
    return image / 255 

def load_image(image_file):
    image = load(image_file)
    image = resize_image(image, IMG_SIZE)
    image = normalize(image)
    return image

def compute_reconstruction_loss(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_true-y_pred))

def compute_kl_loss(z_mean, z_log_var):
    return tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))

def compute_elbo_loss(reconstruction_loss, kl_loss):
    return reconstruction_loss + kl_loss

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

def sample_triplets(images, z_mean):
    plt.figure(figsize = (9, 3 * AMOUNT_OF_SAMPLES))
    start_time = time.time()
    for i in range(AMOUNT_OF_SAMPLES):
        if not os.path.exists(os.path.join(IMG_SAVING_PATH, f"{i}")):
            os.makedirs(os.path.join(IMG_SAVING_PATH, f"{i}"))
        
        # Anchor
        img = next(iter(images)).numpy()
        plt.imsave(os.path.join(IMG_SAVING_PATH, f"{i}/Anchor{i}.png"), img[0])

        # Reconstructed Anchor
        img_z_mean, img_z_log_var, img_z = vae.encoder.predict(img, verbose = 0)
        rec_img = np.clip(vae.decoder.predict(np.array(img_z_mean), verbose = 0), 0, 1)[0]
        plt.imsave(os.path.join(IMG_SAVING_PATH, f"{i}/Reconstructed{i}.png"), rec_img)

        # Positive & Negative Sampling
        pos_sample, neg_sample = np.zeros((LATENT_DIM)), np.zeros((LATENT_DIM))
        for dim in range(LATENT_DIM):
            global_mean = z_mean[dim]
            mean = img_z_mean[0][dim]
            log_var = img_z_log_var[0][dim]
            if np.random.rand(1)[0] <= POS_SAMPLE_TOWARDS_MEAN_PROB:
                if mean > global_mean:
                    pos_sample[dim] = mean - (POS_RADIUS * np.abs(log_var))
                else:
                    pos_sample[dim] = mean + (POS_RADIUS * np.abs(log_var))
            else:
                if mean > global_mean:
                    pos_sample[dim] = mean + (POS_RADIUS * np.abs(log_var)) 
                else:
                    pos_sample[dim] = mean - (POS_RADIUS * np.abs(log_var))
            if np.random.rand(1)[0] <= NEG_SAMPLE_TOWARDS_MEAN_PROB:
                if mean > global_mean:
                    neg_sample[dim] = mean - (NEG_RADIUS * np.abs(log_var))
                else:
                    neg_sample[dim] = mean + (NEG_RADIUS * np.abs(log_var)) 
            else:
                if mean > global_mean:
                    neg_sample[dim] = mean + (NEG_RADIUS * np.abs(log_var)) 
                else:
                    neg_sample[dim] = mean - (NEG_RADIUS * np.abs(log_var))
        pos_sample, neg_sample = np.expand_dims(pos_sample, axis=0), np.expand_dims(neg_sample, axis=0)
        pos_img = np.clip(vae.decoder.predict(np.array(pos_sample), verbose = 0), 0, 1)[0]
        plt.imsave(os.path.join(IMG_SAVING_PATH, f"{i}/Positive{i}.png"), pos_img)
        neg_img = np.clip(vae.decoder.predict(np.array(neg_sample), verbose = 0), 0, 1)[0]
        plt.imsave(os.path.join(IMG_SAVING_PATH, f"{i}/Negative{i}.png"), neg_img)
        
        # Plotting
        if VISUALISE:
            plt.subplot(AMOUNT_OF_SAMPLES, 4, i*4+1)
            plt.gca().set_title('Anchor')
            plt.imshow(img[0])
            plt.subplot(AMOUNT_OF_SAMPLES, 4, i*4+2)
            plt.gca().set_title('Anchor Reconstructed')
            plt.imshow(rec_img)
            plt.subplot(AMOUNT_OF_SAMPLES, 4, i*4+3)
            plt.gca().set_title('Positive')
            plt.imshow(pos_img)
            plt.subplot(AMOUNT_OF_SAMPLES, 4, i*4+4)
            plt.gca().set_title('Negative')
            plt.imshow(neg_img)
        
        print(f"{i+1}/{AMOUNT_OF_SAMPLES} Triplets Sampled -- Total Running Time: {np.round(time.time() - start_time, 1)}s", end='\r')
    end_time = time.time()
    print(f"\nSampling finished in {end_time - start_time}s")


sampling_images = tf.data.Dataset.list_files('./Data/Logs/Test/*.jpg')
sampling_images = sampling_images.map(load_image)
sampling_images = sampling_images.batch(BATCH_SIZE)

encoder = tf.keras.models.load_model('./Models/VAEModel/Encoder')
encoder.compile()
decoder = tf.keras.models.load_model('./Models/VAEModel/Decoder')
decoder.compile()

vae = VAE(encoder, decoder, name="VAE")
vae.compile(optimizer=keras.optimizers.Adam())

data_z_mean, data_z_log_var, data_z = vae.encoder.predict(sampling_images, verbose = 0)
data_z_mean = np.mean(data_z_mean, axis = 0)

sample_triplets(sampling_images, data_z_mean)