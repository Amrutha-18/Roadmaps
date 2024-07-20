import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LambdaCallback, ReduceLROnPlateau, EarlyStopping
import datetime
from tensorflow.keras.mixed_precision import set_global_policy, Policy

policy = Policy('mixed_float16')
set_global_policy(policy)

class RoadDataset(tf.keras.utils.Sequence):
    def __init__(self, root_dir, batch_size=2, patch_size=256, has_masks=True, shuffle= False, augment=False):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.has_masks = has_masks
        self.shuffle = shuffle
        self.augment = augment
        self.image_list = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith('.jpg')]
        self.mask_list = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith('.png')] if has_masks else []
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_list) * (self.get_number_of_patches() // self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_number_of_patches(self):
        return (1024 // self.patch_size) ** 2  

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, self.patch_size, self.patch_size, 3))
        y = np.empty((self.batch_size, self.patch_size, self.patch_size, 1)) if self.has_masks else None

        for i, idx in enumerate(indexes):
            img = tf.keras.preprocessing.image.load_img(self.image_list[idx % len(self.image_list)], target_size=(1024, 1024))
            img = np.array(img) / 255.0
            
            if self.has_masks:
                mask = tf.keras.preprocessing.image.load_img(self.mask_list[idx % len(self.mask_list)], target_size=(1024, 1024), color_mode='grayscale')
                mask = np.array(mask)
                mask = (mask > 128).astype(np.float32)

            for j in range(self.batch_size):
                x_start = np.random.randint(0, 1024 - self.patch_size)
                y_start = np.random.randint(0, 1024 - self.patch_size)
                X[i,] = img[x_start:x_start + self.patch_size, y_start:y_start + self.patch_size, :]

                if self.has_masks:
                    y[i,] = np.expand_dims(mask[x_start:x_start + self.patch_size, y_start:y_start + self.patch_size], axis=-1)

        return X, y if self.has_masks else X

    def __augment_image(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image

def conv_block(x, filters):
    x = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    return x

def down_block(x, filters):
    conv = conv_block(x, filters)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool

def up_block(x, skip_conn, filters):
    x = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(x), skip_conn])
    x = conv_block(x, filters)
    return x

def create_unet(input_shape, filters=64, num_classes=1):
    inputs = Input(input_shape)

    conv1, pool1 = down_block(inputs, filters)
    conv2, pool2 = down_block(pool1, filters * 2)
    conv3, pool3 = down_block(pool2, filters * 4)
    conv4, pool4 = down_block(pool3, filters * 8)

    bottleneck = conv_block(pool4, filters * 16)

    up5 = up_block(bottleneck, conv4, filters * 8)
    up6 = up_block(up5, conv3, filters * 4)
    up7 = up_block(up6, conv2, filters * 2)
    up8 = up_block(up7, conv1, filters)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid', dtype='float32')(up8)  # Set dtype to float32

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def train_model(model, train_dataset, valid_dataset, epochs=5):
    def on_epoch_begin(epoch, logs):
        print(f"Epoch {epoch + 1}/{epochs} start time: {datetime.datetime.now()}")

    def on_epoch_end(epoch, logs):
        print(f"Epoch {epoch + 1}/{epochs} end time: {datetime.datetime.now()}")
        print(f" - loss: {logs['loss']:.4f}")

    time_callback = LambdaCallback(on_epoch_begin=on_epoch_begin, on_epoch_end=on_epoch_end)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])  # Placeholder loss
    model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, callbacks=[time_callback, ReduceLROnPlateau(), EarlyStopping(patience=2)])

def visualize_predictions(model, dataset, num_images=4):
    for i in range(num_images):
        image, _ = dataset[i]
        prediction = model.predict(np.expand_dims(image, axis=0))
        prediction = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Image')
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.title('Prediction')
        plt.imshow(prediction, cmap='gray')
        plt.show()
train_dir = '/Users/amruthapullagummi/Downloads/train'
valid_dir = '/Users/amruthapullagummi/Downloads/valid'
test_dir = '/Users/amruthapullagummi/Downloads/test'

train_dataset = RoadDataset(train_dir, has_masks=True, batch_size=2, patch_size=256, augment=True)
valid_dataset = RoadDataset(valid_dir, has_masks=False, batch_size=2, patch_size=256)
test_dataset = RoadDataset(test_dir, has_masks=False, batch_size=2, patch_size=256)

input_shape = (256, 256, 3)  
model = create_unet(input_shape)

print("Training the model...")
train_model(model, train_dataset, valid_dataset, epochs=5)

print("Validation Predictions:")
visualize_predictions(model, valid_dataset, num_images=4)
print("Test Predictions:")
visualize_predictions(model, test_dataset, num_images=4)

model.save('/Users/amruthapullagummi/Downloads/road_segmentation_model.h5')
