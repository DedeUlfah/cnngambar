import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Path ke folder dataset
dataset_path = r'C:\Users\ASUS\Downloads\data_makanan'

# ImageDataGenerator untuk augmentasi data
datagen = ImageDataGenerator(
    rescale=1.0/255.0,                # Normalisasi nilai piksel menjadi [0, 1]
    rotation_range=20,                # Rotasi gambar secara acak hingga 20 derajat
    width_shift_range=0.2,            # Pergeseran horizontal acak sebesar 20% dari lebar gambar
    height_shift_range=0.2,           # Pergeseran vertikal acak sebesar 20% dari tinggi gambar
    shear_range=0.2,                  # Transformasi shearing gambar
    zoom_range=0.2,                   # Zoom in atau zoom out acak sebesar 20%
    horizontal_flip=True,             # Membalik gambar secara horizontal
    fill_mode='nearest'               # Mengisi piksel kosong yang muncul akibat augmentasi
)

# Fungsi untuk memuat dan memeriksa apakah gambar memiliki 3 channel (RGB)
def load_and_preprocess_image(image_path):
    # Muat gambar
    img = tf.keras.preprocessing.image.load_img(image_path)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Periksa apakah gambar memiliki 3 channel (RGB)
    if img_array.shape[2] != 3:
        img_array = tf.image.grayscale_to_rgb(img_array)  # Mengonversi gambar grayscale ke RGB
    
    return img_array

# Generator untuk memuat gambar dari folder dataset
def image_generator(datagen, dataset_path, target_size=(224, 224), batch_size=32, class_mode='categorical'):
    generator = datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,   # Mengubah ukuran gambar menjadi 224x224
        batch_size=batch_size,
        class_mode=class_mode,     # Menggunakan klasifikasi kategorikal
        shuffle=True
    )
    return generator

# Contoh penggunaan generator
train_generator = image_generator(datagen, dataset_path)
