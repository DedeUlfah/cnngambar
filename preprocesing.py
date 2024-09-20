import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Konfigurasi
image_dir = r'C:\Users\ASUS\Downloads\data_makanan'
min_size = (100, 100)  # Ukuran minimum gambar yang diizinkan
image_size = (224, 224)  # Ukuran target untuk model
num_augmented_images = 5  # Jumlah gambar augmentasi per gambar asli

# 1. Pembersihan Data
def remove_corrupted_images(image_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            try:
                img = Image.open(image_path)
                img.verify()  # Verifikasi apakah gambar valid
            except (IOError, SyntaxError):
                print(f"Removing corrupted image: {filename}")
                os.remove(image_path)

def remove_irrelevant_images(image_dir, min_size):
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            with Image.open(image_path) as img:
                if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                    print(f"Removing small image: {filename}")
                    os.remove(image_path)

remove_corrupted_images(image_dir)
remove_irrelevant_images(image_dir, min_size)

# 2. Resize dan Normalisasi
def resize_and_normalize_images(image_dir, target_dir, image_size):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)
            image = image.resize(image_size)
            image_array = np.array(image) / 255.0  # Normalisasi gambar
            image = Image.fromarray((image_array * 255).astype(np.uint8))
            image.save(os.path.join(target_dir, filename))

resize_and_normalize_images(image_dir, processed_dir, image_size)

# 3. Augmentasi Data
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_images(image_dir, target_dir, image_size, num_augmented_images):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)
            image = image.resize(image_size)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            for i, batch in enumerate(datagen.flow(image_array, batch_size=1)):
                if i >= num_augmented_images:
                    break
                augmented_image = batch[0] * 255
                augmented_image = Image.fromarray(augmented_image.astype(np.uint8))
                augmented_image.save(os.path.join(target_dir, f'augmented_{filename}_{i}.jpg'))

augment_images(processed_dir, augmented_dir, image_size, num_augmented_images)

# 4. Menambahkan Gambar Baru (misalnya gambar putih atau hitam)
def add_custom_images(target_dir, image_size, num_images=10):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for i in range(num_images):
        image_array = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)  # Gambar hitam
        image = Image.fromarray(image_array)
        image.save(os.path.join(target_dir, f'black_image_{i}.jpg'))

add_custom_images(augmented_dir, image_size)

print("Preprocessing complete.")
