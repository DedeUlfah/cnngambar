import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import os

# Fungsi untuk memuat model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(r'C:\Users\Sinta\Documents\Documents\Dede Ulfah\cnngambar\new\model100\models\new_model.h5')
    return model

# Fungsi untuk melakukan prediksi
def predict_image(model, image):
    image = image.resize((227, 227))  # Sesuaikan dengan ukuran input model
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan batch size (1, 224, 224, 3)
    img_array /= 255.0  # Normalisasi gambar
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Fungsi untuk membaca resep dari file CSV
def get_recipe(predicted_class, class_names):
    # Baca file CSV untuk resep
    recipes_data = pd.read_csv(r'C:\Users\Sinta\Documents\Documents\Dede Ulfah\cnngambar\dataCSV\data_resep.csv', delimiter=';')

    # Ubah nama makanan menjadi huruf kecil
    food_name_lower = class_names[predicted_class].lower()
    
    # Filter berdasarkan kelas yang diprediksi, menggunakan huruf kecil
    recipe_data = recipes_data[recipes_data['food_name'].str.lower() == food_name_lower].iloc[0]
    
    # Ambil ingredient dan langkah-langkah
    ingredients = recipe_data['ingredient']
    steps = recipe_data['step']

    # Format resep untuk ditampilkan
    recipe = f"""
    **Ingredients:**
    {ingredients}

    **Steps:**
    {steps}
    """
    return recipe

# Fungsi untuk membaca nutrisi dari file CSV
def get_nutrition(predicted_class, class_names):
    # Baca file CSV untuk nutrisi
    nutrition_data = pd.read_csv(r'C:\Users\Sinta\Documents\Documents\Dede Ulfah\cnngambar\dataCSV\total_nutrition_per_food.csv')

    # Ubah nama makanan menjadi huruf kecil
    food_name_lower = class_names[predicted_class].lower()
    
    # Filter berdasarkan kelas yang diprediksi, menggunakan huruf kecil
    nutrition_info = nutrition_data[nutrition_data['food_name'].str.lower() == food_name_lower].iloc[0]
    
    nutrition = f"""
    Calories: {nutrition_info['calories']} kcal\n
    Proteins: {nutrition_info['proteins']} g\n
    Fat: {nutrition_info['fat']} g\n
    Carbohydrate: {nutrition_info['carbohydrate']} g
    """
    return nutrition

# Main function for Streamlit App
def main():
    st.title("Indonesian Food Classifier")

    # Inisialisasi model
    model = load_model()

    # Daftar nama kelas (sesuaikan dengan urutan pada model)
    class_names = ['ayam goreng', 'ayam pop', 'gulai tambusu', 'kue ape', 'kue bika ambon', 
                   'kue cenil', 'kue dadar gulung', 'kue gethuk lidri', 'kue kastangel', 
                   'kue klepon', 'kue lapis', 'kue lumpur', 'kue nagasari', 'kue pastel', 
                   'kue putri salju', 'kue risoles', 'lemper', 'lumpia', 'putu ayu',
                   'serabi solo', 'telur balado', 'telur dadar', 'wajik']

    # Unggah gambar
    uploaded_file = st.file_uploader("Upload an image of Indonesian food", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image', use_column_width=True)

        # Prediksi gambar
        st.write("Classifying...")
        predicted_class = predict_image(model, image)
        st.write(f"Prediction: {class_names[predicted_class]}")

        # Tampilkan resep
        recipe = get_recipe(predicted_class, class_names)
        st.subheader("Recipe")
        st.write(recipe)

        # Tampilkan informasi nutrisi
        nutrition = get_nutrition(predicted_class, class_names)
        st.subheader("Nutrition Information")
        st.write(nutrition)

# Run the app
if __name__ == "__main__":
    main()
