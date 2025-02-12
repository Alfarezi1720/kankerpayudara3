from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

# Memuat model yang sudah dioptimasi dan menggunakan 10 fitur rata-rata
model = joblib.load('gnb (2).pkl')

app = Flask(__name__)

# Rute untuk halaman Home (home.html)
@app.route('/')
def home_page():
    return render_template('home.html')

# Rute untuk halaman Klasifikasi (index.html)
@app.route('/classify')
def classify():
    return render_template('index.html')

# Rute untuk melakukan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengumpulkan input dari form untuk 10 fitur rata-rata
        features = [
            float(request.form['radius_mean']),
            float(request.form['texture_mean']),
            float(request.form['perimeter_mean']),
            float(request.form['area_mean']),
            float(request.form['smoothness_mean']),
            float(request.form['compactness_mean']),
            float(request.form['concavity_mean']),
            float(request.form['concave_points_mean']),
            float(request.form['symmetry_mean']),
            float(request.form['fractal_dimension_mean'])
        ]
        
        # Mengonversi input ke array untuk prediksi
        input_data = np.array(features).reshape(1, -1)
        
        # Melakukan prediksi menggunakan model
        prediction = model.predict(input_data)

        # Mapping hasil prediksi ke bentuk yang mudah dibaca
        result = "Malignant" if prediction[0] == 'malignant' else "Benign"
        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"Error: {str(e)}"

# Rute untuk halaman Dataset (dataset.html)
@app.route('/dataset')
def dataset_page():
    try:
        # Membaca file Excel
        dataset = pd.read_excel('wisconsin_breast_cancer_diagnostic_updated.xlsx')
        # Mengonversi dataset ke HTML table
        dataset_html = dataset.to_html(classes='table table-bordered', index=False)
        return render_template('dataset.html', dataset_table=dataset_html)
    except Exception as e:
        return f"Error: {str(e)}"
    
# Rute untuk halaman Profile (profile.html)
@app.route('/profile')
def profile_page():
    return render_template('profile.html')

if __name__ == '__main__':
    app.run(debug=True)
