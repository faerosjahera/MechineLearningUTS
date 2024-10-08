from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load the dataset
brain_tumor_path = 'Brain Tumor.csv'
brain_tumor_df = pd.read_csv(brain_tumor_path)

# Siapkan data (ini sama dengan kode yang sudah kamu buat sebelumnya)
X = brain_tumor_df.drop(['Image', 'Class'], axis=1)
y = brain_tumor_df['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan train model Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')  # Halaman utama yang menampilkan form input

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Mengambil data dari form input
    data = request.form
    
    # Mengambil input dari form dan konversi ke tipe data yang sesuai
    Image = float(data['Image'])
    mean = float(data['Mean'])
    variance = float(data['Variance'])
    entropy = float(data['Entropy'])
    skewness = float(data['Skewness'])
    kurtosis = float(data['Kurtosis'])
    contrast = float(data['Contrast'])
    energy = float(data['Energy'])
    asm = float(data['ASM'])
    homogeneity = float(data['Homogeneity'])
    dissimilarity = float(data['Dissimilarity'])
    correlation = float(data['Correlation'])
    coarseness = float(data['Coarseness'])

    # Buat array untuk prediksi
    input_data = np.array([[Image, mean, variance, entropy, skewness, kurtosis, contrast, energy, asm, homogeneity, dissimilarity, correlation, coarseness]])

    # Prediksi menggunakan model yang sudah dilatih
    prediksi = rf_model.predict(input_data)

    # Tampilkan hasil prediksi
    if prediksi[0] == 1:
        result = "Tumor"
    else:
        result = "No Tumor"

    # Mengirimkan hasil prediksi ke halaman hasil
    return render_template('result.html', prediction=result)

# Menjalankan aplikasi Flask
if __name__ == "__main__":
    app.run(debug=True)
