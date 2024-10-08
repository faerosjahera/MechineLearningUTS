import pandas as pd
from scipy import stats
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
brain_tumor_path = 'Brain Tumor.csv'
brain_tumor_df = pd.read_csv(brain_tumor_path)
df = pd.read_csv('Brain Tumor.csv')
print(df.isna().sum())

# Memilih kolom numerik
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Class')

print("Kolom numerik yang digunakan:", numeric_cols)


z_scores = np.abs(stats.zscore(df[numeric_cols]))
threshold = 100
outliers = (z_scores > threshold).any(axis=1)

# Menampilkan jumlah outlier
print(f"Jumlah outlier yang terdeteksi: {outliers.sum()}")

# menampilkan outlier
df_outliers = df[outliers]
print("Data outlier:")
print(df_outliers)

# hapus outlier
df_clean = df[~outliers]
print(f"Dataset setelah menghapus outlier: {df_clean.shape}")

# ngecek outlier lagi
z_scores_clean = np.abs(stats.zscore(df_clean[numeric_cols]))
outliers_clean = (z_scores_clean > threshold).any(axis=1)
print(f"Jumlah outlier setelah penanganan: {outliers_clean.sum()}")

# Z-Score
print("\nNormalisasi Z Score:")

df['Mean'] = (df['Mean'] - df['Mean'].mean()) / df['Mean'].std()
df['Variance_Z'] = (df['Variance'] - df['Variance'].mean()) / df['Variance'].std()
df['Entropy'] = (df['Entropy'] - df['Entropy'].mean()) / df['Entropy'].std()
df['Skewness'] = (df['Skewness'] - df['Skewness'].mean()) / df['Skewness'].std()
df['Kurtosis'] = (df['Kurtosis'] - df['Kurtosis'].mean()) / df['Kurtosis'].std()
df['Contrast'] = (df['Contrast'] - df['Contrast'].mean()) / df['Contrast'].std()
df['Energy'] = (df['Energy'] - df['Energy'].mean()) / df['Energy'].std()
df['ASM'] = (df['ASM'] - df['ASM'].mean()) / df['ASM'].std()
df['Homogeneity'] = (df['Homogeneity'] - df['Homogeneity'].mean()) / df['Homogeneity'].std()
df['Dissimilarity'] = (df['Dissimilarity'] - df['Dissimilarity'].mean()) / df['Dissimilarity'].std()
df['Correlation'] = (df['Correlation'] - df['Correlation'].mean()) / df['Correlation'].std()
df['Coarseness'] = (df['Coarseness'] - df['Coarseness'].mean()) / df['Coarseness'].std()

print(df[['Mean','Variance', 'Entropy', 'Skewness', 'Kurtosis', 'Contrast', 'Energy', 'ASM', 'Homogeneity', 'Dissimilarity', 'Correlation', 'Coarseness']])


# siapkan data
X = brain_tumor_df.drop(['Image', 'Class'], axis=1)  
y = brain_tumor_df['Class']  

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

# Evaluasi dan confusion matrix (dijelasin ya)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred) 
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Figure confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Random Forest')
plt.show()

# Figure random forest
estimator = rf_model.estimators_[0]
fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(estimator, 
                   feature_names=X.columns,  
                   class_names=["No Tumor", "Tumor"],
                   filled=True)
plt.savefig('random_forest_tree.png')
plt.show()

# Print the metrics
print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# Fungsi untuk menerima input data baru dari user
def input_data_baru():
    print("Masukkan data baru:")

    Image = input("Image ID: ")
    Mean = float(input("Mean: "))
    Variance = float(input("Variance: "))
    Entropy = float(input("Entropy: "))
    Skewness = float(input("Skewness: "))
    Kurtosis = float(input("Kurtosis: "))
    Contrast = float(input("Contrast: "))
    Energy = float(input("Energy: "))
    ASM = float(input("ASM: "))
    Homogeneity = float(input("Homogeneity: "))
    Dissimilarity = float(input("Dissimilarity: "))
    Correlation = float(input("Correlation: "))
    Coarseness = float(input("Coarseness: "))
    
    # Buat array yang berisi input dari user
    data_baru = np.array([[Image, Mean, Variance, Entropy, Skewness, Kurtosis, Contrast, Energy, ASM, Homogeneity, Dissimilarity, Correlation, Coarseness]])
    return data_baru

# Memanggil fungsi untuk input data baru
data_baru = input_data_baru()

# Melakukan prediksi dengan model Random Forest
prediksi_baru = rf_model.predict(data_baru)

# Menampilkan hasil prediksi
if prediksi_baru == 1:
    print("Hasil prediksi: Tumor")
else:
    print("Hasil prediksi: No Tumor")