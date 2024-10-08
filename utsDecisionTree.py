import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load data
brain_tumor_path = 'Brain Tumor.csv'
brain_tumor_df = pd.read_csv(brain_tumor_path)
df = pd.read_csv('Brain Tumor.csv')
print(df.isna().sum())

# Memilih kolom numerik
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Mengeluarkan kolom 'Class' dari daftar kolom numerik
numeric_cols.remove('Class')

print("Kolom numerik yang digunakan:", numeric_cols)


# Menghitung Z-Score
z_scores = np.abs(stats.zscore(df[numeric_cols]))

# Menentukan threshold
threshold = 3

# Menemukan baris yang memiliki setidaknya satu nilai yang melebihi threshold
outliers = (z_scores > threshold).any(axis=1)

# Menampilkan jumlah outlier
print(f"Jumlah outlier yang terdeteksi: {outliers.sum()}")

# Menampilkan baris yang dianggap outlier
df_outliers = df[outliers]
print("Data outlier:")
print(df_outliers)

# Menghapus baris outlier dari dataset
df_clean = df[~outliers]
print(f"Dataset setelah menghapus outlier: {df_clean.shape}")


# Menghitung ulang Z-Score setelah penanganan outlier
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

# Preparing the dataset
X = brain_tumor_df.drop(['Image', 'Class'], axis=1)  # Features
y = brain_tumor_df['Class']  # Target (Class)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and training the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Evaluating the model using confusion matrix and calculating performance metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Visualisasi Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.show()

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Visualizing the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(dt_classifier, feature_names=X.columns, class_names=["Non-Tumor", "Tumor"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Printing results
print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
