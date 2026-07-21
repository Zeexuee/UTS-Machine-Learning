import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

def main():
    # 1. Memuat Dataset
    try:
        df = pd.read_csv('dataset_siap_model_final.csv', index_col='ID_Individu')
    except FileNotFoundError:
        print("Kesalahan: File 'dataset_siap_model_final.csv' tidak ditemukan.")
        return

    target_col = 'Tingkat_Kemiskinan'
    if target_col not in df.columns:
        print(f"Kesalahan: Kolom {target_col} tidak ditemukan.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. SMOTE pada data latih
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)

    # 5. Inisialisasi dan pelatihan Logistic Regression (multinomial)
    print("Melatih Logistic Regression (scikit-learn) pada data hasil SMOTE...")
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train_smote)

    # 6. Prediksi dan evaluasi
    y_pred = lr.predict(X_test_scaled)

    print("\n--- EVALUASI LOGISTIC REGRESSION (LIBRARY) ---")
    print(f"Akurasi Keseluruhan: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1,2,3,4], yticklabels=[1,2,3,4])
    plt.title('Confusion Matrix - Logistic Regression (Library)')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
