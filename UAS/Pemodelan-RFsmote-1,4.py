import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

    # 2. Train-Test Split (Rasio 80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. SMOTE (Hanya pada Data Latih)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 4. Pelatihan Model Final (Random Forest)
    print("Melatih Model Final: Random Forest dengan SMOTE...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_smote, y_train_smote)

    # 5. Prediksi dan Evaluasi
    y_pred = rf_model.predict(X_test)

    print("\n--- EVALUASI MODEL FINAL ---")
    print(f"Akurasi Keseluruhan: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Visualisasi Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['1 (Sgt Miskin)', '2 (Miskin)', '3 (Rentan)', '4 (Tdk Miskin)'],
                yticklabels=['1 (Sgt Miskin)', '2 (Miskin)', '3 (Rentan)', '4 (Tdk Miskin)'])
    plt.title('Confusion Matrix - Model Final')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')

    # 7. Visualisasi Top 10 Feature Importance dari Model Final
    importances = rf_model.feature_importances_
    df_imp = pd.DataFrame({'Fitur': X.columns, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(10)

    plt.subplot(1, 2, 2)
    sns.barplot(x='Importance', y='Fitur', data=df_imp, palette='viridis')
    plt.title('Top 10 Fitur Penentu (Final Model)')
    plt.xlabel('Tingkat Kepentingan (Skor)')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()