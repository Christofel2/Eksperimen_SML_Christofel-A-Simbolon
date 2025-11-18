from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd

def preprocess_data(data, target_column, preprocessor_path, output_path):
    #Drop Kolom name dan city serta duplikasi data
    drop_cols = ["name", "city"]
    data = data.drop(columns=drop_cols, errors="ignore")
    data = data.drop_duplicates()

    # Pisah fitur dan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Definisikan kolom numerik yang akan dipakai
    num_cols = ["income", "loan_amount", "points", "credit_score", "years_employed"]

    # Buat pipeline untuk preprocessing data numerik
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),     
        ("scaler", RobustScaler())                        
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols)
        ],
        remainder="drop"
    )

    # Set output ke pandas DataFrame jika memungkinkan
    try:
        preprocessor.set_output(transform="pandas")
    except:
        pass

    # Split data menjadi train dan test set 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Simpan preprocessor ke file
    dump(preprocessor, preprocessor_path)
    # Simpan data train yang sudah dipreproses ke file CSV
    df_train = X_train_t.copy()
    df_train[target_column] = y_train.values
    df_train.to_csv(output_path, index=False)

    print(f"[OK] Preprocessed train data saved to: {output_path}")
    print(f"[OK] Preprocessor joblib saved to:  {preprocessor_path}")

    return X_train_t, X_test_t, y_train, y_test


def inference(data, preprocessor_path):
    pre = load(preprocessor_path)
    return pre.transform(data)

## Menjalankan proses preprocessing otomatis untuk membaca dataset mentah, memprosesnya, lalu menyimpan pipeline dan data hasil preprocessing.
if __name__ == "__main__":
    data = pd.read_csv("loan_approval_raw/loan_approval.csv")
    target_column = "loan_approved"
    preprocessor_path = "preprocessing/preprocessor.joblib"
    output_path = "preprocessing/loan_approved_processed.csv"

    preprocess_data(
        data=data,
        target_column=target_column,
        preprocessor_path=preprocessor_path,
        output_path=output_path
    )

    print("Preprocessing complete!")
