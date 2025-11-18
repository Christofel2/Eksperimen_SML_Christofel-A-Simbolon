from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd

def preprocess_data(data, target_column, preprocessor_path, output_path):


    drop_cols = ["name", "city"]
    data = data.drop(columns=drop_cols, errors="ignore")


    data = data.drop_duplicates()


    X = data.drop(columns=[target_column])
    y = data[target_column]


    num_cols = ["income", "loan_amount", "points", "credit_score", "years_employed"]

  
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


    try:
        preprocessor.set_output(transform="pandas")
    except:
        pass

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)


    dump(preprocessor, preprocessor_path)


    df_train = X_train_t.copy()
    df_train[target_column] = y_train.values
    df_train.to_csv(output_path, index=False)

    print(f"[OK] Preprocessed train data saved to: {output_path}")
    print(f"[OK] Preprocessor joblib saved to:  {preprocessor_path}")

    return X_train_t, X_test_t, y_train, y_test


def inference(data, preprocessor_path):
    pre = load(preprocessor_path)
    return pre.transform(data)


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

    print("🚀 Preprocessing complete!")
