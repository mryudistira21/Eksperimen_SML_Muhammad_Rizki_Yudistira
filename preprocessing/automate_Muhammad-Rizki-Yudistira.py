import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_adult_income(path_csv, test_size=0.2, random_state=42):
    """
    Preprocessing dataset Adult Income dan menyimpan hasil split ke CSV.
    """

    # Load dataset
    df = pd.read_csv(path_csv)

    # Hapus duplikat
    df = df.drop_duplicates()

    # Mapping target
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    # Hapus baris dengan "?" di kolom kategorikal
    df = df[~df.isin(["?"]).any(axis=1)]

    # Pisah fitur dan target
    X = df.drop("income", axis=1)
    y = df["income"]

    # One-hot encoding kolom kategorikal
    X_encoded = pd.get_dummies(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Normalisasi kolom numerik
    num_cols = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Simpan ke folder preprocessing
    repo_root = os.path.dirname(os.path.abspath(__file__))
    preprocessed_dir = os.path.join(repo_root, "dataset_preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    X_train.to_csv(os.path.join(preprocessed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(preprocessed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(preprocessed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(preprocessed_dir, "y_test.csv"), index=False)

    print("Dataset disimpan ke folder preprocessing.")

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(repo_root, "..", "dataset_raw", "adult.csv")
    preprocess_adult_income(path_csv=os.path.abspath(dataset_path))
