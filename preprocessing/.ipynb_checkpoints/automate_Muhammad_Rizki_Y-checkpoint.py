{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4de4d507-1f19-4baf-8bce-6e1245e79097",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "def preprocess_creditcard(path_csv, test_size=0.2, random_state=42):\n",
    "    \"\"\"\n",
    "    Fungsi untuk preprocessing otomatis dataset credit card fraud.\n",
    "    \n",
    "    Args:\n",
    "        path_csv (str): path ke file creditcard.csv\n",
    "        test_size (float): proporsi data untuk test split\n",
    "        random_state (int): seed untuk reprodusibilitas\n",
    "\n",
    "    Returns:\n",
    "        X_train_scaled, X_test_scaled, y_train, y_test (numpy arrays)\n",
    "    \"\"\"\n",
    "    # Load dataset\n",
    "    df = pd.read_csv(path_csv)\n",
    "\n",
    "    # Hapus duplikat\n",
    "    df = df.drop_duplicates()\n",
    "\n",
    "    # Pisah fitur dan target\n",
    "    X = df.drop(\"Class\", axis=1)\n",
    "    y = df[\"Class\"]\n",
    "\n",
    "    # Scaling kolom Time dan Amount\n",
    "    scaler = StandardScaler()\n",
    "    X_scaled = X.copy()\n",
    "    cols_to_scale = [\"Time\", \"Amount\"]\n",
    "\n",
    "    X_scaled[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])\n",
    "\n",
    "    # Split data\n",
    "    X_train, X_test, y_train, y_test = train_test_split(\n",
    "        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state\n",
    "    )\n",
    "\n",
    "    return X_train, X_test, y_train, y_test\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (tf_gpu)",
   "language": "python",
   "name": "tf_gpu"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
