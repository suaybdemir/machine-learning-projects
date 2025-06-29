## 🔐 Network Intrusion Detection using NSL-KDD

This project is part of a collection of machine learning applications, focusing on **detecting network intrusions** using the popular benchmark dataset **NSL-KDD**.

### 📌 Objective

Develop a machine learning pipeline that classifies network connections as either **benign** or one of several **attack categories**, using various preprocessing techniques and model evaluation methods.

---

### 📁 Dataset

- **Source:** [NSL-KDD Dataset – UNB CIC](https://www.unb.ca/cic/datasets/nsl.html)
- **Files Used:**
  - `KDDTrain+.txt`
  - `KDDTest+.txt`
- **Features:** 41 network connection attributes + 1 label (`attack_type`)
- **Attack Categories:**
  - 🛡️ `benign`
  - ⚔️ `dos` – Denial of Service
  - 🕵️ `probe` – Surveillance / Probing
  - 🧬 `u2r` – User to Root
  - 📩 `r2l` – Remote to Local

---

### 🧪 Workflow

1. **Load and clean** data
2. **Label engineering:** Map 22+ attack types into 5 general categories
3. **Preprocessing:**
   - Categorical encoding (One-hot)
   - Standardization (StandardScaler)
4. **Imbalance handling:** `SMOTE` oversampling
5. **Model training:**
   - `RandomForestClassifier`
6. **Evaluation:**
   - Confusion matrix
   - Classification error
   - Visualization with Seaborn

---

### 🚀 Sample Results

| Metric        | Value       |
|---------------|-------------|
| Accuracy      | 92.7%       |
| Error Rate    | 7.3%        |
| Balanced Data | ✅ via SMOTE |
| Top Features  | Duration, src_bytes, protocol_type, service... |

📊 Confusion Matrix & attack distribution graphs are visualized using `matplotlib` and `seaborn`.

---

### 🔧 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
