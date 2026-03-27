# 📘 Customer Churn Prediction 

## 💰 Project Overview
This project walks through a full machine‑learning pipeline for predicting customer churn using a Kaggle dataset.
It includes:
- 	Data loading and inspection
- 	Cleaning (missing values, duplicates)
- 	Encoding categorical variables
- 	Feature scaling (Standardization & Normalization)
- 	Train–test splitting
- 	Handling class imbalance using SMOTE
- 	Training a Random Forest classifier
- 	Model evaluation


### 🧰 1. Setup & Dependencies
#### Install required libraries:
```
pip install imbalanced-learn scikit-learn pandas numpy matplotlib seaborn
```
Imported libraries include:
- `pandas`,  `numpy`
- `matplotlib`,  `seaborn`
- `sklearn` (preprocessing, model selection, metrics)
- `imbalanced-learn` (SMOTE)

### 📥 2. Data Sourcing
_ The dataset is loaded from Google Drive:
```
df = pd.read_csv("/content/drive/MyDrive/JengaLabs/customer_churn_dataset-training-master.csv")
```
Initial inspection includes:
- First 5 rows
- Data types
- Shape
- Summary statistics


### 🧹 3. Data Preprocessing
#### 3.1 Missing Values
Missing values were inspected:
```
missing_values = df.isnull().sum()
```
Rows with missing values were dropped. Dropping was chosen because only a small number of rows were affected.

#### 3.2 Duplicate Values
Duplicates were identified and removed.
```
duplicate_count = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
```

#### 3.3 Label Encoding
Categorical columns were  encoded:
- `Gender`
- `Subscription Type`
- `Contract Length`

Using:
```
LabelEncoder().fit_transform()
```
Label encoding converts categorical labels into integer codes so models can process non‑numeric features.

#### 3.4 Feature Scaling
##### 3.4.1 Standardization (StandardScaler) — Technique 1
Applied to numerical columns:
```
scaler = StandardScaler()
df_standard_clean[numerical_cols] = scaler.fit_transform(...)
```
Benefits noted:
Centers features to mean 0 and scales them to unit variance… improving performance of gradient‑based and distance‑based algorithms.

##### 3.4.2 Normalization (MinMaxScaler) — Technique 2
Alternative scaling option:
```
scaler = MinMaxScaler()
df_minmax_clean[numerical_cols] = scaler.fit_transform(...)
```


### 🔀 4. Train–Test Split
Performed using:
```
train_test_split(..., stratify=y)
```
Why? 
Separating data prevents information leakage… ensures evaluation reflects true generalization.


### ⚖️ 5. Handling Class Imbalance (SMOTE)
SMOTE applied only to the training set:
```
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
```
Imbalance biases models toward the majority class… SMOTE helps mitigate this bias.


### 🤖 6. Model Training
A Random Forest Classifier was trained on the SMOTE‑balanced dataset.
```
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_res, y_train_res)
```
Predictions were made on the untouched test set.


### 📊 7. Model Evaluation
Metrics computed:
- Accuracy
- Classification report
- Confusion matrix
Example:
```
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```


### 8. Challenges Encountered
Finding the right order of steps: 
```
inspect → impute → encode → scale → split → resample → train.
```


### 🎓 9. Lessons Learned
- Dropping rows is acceptable when missing data is minimal.
- SMOTE can drastically improve model performance.


### 📁 11. Project Structure
```
├── data/
│   └── customer_churn_dataset.csv
├── notebooks/
│   └── Customer_Churn_Dataset.ipynb
├── README.md
└── LICENSE
```

## 🤝 Contributing
### 🚀 Suggested next steps and improvements
-	Factor notebook logic into testable modules under src/ and add unit tests in tests/.

### 🧭 Style and process
- Tests should import functions from  src/ rather than executing notebook cells..

Thank you for your contributions 🎉

