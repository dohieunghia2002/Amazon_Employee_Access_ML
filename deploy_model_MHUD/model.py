import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import joblib

# Read original dataset
df = pd.read_csv("amazon_after_process_no_scale.csv")

#split dataset
X = df.drop('target', axis = 1)
y = df['target']

#get all feature names
feature_names = df.columns[0:9].tolist()

min_values = X.min()
max_values = X.max()


def RandomForest():
    k_fold = KFold(n_splits=15, shuffle=True, random_state=42)
    f1_scores = []
    models = []  # To store the trained models for each fold
    scalers = []

    for fold_idx, (train_indices, test_indices) in enumerate(k_fold.split(X)):
        # Chia dữ liệu thành tập huấn luyện và tập kiểm thử
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Khởi tạo mô hình
        model = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=100,  min_samples_split=3, min_samples_leaf=6, random_state=42)

        # Huấn luyện mô hình
        model.fit(X_train_scaled, y_train)

        # Dự đoán trên tập kiểm thử
        y_pred = model.predict(X_test_scaled)

        # Đánh giá độ chính xác
        f1 = f1_score(y_test, y_pred, average='macro')

        f1_scores.append(f1)
        models.append(model)
        scalers.append(scaler)

    max_f1, index = max((value, index) for index, value in enumerate(f1_scores))
    print("The best model is", index, "with F1 score is", max_f1)

    # save the trained model to disk
    final_model = models[index]
    model_filename = "rand_forest_model.sav"
    joblib.dump(scalers[index], 'scale_func.sav')
    joblib.dump(final_model, model_filename)


RandomForest()