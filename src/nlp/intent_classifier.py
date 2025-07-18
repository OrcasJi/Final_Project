import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Step 1: 读取训练数据
df = pd.read_csv("../../data/intent_training_data_extended.csv")  # 文件名根据你保存的为准
X = df["text"]
y = df["intent"]

# Step 2: 拆分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Step 3: 构建训练管道
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(solver="liblinear"))
])

# Step 4: 模型训练
pipeline.fit(X_train, y_train)

# Step 5: 模型评估
y_pred = pipeline.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Step 6: 保存模型
joblib.dump(pipeline, "../../models/expanded_intent_classifier_v4.pkl")
print("✅ Model saved as expanded_intent_classifier_v4.pkl")
