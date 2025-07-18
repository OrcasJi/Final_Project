
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes):
    """可视化混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('../../reports/confusion_matrix.png')
    plt.close()

def main():
    # 1. 数据加载与增强
    df = pd.read_csv("../../data/intent_training_data_extended.csv")

    # 数据增强：添加小写版本
    augmented = pd.DataFrame({
        "text": df["text"].apply(str.lower),
        "intent": df["intent"]
    })
    df = pd.concat([df, augmented]).sample(frac=1).reset_index(drop=True)

    # 2. 拆分数据集（分层抽样保持类别分布）
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["intent"], 
        test_size=0.15, 
        random_state=42,
        stratify=df["intent"]  # 保持类别比例
    )

    # 3. 构建优化后的管道
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),  # 使用unigram和bigram
            max_features=5000,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )),
        ("clf", LogisticRegression(
            solver='saga',  # 更好的优化器
            class_weight='balanced',  # 处理类别不平衡
            max_iter=500,
            random_state=42
        ))
    ])

    # 4. 参数网格搜索
    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.1, 1, 10],
        'clf__penalty': ['l1', 'l2']
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1
    )

    # 5. 训练与评估
    print("⏳ Training model with grid search...")
    grid_search.fit(X_train, y_train)

    print("\n🔍 Best parameters found:")
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n=== Optimized Classification Report ===")
    print(classification_report(y_test, y_pred))

    # 6. 可视化分析
    plot_confusion_matrix(y_test, y_pred, best_model.classes_)

    # 7. 保存模型
    joblib.dump(best_model, "../../models/optimized_intent_classifier.pkl")
    print("\n✅ Optimized model saved as optimized_intent_classifier.pkl")

if __name__ == "__main__":
    main()
