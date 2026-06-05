import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# ===================== 全局配置 =====================
SAVE_DIR = ".venv-basic/picture/lesson7"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 任务1：数据准备 =====================
print("=" * 60)
print("任务1：数据准备")
print("=" * 60)

digits = datasets.load_digits()
images = digits.images
X = digits.data
y = digits.target

print(f"图像总数量：{len(images)} 张")
print(f"每张图像大小：{images.shape[1]} × {images.shape[2]} 像素")
print(f"类别标签：{np.unique(y)}")
print(f"特征向量维度（展平后）：{X.shape[1]} 维\n")

# 保存 0~9 每类各一张图片
sample_indices = []
for digit in range(10):
    idx = np.where(y == digit)[0][0]
    sample_indices.append(idx)

plt.figure(figsize=(10, 5))
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[idx], cmap="gray")
    plt.title(f"Label: {y[idx]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "task1_samples.png"))
plt.close()
print(f"✅ 任务1：10类样本图片已保存到 {SAVE_DIR}/task1_samples.png\n")

# ===================== 任务2：数据划分 =====================
print("=" * 60)
print("任务2：数据划分")
print("=" * 60)

X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y, images, test_size=0.25, random_state=42
)

print(f"总样本数量：{len(X)}")
print(f"训练集样本数：{len(X_train)}")
print(f"测试集样本数：{len(X_test)}")
print("训练集：用于训练模型，让模型学习手写数字特征与标签的对应关系。")
print("测试集：用于评估模型在未见过数据上的识别准确率。\n")

# ===================== 任务3：特征表示 =====================
print("=" * 60)
print("任务3：特征表示")
print("=" * 60)

print(f"原始图像形状：{images.shape} → 8×8 像素矩阵")
print(f"特征向量形状：{X.shape} → 1797×64 向量")
print("说明：将8×8图像按行展平为64维向量，是传统机器学习处理图像的常用方式。\n")

# ===================== 任务4：模型训练 =====================
print("=" * 60)
print("任务4：模型训练")
print("=" * 60)

models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    results[name] = acc
    print(f"{name:20s} 准确率: {acc:.4f}")

print()

# ===================== 任务5：结果比较 =====================
print("=" * 60)
print("任务5：结果比较")
print("=" * 60)

print("模型\t\t测试准确率")
print("-" * 30)
for name, acc in results.items():
    print(f"{name:20s} {acc:.4f}")

best_model_name = max(results, key=results.get)
worst_model_name = min(results, key=results.get)

print(f"\n准确率最高模型：{best_model_name} ({results[best_model_name]:.4f})")
print(f"准确率最低模型：{worst_model_name} ({results[worst_model_name]:.4f})")
print("差异原因：与模型假设、拟合能力、对特征的利用效率有关，如朴素贝叶斯受特征独立假设限制，而KNN/SVM对像素特征的匹配能力更强。\n")

# ===================== 任务6：错误样本分析（以KNN为例） =====================
print("=" * 60)
print("任务6：错误样本分析（KNN）")
print("=" * 60)

knn_model = models["KNN"]
y_pred = knn_model.predict(X_test)

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (KNN)")
plt.savefig(os.path.join(SAVE_DIR, "task6_confusion_matrix.png"), bbox_inches="tight")
plt.close()
print(f"✅ 混淆矩阵已保存到 {SAVE_DIR}/task6_confusion_matrix.png")

# 找出错误样本
errors = np.where(y_pred != y_test)[0]
print(f"错误分类样本总数：{len(errors)} 个")

if len(errors) > 0:
    print("\n前5个错误样本（真实 vs 预测）：")
    plt.figure(figsize=(10, 4))
    for idx, i in enumerate(errors[:5]):
        true = y_test[i]
        pred = y_pred[i]
        print(f"真实: {true}, 预测: {pred}")
        plt.subplot(1, 5, idx + 1)
        plt.imshow(img_test[i], cmap="gray")
        plt.title(f"T:{true}\nP:{pred}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "task6_error_samples.png"))
    plt.close()
    print(f"\n✅ 错误样本图片已保存到 {SAVE_DIR}/task6_error_samples.png")
else:
    print("没有错误分类样本，模型表现完美！")

print("\n✅ 所有任务已完成！")