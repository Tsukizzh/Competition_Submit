import numpy as np
from sklearn.metrics import f1_score

def f1_score_macro(y_true, y_pred):
    # 添加调试信息
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    print("\n=== 评估指标调试信息 ===")
    print("预测值范围:", y_pred.min(), "到", y_pred.max())
    print("预测值均值:", y_pred.mean())
    print("真实标签分布:", np.unique(y_true, return_counts=True))
    
    # sigmoid转换
    y_pred_proba = 1 / (1 + np.exp(-y_pred))
    print("sigmoid后预测值范围:", y_pred_proba.min(), "到", y_pred_proba.max())
    print("sigmoid后预测值均值:", y_pred_proba.mean())
    
    # 转换为类别
    y_pred_classes = (y_pred_proba > 0.5).astype(int)
    print("最终预测类别分布:", np.unique(y_pred_classes, return_counts=True))
    
    return f1_score(y_true, y_pred_classes, average='macro')

def root_mean_squared_logarithmic_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))