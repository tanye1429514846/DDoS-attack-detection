 2025/6/4 10:49:26


 2025/7/21 16:17:38


 9:17:35
 src/preprocess.py

 9:18:44
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class RFPPprocessor:
    """实现论文2.1节的RFP特征选择算法"""
    def __init__(self):
        self.selected_features = None
        
    def fit(self, X, y):
        # 步骤1: 随机森林特征重要性 (公式2.1)
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()
        rf.fit(X, y)
        importance = rf.feature_importances_
        
        # 步骤2: Pearson相关性分析 (公式2.2)
        corr_matrix = X.corr().abs()
        
        # 步骤3: 混合选择 (论文表2流程)
        self.selected_features = self._hybrid_select(importance, corr_matrix)
        return self
    
    def transform(self, X):
        return X[self.selected_features]
    
    def _hybrid_select(self, importance, corr_matrix, thresh=0.8):
        """论文中的混合选择逻辑"""
        selected = []
        features = list(corr_matrix.columns)
        
        for i, feat in enumerate(features):
            if importance[i] < 0.001:
                continue
                
            # 检查相关性
            correlated = False
            for sel in selected:
                if corr_matrix.loc[feat, sel] > thresh:
                    correlated = True
                    break
                    
            if not correlated:
                selected.append(feat)
                
        return selected

