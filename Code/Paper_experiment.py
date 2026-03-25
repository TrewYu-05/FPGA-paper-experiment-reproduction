import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from FGAS import FGAS

def main():
    datasets_dir = r"d:\learning\LiuChang\FGAS\Datasets"
    mat_files = [f for f in os.listdir(datasets_dir) if f.endswith('.mat')]
    mat_files.sort()
    
    n_datasets = len(mat_files)
    if n_datasets != 15:
        print(f"Warning: Found {n_datasets} .mat files, expected 15.")
        
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()
    
    auc_results = []
    
    sigma_values = np.arange(0.05, 1.05, 0.05) # From 0.05 to 1.0, step 0.05
    
    for i, file_name in enumerate(mat_files):
        print(f"Processing ({i+1}/{n_datasets}): {file_name}")
        file_path = os.path.join(datasets_dir, file_name)
        
        mat = loadmat(file_path)
        trandata = mat['trandata']
        
        X = trandata[:, :-1]
        y_true = trandata[:, -1]
        
        # Preprocessing: MinMax scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        best_auc = -1
        best_sigma = None
        best_fpr = None
        best_tpr = None
        
        for current_sigma in sigma_values:
            anomaly_scores = FGAS(X_scaled, current_sigma)
            fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
            roc_auc = auc(fpr, tpr)
            
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_sigma = current_sigma
                best_fpr = fpr
                best_tpr = tpr
        
        dataset_name = file_name.replace('.mat', '')
        auc_results.append({
            'Dataset': dataset_name,
            'Best AUC': best_auc,
            'Best Sigma': best_sigma
        })
        
        if i < len(axes):
            ax = axes[i]
            ax.plot(best_fpr, best_tpr, color='darkorange', lw=2, label=f'AUC = {best_auc:.4f} (σ={best_sigma:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate (FPR)')
            ax.set_ylabel('True Positive Rate (TPR)')
            ax.set_title(f'ROC - {dataset_name}')
            ax.legend(loc="lower right")
            
    plt.tight_layout()
    plt.savefig(r"d:\learning\LiuChang\FGAS\Code\ROC_Curves.png")
    
    df_auc = pd.DataFrame(auc_results)
    print("\nBest AUC Results Table:")
    print("-" * 50)
    print(df_auc.to_string(index=False))
    print("-" * 50)
    
    df_auc.to_csv(r"d:\learning\LiuChang\FGAS\Code\AUC_Results.csv", index=False)

if __name__ == "__main__":
    main()
