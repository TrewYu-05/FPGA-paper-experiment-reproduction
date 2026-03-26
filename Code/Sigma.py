import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from FGAS import FGAS

def main():
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datasets")
    if not os.path.exists(datasets_dir):
        datasets_dir = "Datasets"

    mat_files = [f for f in os.listdir(datasets_dir) if f.endswith('.mat')]
    mat_files.sort()

    n_datasets = len(mat_files)

    sigma_values = np.arange(0.0, 1.00, 0.05)

    results = {}

    for i, file_name in enumerate(mat_files):
        print(f"Processing ({i+1}/{n_datasets}): {file_name}")
        file_path = os.path.join(datasets_dir, file_name)

        mat = loadmat(file_path)
        if 'trandata' in mat:
            trandata = mat['trandata']
            X = trandata[:, :-1]
            y_true = trandata[:, -1]
        elif 'X' in mat and 'y' in mat:
            X = mat['X']
            y_true = mat['y'].flatten()
        else:
            print(f"Could not find data in {file_name}")
            continue

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        dataset_name = file_name.replace('.mat', '')
        auc_list = []

        for sigma in sigma_values:
            try:
                scores = FGAS(X_scaled, sigma)
                scores = np.nan_to_num(scores)
                fpr, tpr, _ = roc_curve(y_true, scores)
                roc_auc = auc(fpr, tpr)
                auc_list.append(roc_auc)
            except Exception as e:
                auc_list.append(0.0)

        results[dataset_name] = auc_list

    dataset_names = list(results.keys())

    group1 = dataset_names[:8]
    group2 = dataset_names[8:]

    plt.figure(figsize=(10, 6))
    for name in group1:
        plt.plot(sigma_values, results[name], marker='o', label=name)
    plt.xlabel('Sigma')
    plt.ylabel('AUC')
    plt.title('FGAS AUC vs. Sigma (Datasets 1-8)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.00, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sigma_Plot_1.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name in group2:
        plt.plot(sigma_values, results[name], marker='o', label=name)
    plt.xlabel('Sigma')
    plt.ylabel('AUC')
    plt.title('FGAS AUC vs. Sigma (Datasets 9-15)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.00, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sigma_Plot_2.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
