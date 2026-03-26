import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import sys
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FGAS import FGAS

def inject_noise(X, noise_ratio):
    """
    Inject attribute noise according to strategy:
    For each attribute, select ceil(x * n) samples randomly.
    Replace their values with a random value between the min and max of that attribute.
    """
    n, m = X.shape
    X_noisy = X.copy()

    if noise_ratio == 0:
        return X_noisy

    num_noisy = int(np.ceil(noise_ratio * n))

    for j in range(m):
        col_min = X[:, j].min()
        col_max = X[:, j].max()

        noisy_indices = np.random.choice(n, num_noisy, replace=False)
        random_values = np.random.uniform(col_min, col_max, size=num_noisy)
        X_noisy[noisy_indices, j] = random_values

    return X_noisy

def main():
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datasets")
    if not os.path.exists(datasets_dir):
        datasets_dir = "Datasets"

    mat_files = [f for f in os.listdir(datasets_dir) if f.endswith('.mat')]
    mat_files.sort()

    n_datasets = len(mat_files)
    if n_datasets != 15:
        print(f"Warning: Found {n_datasets} .mat files, expected 15.")

    noise_levels = np.arange(0.0, 1.1, 0.1)
    results = {}

    for i, file_name in enumerate(mat_files):
        print(f"Processing ({i+1}/{n_datasets}): {file_name}")
        sys.stdout.flush()
        file_path = os.path.join(datasets_dir, file_name)

        mat = loadmat(file_path)
        if 'trandata' in mat:
            trandata = mat['trandata']
            X_raw = trandata[:, :-1]
            y_true = trandata[:, -1]
        elif 'X' in mat and 'y' in mat:
            X_raw = mat['X']
            y_true = mat['y'].flatten()
        else:
            print(f"Could not find data in {file_name}")
            continue

        dataset_name = file_name.replace('.mat', '')
        auc_list = []

        scaler = MinMaxScaler()
        X_clean_scaled = scaler.fit_transform(X_raw)

        best_auc = -1
        best_sigma = 0.5
        for s in np.arange(0.05, 1.05, 0.05):
            try:
                scores = FGAS(X_clean_scaled, s)
                scores = np.nan_to_num(scores)
                fpr, tpr, _ = roc_curve(y_true, scores)
                roc_auc = auc(fpr, tpr)
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    best_sigma = s
            except Exception as e:
                pass

        print(f"  Best sigma found: {best_sigma:.2f} (clean AUC: {best_auc:.4f})")
        sys.stdout.flush()

        for noise_ratio in noise_levels:
            if noise_ratio == 0.0:
                auc_list.append(best_auc)
                continue

            X_noisy = inject_noise(X_raw, noise_ratio)
            X_noisy_scaled = scaler.transform(X_noisy)

            try:
                scores = FGAS(X_noisy_scaled, best_sigma)
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

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results")
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for name in group1:
        plt.plot(noise_levels, results[name], marker='o', label=name)
    plt.xlabel('Noise Ratio (%)')
    plt.xticks(noise_levels, [f"{int(x*100)}%" for x in noise_levels])
    plt.ylabel('AUC')
    plt.title('FGAS Robustness to Attribute Noise (Datasets 1-8)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "Noise_Plot_1.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name in group2:
        plt.plot(noise_levels, results[name], marker='o', label=name)
    plt.xlabel('Noise Ratio (%)')
    plt.xticks(noise_levels, [f"{int(x*100)}%" for x in noise_levels])
    plt.ylabel('AUC')
    plt.title('FGAS Robustness to Attribute Noise (Datasets 9-15)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "Noise_Plot_2.png"), dpi=300)
    plt.close()

    print(f"Saved Noise plots to {results_dir}")

if __name__ == "__main__":
    main()
