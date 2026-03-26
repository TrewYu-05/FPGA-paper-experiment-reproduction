import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import sys
import warnings

warnings.filterwarnings('ignore')

from pyod.models.ecod import ECOD

# PyNomaly for LoOP
from PyNomaly import LocalOutlierProbability

from FGAS import FGAS
from baseline import DIS, ODIN, LDOF, OutRanka, WNINOD, INFLO, COF, FastABOD, kNN, IForest
from cdrod import DCROD

def main():
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datasets")
    if not os.path.exists(datasets_dir):
        datasets_dir = "Datasets"

    mat_files = [f for f in os.listdir(datasets_dir) if f.endswith('.mat')]
    mat_files.sort()

    n_datasets = len(mat_files)
    if n_datasets != 15:
        print(f"Warning: Found {n_datasets} .mat files, expected 15.")

    fig, axes = plt.subplots(5, 3, figsize=(20, 25))
    axes = axes.flatten()

    auc_results = []

    k_values = np.arange(1, 61, 1)
    sigma_values = np.arange(0.05, 1.05, 0.05)
    h_values = np.arange(1, 11, 1)

    algorithms = ['DIS', 'COF', 'FastABOD', 'INFLO', 'kNN', 'LDOF', 'LoOP', 'ODIN', 'DCROD', 'ECOD', 'IForest', 'OutRanka', 'WNINOD', 'FGAS']
    colors = plt.cm.tab20(np.linspace(0, 1, len(algorithms)))

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
        n_samples = X.shape[0]

        max_k_for_dataset = min(60, n_samples - 2)
        valid_k_values = k_values[k_values <= max_k_for_dataset]
        if len(valid_k_values) == 0:
            valid_k_values = [1]

        dataset_name = file_name.replace('.mat', '')
        dataset_results = {'Dataset': dataset_name}

        best_rocs = {}

        for algo_idx, algo_name in enumerate(algorithms):
            print(f"  Running {algo_name}...")
            sys.stdout.flush()
            best_auc = -1
            best_fpr = None
            best_tpr = None

            if algo_name in ['COF', 'FastABOD', 'INFLO', 'kNN', 'LoOP', 'DCROD', 'ODIN', 'LDOF', 'OutRanka']:
                for k in valid_k_values:
                    try:
                        if algo_name == 'kNN':
                            clf = kNN(n_neighbors=k)
                            clf.fit(X_scaled)
                            scores = clf.decision_scores_
                        elif algo_name == 'COF':
                            clf = COF(n_neighbors=k)
                            clf.fit(X_scaled)
                            scores = clf.decision_scores_
                        elif algo_name == 'FastABOD':
                            clf = FastABOD(n_neighbors=k)
                            clf.fit(X_scaled)
                            scores = clf.decision_scores_
                        elif algo_name == 'INFLO':
                            clf = INFLO(n_neighbors=k)
                            clf.fit(X_scaled)
                            scores = clf.decision_scores_
                        elif algo_name == 'LoOP':
                            clf = LocalOutlierProbability(X_scaled, n_neighbors=k).fit()
                            scores = clf.local_outlier_probabilities
                        elif algo_name == 'DCROD':
                            clf = DCROD(n_neighbors=k)
                            clf.fit(X_scaled)
                            scores = clf.decision_scores_
                        elif algo_name == 'ODIN':
                            scores = ODIN(X_scaled, k=k)
                        elif algo_name == 'LDOF':
                            scores = LDOF(X_scaled, k=k)
                        elif algo_name == 'OutRanka':
                            scores = OutRanka(X_scaled, k=k)

                        scores = np.nan_to_num(scores)
                        fpr, tpr, _ = roc_curve(y_true, scores)
                        roc_auc = auc(fpr, tpr)
                        if roc_auc > best_auc:
                            best_auc = roc_auc
                            best_fpr, best_tpr = fpr, tpr
                    except Exception as e:
                        pass

            elif algo_name == 'IForest':
                clf = IForest(n_estimators=100, random_state=42)
                clf.fit(X_scaled)
                scores = clf.decision_scores_
                scores = np.nan_to_num(scores)
                fpr, tpr, _ = roc_curve(y_true, scores)
                best_auc = auc(fpr, tpr)
                best_fpr, best_tpr = fpr, tpr

            elif algo_name == 'ECOD':
                clf = ECOD()
                clf.fit(X_scaled)
                scores = clf.decision_scores_
                scores = np.nan_to_num(scores)
                fpr, tpr, _ = roc_curve(y_true, scores)
                best_auc = auc(fpr, tpr)
                best_fpr, best_tpr = fpr, tpr

            elif algo_name == 'DIS':
                scores = DIS(X_scaled)
                scores = np.nan_to_num(scores)
                fpr, tpr, _ = roc_curve(y_true, scores)
                best_auc = auc(fpr, tpr)
                best_fpr, best_tpr = fpr, tpr

            elif algo_name == 'WNINOD':
                for h in h_values:
                    try:
                        scores = WNINOD(X_scaled, h=h)
                        scores = np.nan_to_num(scores)
                        fpr, tpr, _ = roc_curve(y_true, scores)
                        roc_auc = auc(fpr, tpr)
                        if roc_auc > best_auc:
                            best_auc = roc_auc
                            best_fpr, best_tpr = fpr, tpr
                    except Exception as e:
                        pass

            elif algo_name == 'FGAS':
                for current_sigma in sigma_values:
                    try:
                        scores = FGAS(X_scaled, current_sigma)
                        scores = np.nan_to_num(scores)
                        fpr, tpr, _ = roc_curve(y_true, scores)
                        roc_auc = auc(fpr, tpr)
                        if roc_auc > best_auc:
                            best_auc = roc_auc
                            best_fpr, best_tpr = fpr, tpr
                    except Exception as e:
                        pass

            if best_auc == -1: best_auc = 0
            dataset_results[algo_name] = round(best_auc, 3)
            best_rocs[algo_name] = (best_fpr, best_tpr, best_auc)

        auc_results.append(dataset_results)

        if i < len(axes):
            ax = axes[i]
            for algo_idx, algo_name in enumerate(algorithms):
                fpr, tpr, roc_auc = best_rocs[algo_name]
                if fpr is not None:
                    ax.plot(fpr, tpr, color=colors[algo_idx], lw=2, label=f'{algo_name} ({roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate (FPR)')
            ax.set_ylabel('True Positive Rate (TPR)')
            ax.set_title(f'{dataset_name}')
            ax.legend(loc="lower right", prop={'size': 6})

    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results")
    os.makedirs(results_dir, exist_ok=True)
    output_png = os.path.join(results_dir, "ROC_Curves.png")
    plt.savefig(output_png, dpi=300)

    df_auc = pd.DataFrame(auc_results)
    cols = ['Dataset'] + algorithms
    df_auc = df_auc[cols]

    print("\nBest AUC Results Table:")
    print("-" * 100)
    print(df_auc.to_string(index=False))
    print("-" * 100)

    output_csv = os.path.join(results_dir, "AUC_Results.csv")
    df_auc.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv} and {output_png}")

if __name__ == "__main__":
    main()
