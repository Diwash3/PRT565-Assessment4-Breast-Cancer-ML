# Generates every chart in the report. Usage: python analysis_and_charts.py figures_folder
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

FIG = sys.argv[1]; os.makedirs(FIG, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#ffffff"
BEN, MAL = "#2a78d6", "#eb6834"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"],
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 200,
    "legend.frameon": False,
})
def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close()

d = load_breast_cancer()
X = pd.DataFrame(d.data, columns=d.feature_names); y = pd.Series(d.target)
lab = np.where(y == 0, "Malignant", "Benign")
stats = {}

# 1 class distribution
fig, ax = plt.subplots(figsize=(5, 3.2))
counts = [int((y == 1).sum()), int((y == 0).sum())]
bars = ax.bar(["Benign", "Malignant"], counts, color=[BEN, MAL], width=0.55)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width()/2, c + 6, f"{c} ({c/len(y):.1%})", ha="center", color=INK)
ax.set_ylabel("Number of samples"); ax.set_title("Class distribution of the diagnosis label")
ax.set_ylim(0, 400); ax.grid(axis="x", visible=False)
save("fig01_class_distribution.png")

# 2 correlation heatmap (30 features)
corr = X.corr()
cmap = LinearSegmentedColormap.from_list("div", ["#104281", "#6da7ec", "#f0efec", "#ef8a7a", "#a51f1f"])
fig, ax = plt.subplots(figsize=(9.5, 8))
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(30)); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(30)); ax.set_yticklabels(corr.columns, fontsize=7)
ax.grid(False)
for v in [9.5, 19.5]:
    ax.axhline(v, color="white", lw=2); ax.axvline(v, color="white", lw=2)
cb = plt.colorbar(im, fraction=0.04, pad=0.02); cb.set_label("Pearson correlation")
ax.set_title("Correlation matrix of the 30 features (mean | SE | worst blocks)")
save("fig02_correlation_heatmap.png")
up = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
stats["n_pairs_above_0.9"] = int((up.abs() > 0.9).sum())
stats["top_pairs"] = {" ~ ".join(k): v for k, v in up.abs().sort_values(ascending=False).head(6).round(3).items()}
stats["corr_with_malignant"] = X.corrwith(1 - y).sort_values(ascending=False).round(3).to_dict()

# 3 distributions of key features by class
feats = ["worst perimeter", "worst concave points", "mean texture", "mean fractal dimension"]
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for ax, f in zip(axes.ravel(), feats):
    bins = np.linspace(X[f].min(), X[f].max(), 30)
    ax.hist(X.loc[y == 1, f], bins=bins, color=BEN, alpha=0.65, label="Benign", edgecolor="white", linewidth=0.5)
    ax.hist(X.loc[y == 0, f], bins=bins, color=MAL, alpha=0.65, label="Malignant", edgecolor="white", linewidth=0.5)
    ax.set_title(f); ax.set_ylabel("Count")
axes[0, 0].legend()
fig.suptitle("Feature distributions by diagnosis", fontweight="bold", color=INK)
save("fig03_feature_distributions.png")

# 4 boxplots of standardised mean features
mean_cols = [c for c in X.columns if c.startswith("mean")]
Z = (X[mean_cols] - X[mean_cols].mean()) / X[mean_cols].std()
fig, ax = plt.subplots(figsize=(10, 4.2))
pos = np.arange(len(mean_cols))
for off, cls, col in [(-0.2, 1, BEN), (0.2, 0, MAL)]:
    bp = ax.boxplot([Z.loc[y == cls, c] for c in mean_cols], positions=pos + off, widths=0.33,
                    patch_artist=True, showfliers=True, flierprops=dict(marker="o", markersize=2, alpha=0.5, markeredgecolor=col),
                    medianprops=dict(color=INK, linewidth=1))
    for p in bp["boxes"]: p.set_facecolor(col); p.set_alpha(0.75); p.set_edgecolor(col)
    for k in ["whiskers", "caps"]:
        for l in bp[k]: l.set_color(col)
ax.set_xticks(pos); ax.set_xticklabels([c.replace("mean ", "") for c in mean_cols], rotation=25)
ax.set_ylabel("Standardised value (z-score)")
ax.legend([plt.Rectangle((0,0),1,1,color=BEN), plt.Rectangle((0,0),1,1,color=MAL)], ["Benign", "Malignant"], loc="upper right")
ax.set_title("Mean features by diagnosis (standardised boxplots)")
save("fig04_boxplots.png")

# 5 pairplot of 4 features
pf = ["worst radius", "worst concave points", "worst texture", "worst smoothness"]
fig, axes = plt.subplots(4, 4, figsize=(9, 9))
for i, a in enumerate(pf):
    for j, b in enumerate(pf):
        ax = axes[i, j]
        if i == j:
            bins = np.linspace(X[a].min(), X[a].max(), 25)
            ax.hist(X.loc[y == 1, a], bins=bins, color=BEN, alpha=0.6)
            ax.hist(X.loc[y == 0, a], bins=bins, color=MAL, alpha=0.6)
        else:
            ax.scatter(X.loc[y == 1, b], X.loc[y == 1, a], s=7, color=BEN, alpha=0.55, linewidths=0)
            ax.scatter(X.loc[y == 0, b], X.loc[y == 0, a], s=7, color=MAL, alpha=0.55, linewidths=0)
        if j == 0: ax.set_ylabel(a, fontsize=9)
        else: ax.set_yticklabels([])
        if i == 3: ax.set_xlabel(b, fontsize=9)
        else: ax.set_xticklabels([])
        ax.tick_params(labelsize=7)
fig.legend([plt.Line2D([], [], marker="o", ls="", color=BEN), plt.Line2D([], [], marker="o", ls="", color=MAL)],
           ["Benign", "Malignant"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
fig.suptitle("Pair plot of four 'worst' features", fontweight="bold", color=INK, y=1.04)
save("fig05_pairplot.png")

# 6 PCA
Xs_all = StandardScaler().fit_transform(X)
pca = PCA(n_components=10).fit(Xs_all); P = pca.transform(Xs_all)
stats["pca_var"] = pca.explained_variance_ratio_.round(4).tolist()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.scatter(P[y == 1, 0], P[y == 1, 1], s=9, color=BEN, alpha=0.6, label="Benign", linewidths=0)
ax.scatter(P[y == 0, 0], P[y == 0, 1], s=9, color=MAL, alpha=0.6, label="Malignant", linewidths=0)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"); ax.legend(); ax.set_title("Samples projected onto the first two PCs")
ax = axes[1]
cum = np.cumsum(pca.explained_variance_ratio_)
ax.bar(range(1, 11), pca.explained_variance_ratio_, color=BEN, width=0.6)
ax.plot(range(1, 11), cum, color=INK2, marker="o", lw=1.5, markersize=4)
for k in [0, 1, 4, 9]: ax.text(k + 1, cum[k] + 0.03, f"{cum[k]:.0%}", ha="center", fontsize=8, color=INK)
ax.set_ylim(0, 1.12); ax.set_xticks(range(1, 11)); ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance ratio"); ax.set_title("Variance explained (bars) and cumulative (line)")
save("fig06_pca.png")

# 7 scaling effect
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
sel = ["mean area", "worst area", "mean perimeter", "mean texture", "mean smoothness", "mean fractal dimension"]
axes[0].boxplot([X[c] for c in sel], vert=False, patch_artist=True, boxprops=dict(facecolor="#9ec5f4", edgecolor=BEN),
                medianprops=dict(color=INK), flierprops=dict(markersize=2))
axes[0].set_yticks(range(1, 7)); axes[0].set_yticklabels(sel); axes[0].set_xscale("log")
axes[0].set_title("Before scaling (raw units, log axis)")
Zs = StandardScaler().fit_transform(X[sel])
axes[1].boxplot([Zs[:, i] for i in range(6)], vert=False, patch_artist=True, boxprops=dict(facecolor="#9ec5f4", edgecolor=BEN),
                medianprops=dict(color=INK), flierprops=dict(markersize=2))
axes[1].set_yticks(range(1, 7)); axes[1].set_yticklabels([]); axes[1].set_title("After StandardScaler (z-scores)")
save("fig07_scaling.png")
stats["ranges"] = {c: [round(float(X[c].min()), 4), round(float(X[c].max()), 4)] for c in sel}
stats["skew_top"] = X.skew().sort_values(ascending=False).head(5).round(2).to_dict()

# ---- models
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler(); X_trs = sc.fit_transform(X_tr); X_tes = sc.transform(X_te)
stats["train_counts"] = y_tr.value_counts().to_dict(); stats["test_counts"] = y_te.value_counts().to_dict()
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                               early_stopping=True, max_iter=1000, random_state=42),
}
res, cms, rocs = {}, {}, {}
for n, m in models.items():
    a, b = (X_tr, X_te) if n == "Naive Bayes" else (X_trs, X_tes)
    m.fit(a, y_tr); p = m.predict(b); pr = m.predict_proba(b)[:, 1]
    res[n] = dict(Accuracy=accuracy_score(y_te, p), Precision=precision_score(y_te, p), Recall=recall_score(y_te, p),
                  F1=f1_score(y_te, p), AUC=roc_auc_score(y_te, pr),
                  MalPrecision=precision_score(y_te, p, pos_label=0), MalRecall=recall_score(y_te, p, pos_label=0))
    cms[n] = confusion_matrix(y_te, p).tolist(); rocs[n] = roc_curve(y_te, pr)
mlp = models["ANN (MLP)"]
stats["mlp_iters"] = int(mlp.n_iter_); stats["mlp_best_val"] = round(float(mlp.best_validation_score_), 4)
stats["dt_leaves"] = int(models["Decision Tree"].get_n_leaves()); stats["dt_depth"] = int(models["Decision Tree"].get_depth())
stats["svm_n_sv"] = models["SVM (RBF)"].n_support_.tolist()
stats["results"] = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in res.items()}
stats["cms"] = cms

# 8 metric comparison
mets = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
fig, ax = plt.subplots(figsize=(10, 4.2))
w = 0.16; xpos = np.arange(len(mets))
for i, (n, c) in enumerate(zip(models, SERIES)):
    vals = [res[n][m] for m in mets]
    ax.bar(xpos + (i - 2) * w, vals, w * 0.9, color=c, label=n)
ax.set_xticks(xpos); ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
ax.set_ylim(0.88, 1.0); ax.set_ylabel("Score (axis starts at 0.88)"); ax.grid(axis="x", visible=False)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
save("fig08_metric_comparison.png")

# 9 confusion matrices
blues = LinearSegmentedColormap.from_list("seq", ["#f4f8fe", "#86b6ef", "#2a78d6", "#104281"])
fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
axes = axes.ravel(); axes[5].axis("off")
axes[5].text(0.05, 0.5, "Rows = actual class\nColumns = predicted class\n\nDiagonal = correct predictions\nTop-right = malignant missed (FN)\nBottom-left = benign flagged (FP)\n\nTest set: 42 malignant, 72 benign",
             va="center", fontsize=11, color=INK2, linespacing=1.5)
for ax, n in zip(axes[:5], models):
    cm = np.array(cms[n]); ax.imshow(cm, cmap=blues, vmin=0, vmax=72); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > 30 else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Malig.", "Benign"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Malig.", "Benign"])
    ax.set_title(n); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
save("fig09_confusion_matrices.png")

# 10 ROC
fig, ax = plt.subplots(figsize=(6, 5))
for n, c in zip(models, SERIES):
    fpr, tpr, _ = rocs[n]; ax.plot(fpr, tpr, color=c, lw=2, label=f"{n} (AUC = {res[n]['AUC']:.3f})")
ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=1, label="Random guess")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curves on the test set")
ax.legend(loc="lower right", fontsize=8.5)
save("fig10_roc.png")

# 11 feature importance
imp = pd.Series(models["Random Forest"].feature_importances_, index=X.columns).sort_values()
stats["importance_top10"] = imp.sort_values(ascending=False).head(10).round(4).to_dict()
stats["importance_by_group"] = {g: round(float(imp[[c for c in X.columns if c.startswith(g)]].sum()), 3) for g in ["mean", "worst"]}
stats["importance_by_group"]["error"] = round(float(imp[[c for c in X.columns if c.endswith("error")]].sum()), 3)
top = imp.tail(15)
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(top.index, top.values, color=BEN, height=0.65)
for i, v in enumerate(top.values): ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
ax.set_xlabel("Mean decrease in impurity"); ax.set_title("Random Forest: top 15 feature importances"); ax.grid(axis="y", visible=False)
save("fig11_feature_importance.png")

# 12 decision tree (top 3 levels)
fig, ax = plt.subplots(figsize=(13, 5.5))
plot_tree(models["Decision Tree"], max_depth=2, feature_names=X.columns, class_names=["Malignant", "Benign"],
          filled=False, rounded=True, fontsize=10, ax=ax, impurity=True)
ax.set_title("Decision Tree (top three levels shown)")
save("fig12_decision_tree.png")
tree = models["Decision Tree"].tree_
stats["dt_root"] = [X.columns[tree.feature[0]], round(float(tree.threshold[0]), 3)]

json.dump(stats, open(os.path.join(FIG, "stats.json"), "w"), indent=1, default=str)
print(json.dumps(stats, indent=1, default=str))
