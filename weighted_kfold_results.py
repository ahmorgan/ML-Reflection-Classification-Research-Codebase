import glob
import csv
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import os


def _calculate_weighted_metrics(folder, experiment, reflection, variation, k):
    paths = glob.glob(f"{folder}/*")
    paths = [path for path in paths if experiment in path and reflection in path and variation in path and f"{experiment}_results" in path]
    paths = [path for path in paths if paths.count(path) < 2]
    if len(paths) == 0:
        print("###############")
        print(f"Experiment incomplete: {folder}, {experiment}, {reflection}, {variation}")
        print("###############")
        return
    else:
        assert len(paths) == k, f"Did you change the k in the call to weighted_kfold_metrics() or not empty the results directory? Debug: {experiment}, {reflection}, {variation}, num_folds: {len(paths)}, {paths}"  # each call of calculate_weighted_metrics should be for the ten folds for each experiment variation

    # detect length of confusion matrix and accuracy row
    with open(paths[0], "r", encoding="utf-8") as p:
        result = list(csv.reader(p))
        labels = result[0]
        cm_len = len(labels)

        acc_row = 0
        for row, i in zip(result[cm_len:], range(cm_len, len(result))):
            if "accuracy" in row:
                acc_row = i + 1

    cm_tot = np.zeros((cm_len, cm_len))
    accs = []

    # summing the confusion matrices to get the total tp, fn, fp counts
    for file in paths:
        with open(file, "r", encoding="utf-8") as f:
            result = list(csv.reader(f))
            cm = result[1:cm_len+1]
            cm = np.array(cm)
            cm = cm.astype("int")
            assert len(cm) == cm_len

            cm_tot += cm

            accs.append(float(result[acc_row][0]))

    assert len(accs) == k
    avg_acc = sum(accs) / k

    # credit: https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    fp_all = cm_tot.sum(axis=0) - np.diag(cm_tot)
    fn_all = cm_tot.sum(axis=1) - np.diag(cm_tot)
    tp_all = np.diag(cm_tot)
    supports = cm_tot.sum(axis=1)  # summing along the rows to get the support for each label class
    tot_support = sum(supports)
    check_support = 0

    assert len(fp_all) == len(labels)
    assert len(fn_all) == len(labels)
    assert len(tp_all) == len(labels)
    assert len(supports) == len(labels)

    weighted_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    for fp, fn, tp, support in zip(fp_all, fn_all, tp_all, supports):
        check_support += support
        if support != 0:
            class_support = support/tot_support
            class_f1 = class_support * ((2*tp) / (2*tp+fp+fn))
            class_p = class_support * (tp / (tp+fp))
            class_r = class_support * (tp / (tp+fn))
        else:
            class_f1 = 0.0
            class_p = 0.0
            class_r = 0.0

        weighted_f1 += class_f1
        weighted_precision += class_p
        weighted_recall += class_r

    assert check_support == tot_support

    metrics = {
        "arithmetic mean accuracy": avg_acc,
        "weighted precision": weighted_precision,
        "weighted recall": weighted_recall,
        "weighted f1": weighted_f1,
        "cm": cm_tot
    }

    folder = folder[folder.rfind("/")+1:] if folder.rfind("/") else folder[folder.rfind("\\")+1:]

    if not os.path.isdir(f"results/weighted-{folder}"):
        os.mkdir(f"results/weighted-{folder}")

    with open(f"results/weighted-{folder}/{experiment}_{reflection}_{variation}-avg_results.csv", "w", encoding="utf-8", newline="") as ar:
        c_w = csv.writer(ar)
        for item in metrics.items():
            c_w.writerow(list(item))

    display = ConfusionMatrixDisplay(cm_tot, display_labels=labels)
    display.plot(values_format=".0f")
    plt.gca().set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.savefig(f"results/weighted-{folder}/{experiment}_{reflection}_{variation}-avg_cm.png", bbox_inches="tight")
    plt.cla()


"""
def handle_misclassified(folder, experiment, reflection, variation):
    paths = glob.glob(f"{folder}/*.csv")
    paths = [path for path in paths if
             experiment in path and reflection in path and variation in path and "raw_results" in path]
    paths = [path for path in paths if paths.count(path) < 2]

    classifs = [pd.read_csv(paths[0], header=None)]

    for file in paths:
        classif = pd.read_csv(file, header=None)
        classifs.append(classif.drop(classif.columns[0], axis=1))

    classifs_full = pd.concat(classifs, axis=1)

    def consensus(r):
        labels = r[1:]
        con_label = Counter(labels).most_common(1)[0][0]
        return con_label

    classifs_full["consensus"] = classifs_full.apply(consensus, axis=1)
    classifs_full.columns[0] = "text"
    classifs_full = classifs_full[["text", "consensus"]]

    classifs_full.to_csv(f"rawpreds-{folder}/{reflection}_{variation}-rawpreds.csv")


def get_classifications(folder, experiment, reflection, variation):
    paths = glob.glob(f"{folder}/*.csv")
    paths = [path for path in paths if
             experiment in path and reflection in path and variation in path and "raw_results" in path]
    paths = [path for path in paths if paths.count(path) < 2]
    assert len(paths) == 10, f"{experiment}, {reflection}, {variation}, num_folds: {len(paths)}, {paths}"

    for file, i in zip(paths, range(0, len(paths))):
        with open(file, "r", encoding="utf-8") as f:
            classifs = list(csv.reader(f))

        with open(f"classifications/{experiment}_{reflection}_{variation}_fold_{i}.csv", "w", encoding="utf-8", newline="") as c:
            c_w = csv.writer(c)
            c_w.writerows(classifs)
"""


def weighted_kfold_metrics(results_folder, models, reflection_sets, variations, k):
    """
    Calculates weighted kfold F1 and other metrics. Sklearn does not provide an implementation of weighted k-fold F1/
    precision/recall that plays nicely with our setup, so this function is provided.

    :param results_folder: path to folder with results for a kfold experiment
    :param models: which models to incorporate into results calculation, e.g., ["paraphrase-distilroberta-base-v2", "..."]
    :param reflection_sets: which reflection sets to incorporate into results calculation, e.g. ["r1", "r2", "..."]
    :param variations: which experiment variations to incorporate into results calculation, in the format "{agreement}_{shot}", e.g. ["80_10", "100_10", "..."]
    :return:
    """
    for model in models:
        for reflection in reflection_sets:
            for variation in variations:
                _calculate_weighted_metrics(folder=results_folder, experiment=model, reflection=reflection, variation=variation, k=k)
