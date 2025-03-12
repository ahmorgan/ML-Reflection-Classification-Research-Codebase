import glob
import csv
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def calculate_weighted_metrics():
    paths = glob.glob("sync_experiment_results/*")

    for path in paths:
        files = glob.glob(path + "/*.csv")
        cm_tot = np.zeros((5, 5))
        accs = []

        # summing the confusion matrices to get the total tp, fn, fp counts
        for file in files:
            print(file)
            with open(file, "r", encoding="utf-8") as f:
                result = list(csv.reader(f))
                cm = result[1:6]
                cm = np.array(cm)
                cm = cm.astype("int")
                print(cm)
                cm_tot += cm

                accs.append(float(result[38][0]))

        avg_acc = sum(accs) / len(files)

        # credit: https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
        fp_all = cm_tot.sum(axis=0) - np.diag(cm_tot)
        fn_all = cm_tot.sum(axis=1) - np.diag(cm_tot)
        tp_all = np.diag(cm_tot)
        supports = cm_tot.sum(axis=1)  # summing along the rows to get the support for each label class
        tot_support = sum(supports)
        check_support = 0

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

        with open("avg-results/" + path[24:] + "-avg_results.csv", "w", encoding="utf-8", newline="") as ar:
            c_w = csv.writer(ar)
            for item in metrics.items():
                c_w.writerow(list(item))

        labels = ["IDE and Environment Setup", "None", "Other", "Python and Coding", "Time Management and Motivation"]
        display = ConfusionMatrixDisplay(cm_tot, display_labels=labels)
        display.plot(values_format=".0f")
        plt.gca().set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
        plt.savefig("avg-results/" + path[24:] + "-avg-cm.png", bbox_inches="tight")
        plt.cla()


def main():
    calculate_weighted_metrics()


if __name__ == "__main__":
    main()
