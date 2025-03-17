import glob
import csv
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def calculate_weighted_metrics():
    # Using a folder named sync_experiment_results, which contains nested folders that correspond to the experiment variation;
    # e.g. sync_experiment_results/dR-F-80-10/...
    # Each subfolder contains the 10-fold results for that experiment variation, with one file per fold
    # The fold results were taken from my raw results files, posted for each experiment in the Sp25-Experiment-Analysis-Templates spreadsheet
    # (They are all of the files prepended with "results_")
    paths = glob.glob("sync_experiment_results/*")

    # Iterating over the resutls for the 10 folds
    # Paths is the folder containing the experiemnt variation results
    for path in paths:
        # Files contains the individual fold results for each variation
        files = glob.glob(path + "/*.csv")
        cm_tot = np.zeros((5, 5))
        accs = []

        # summing the confusion matrices element-wise to get the total tp, fn, fp counts
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

        # Getting the tp, fp, fn from the summed confusion matrix for each label class
        # credit: https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
        fp_all = cm_tot.sum(axis=0) - np.diag(cm_tot)
        fn_all = cm_tot.sum(axis=1) - np.diag(cm_tot)
        tp_all = np.diag(cm_tot)
        supports = cm_tot.sum(axis=1)  # summing along the rows to get the support for each label class
        tot_support = sum(supports)
        check_support = 0  # used for test case later on

        weighted_f1 = 0.0
        weighted_precision = 0.0
        weighted_recall = 0.0
        # Calculating the F1, precision, recall for each label class based on the tp, fp, fn values calculated previously
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

        # Adding all of the support-weighted label class f1s yields the overall weighted f1
        # e.g. (0.3 * 0.9) + (0.7 * 0.8) with two label classes would yield the overall weighted f1

        assert check_support == tot_support

        metrics = {
            "arithmetic mean accuracy": avg_acc,
            "weighted precision": weighted_precision,
            "weighted recall": weighted_recall,
            "weighted f1": weighted_f1,
            "cm": cm_tot
        }

        # path[24:] is the experiment variation name in my setup, change if using different file names
        with open("avg-results/" + path[24:] + "-avg_results.csv", "w", encoding="utf-8", newline="") as ar:
            c_w = csv.writer(ar)
            for item in metrics.items():
                c_w.writerow(list(item))

        # Confusion matrix display for summed cm
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
