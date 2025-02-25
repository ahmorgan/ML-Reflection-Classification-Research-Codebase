from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
from optuna import Trial
import numpy
import csv
import torch
import random
from matplotlib import pyplot as plt


results_file = "results.csv"
raw_results_file = "raw_results.csv"
raw_results_probs_file = "raw_results_probabilities.csv"
confusion_matrix_file = "confusion_matrix.png"


# Generate a confusion matrix for each label in the dataset. For each column/vector
# in the label_num by reflection_num matrix of predictions output by the model,
# one confusion matrix will be created. That will represent the confusion for
# that label. Repeat process for each label. Hopefully, with enough predictions
# for each class, a minimally noisy confusion matrix can be created for each label
def compute_metrics(y_pred, y_true, y_pred_probs) -> dict[str, float]:
    # initialize labels
    labels = ['IDE and Environment Setup', 'None ', 'Other',
              'Python and Coding', 'Time Management and Motivation']
    if not any(item in y_true for item in [i for i in range(2, max(y_true))]):  # MULTI-LABEL CASE if y_true doesn't contain numbers other than 0,1
        # save the raw predictions made by the model
        with open("raw_setfit_preds.csv", "w", encoding="utf-8", newline='') as rsp:
            c_w = csv.writer(rsp)
            for i in range(0, len(y_true)):
                row = []
                for j in range(0, len(labels)):
                    row.append(y_pred[i][j].item())
                c_w.writerow(row)

        # confusion_matrices is a list of n-dimensional numpy arrays
        # list is of size num_labels
        confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)

        result = {}
        x = 0
        for matrix in confusion_matrices:
            # flatten confusion matrix to list
            matrix = matrix.ravel()
            # populate results with information from the label's confusion matrix
            result.update({f"{labels[x]}-tn": matrix[0].item()})
            result.update({f"{labels[x]}-fp": matrix[1].item()})
            result.update({f"{labels[x]}-fn": matrix[2].item()})
            result.update({f"{labels[x]}-tp": matrix[3].item()})
            x += 1
            if x >= len(labels):
                break
        accuracy = 0.0
        print(len(y_true))
        for label in labels:
            # len(y_true) is the number of reflections used in evaluation
            # acc = num_of_correct_classifications / num_reflections
            accuracy += (result[f"{label}-tp"] + result[f"{label}-tn"]) / len(y_true)
        accuracy /= len(labels)
        result.update({"accuracy": accuracy})
        return result
    else:
        matrix = confusion_matrix(y_true, y_pred, labels=[i for i in range(0, len(labels))])  # max(y_true) + 1)])
        report = classification_report(y_true, y_pred, labels=[i for i in range(0, len(labels))],  # max(y_true) + 1)],
                                       target_names=labels, output_dict=True)
        f1 = f1_score(y_true, y_pred, average="weighted")

        with open(results_file, "w", encoding="utf-8", newline="") as results:
            c_w = csv.writer(results)
            c_w.writerow(labels)
            for row in matrix:
                c_w.writerow(row)
            c_w.writerow([])
            for label in report.keys():
                c_w.writerow([label])
                if type(report[label]) != dict:
                    if type(report[label]) == float:
                        report[label] = [report[label]]
                    c_w.writerow(report[label])
                else:
                    for item in report[label].items():
                        c_w.writerow(item)
                c_w.writerow([])

        with open(raw_results_file, "w", encoding="utf-8", newline="") as rr:
            c_w = csv.writer(rr)
            for pred in y_pred:
                c_w.writerow([labels[pred]])

        with open(raw_results_probs_file, "w", encoding="utf-8", newline="") as rrp:
            c_w = csv.writer(rrp)
            for pred in y_pred_probs:
                c_w.writerow(pred)

        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
        display.plot()
        plt.gca().set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
        plt.savefig(confusion_matrix_file)
        plt.cla()
        return {"F1": f1}


# model instantiation for each trial run of the hyperparameter search
def model_init(params):
    params = {  # "multi_target_strategy": "one-vs-rest",
              "device": torch.device("cuda")}
    # all-MiniLM-L12-v2 is 33.6M params
    return SetFitModel.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v2", **params)


# hyperparameters to optimize during hp search
def hp_space(trial: Trial):
    return {
        "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-4, 1e-2, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 3),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16])
    }


def training_iteration(hps=None, do_hp_search=False, train_file="train.csv", test_file="test.csv"):
    # Multi-label text classification using Setfit
    # loosely followed https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb

    # Instructions: create a folder called "data-splits" containing "setfit-dataset-setfit-dataset-train.csv" and setfit-dataset-setfit-dataset-test.csv", which are generated from the Dataset Construction script
    # Uncomment hyperparameter search code block and comment TrainingArguments code block and "args=args" to run a hyperparameter search
    # Last, change the labels List in compute_metrics if running experiments with different labels than "Python and Coding", "GitHub", "Assignments", and "Time Management"

    # Datasets are generated using the consensus data parser script
    print("Loading datasets...")
    # load two datasets from csv files in dataset dictionary
    dataset = load_dataset('csv', data_files={
        "train": train_file,
        "test": test_file
    })

    print("Processing datasets...")
    # extract the header column in the dataset
    labels = dataset["train"].column_names
    if len(labels) > 2:  # len(labels) > 2 indicates a multi-label dataset
        print("Multi-label dataset detected, doing preprocessing...")
        labels.remove("text")

        # further preprocess data
        # used guide https://medium.com/@farnazgh73/few-shot-text-classification-on-a-multilabel-dataset-with-setfit-e89504f5fb75 for help here
        # .map takes a method and applies it to each entry in the dataset
        # the lambda method converts the entries in the dataset to encoded labels
        # ex. {"Time Management":0, "Python and Coding": 1} becomes {"label": [0,1]} (not a real example, just to illustrate what's happening)
        dataset["train"] = dataset["train"].map(lambda entry: {"label": [entry[label] for label in labels]})
        dataset["test"] = dataset["test"].map(lambda entry: {"label": [entry[label] for label in labels]})

        # collect exactly eight examples of every labeled class in training dataset
        # elegant line of code taken from above medium.com guide
        eight_examples_of_each = numpy.concatenate(
            [numpy.random.choice(numpy.where(dataset["train"][label])[0], 10) for label in labels])
        # replace training dataset with the eight examples of each
        dataset["train"] = dataset["train"].select(eight_examples_of_each)

        # remove unnecessary labels
        dataset["train"] = dataset["train"].select_columns(["text", "label"])
        dataset["test"] = dataset["test"].select_columns(["text", "label"])

        # dataset["train"] is now a collection of 8*num_labels reflections, where there are at least 8
        # reflections with a certain label (there could be more because the dataset is multi-label)
        # dataset["train"] has not had any reflections removed. All that has happened to it is that the
        # labels for each reflection have been encoded into an entry with the form {"label":[0,0,1,...0])

        # therefore, the model will train on eight examples of each label, and metrics will be computed based on
        # on classifications made from a large set of reflections in a randomized order
        # no reflection from the test split will be in the train split, so over-fitting should not be a concern

    # In the single label case, the data is already prepared for classification

    # tokenization as specified in the "Fine tuning BERT (and friends)" notebook is not necessary or worthwhile
    # (as far as I know) working with SetFit models. SetFit must tokenize the data behind the scene

    print("Loading model...")

    if not do_hp_search:
        # fine tune pretrained model using datasets using default hyperparameters (will change as I run experiments with
        # varying hyperparameters, only running default hps for debugging right now)
        trainer = Trainer(
            model_init=model_init,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            metric=compute_metrics,
        )
        trainer.apply_hyperparameters(hps)

        print("Training...")

        trainer.train()
        print("Testing...")
        metrics = trainer.evaluate()  # evaluate() calls compute_metrics()

        """
            print(metrics)

            with open("metrics.csv", "w") as m:
                c_w = csv.writer(m)
                for key in metrics.keys():
                    arr = [key, metrics[key]]
                    c_w.writerow(arr)
            print("Metrics data written to metrics.csv")
        """

        return None
    else:  # HP Search
        trainer = Trainer(
            model_init=model_init,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            metric=compute_metrics,
        )

        # optimizing sentence transformer learning rate and num of epochs with hyperparameter search
        best_run = trainer.hyperparameter_search(
            # compute_objective is the overall accuracy of all labels
            direction="maximize",  # maximize accuracy
            hp_space=hp_space,
            compute_objective=lambda result: result.get("F1"),
            n_trials=10
        )
        print(best_run.hyperparameters)

        return best_run.hyperparameters

    # model.push_to_hub("setfit-multilabel-test")


def create_splits(dataset_file, shot, train_file="data-splits/train.csv", test_file="data-splits/test.csv"):
    # create 80/20 train and test splits
    with open(dataset_file, "r", encoding="utf-8", newline="") as ds:
        c_r = list(csv.reader(ds))
        c_r = c_r[1:]
        random.seed()
        random.shuffle(c_r)

        # FastFit internally treats the string label "None" as None (as in the null value),
        # so circumvent that by changing the name of the label to "None "
        for row in c_r:
            if row[1] == "None":
                row[1] = "None "

        labels = [row[1] for row in c_r]

        train = []
        for label in set(labels):
            # exclude labels that are not common enough and are not "IDE and Environment Setup"
            # (special case label where we don't have enough of them but still want to include)
            if labels.count(label) < shot and label != "IDE and Environment Setup":
                continue
            count = 0
            for row in c_r:
                if row[1] == label:
                    train.append(row)
                    count += 1
                if count == shot:
                    break
        train_labels = [row[1] for row in train]
        test = [row for row in c_r if (row not in train and row[1] in set(train_labels))]

        for row in train:
            assert row not in test, "Test contains reflections from train!"

        for label in set(train_labels):
            if label != "IDE and Environment Setup":
                assert train_labels.count(label) == shot, "Train does not contain ten of each label!"

        test_labels = [row[1] for row in test]
        for label in set(test_labels):
            print(f"{label} label count in test: {test_labels.count(label)}")
        for label in set(train_labels):
            print(f"{label} label count in train: {train_labels.count(label)}")

        with open(test_file, "w", encoding="utf-8", newline="") as tst:
            c_w = csv.writer(tst)
            c_w.writerow(["text", "label"])
            c_w.writerows(test)

        with open(train_file, "w", encoding="utf-8", newline="") as trn:
            c_w = csv.writer(trn)
            c_w.writerow(["text", "label"])
            c_w.writerows(train)


def update_file_paths(results, raw_results, raw_results_probs, cm):
    global results_file
    global raw_results_file
    global raw_results_probs_file
    global confusion_matrix_file
    results_file = results
    raw_results_file = raw_results
    raw_results_probs_file = raw_results_probs
    confusion_matrix_file = cm


def main():
    # Experiment sequence
    # HP searches -> apply hyperparameters
    # k-fold experiments

    ### HP Search(es) ###

    # creates train.csv and test.csv to be trained and tested on
    create_splits(dataset_file="full_datasets/sl-r1-80-proficiency.csv",
                  shot=10,
                  train_file="data-splits/train_80_hps.csv",
                  test_file="data-splits/test_80_hps.csv")
    r180_hps = training_iteration(do_hp_search=True,
                                  train_file="data-splits/train_80_hps.csv",
                                  test_file="data-splits/test_80_hps.csv")
    print(r180_hps)

    ### K-Fold Evaluation(s) ###
    # Regenerating train/testing splits on every iteration to account for evaluation noise

    # 80, 10 shot
    for k in range(0, 5):
        # update paths used in compute_metrics() to log results
        # would pass these as parameters to compute_metrics() but
        # compute_metrics() is called by setfit internally so it's difficult to
        update_file_paths("results/results_r1_80_10_" + str(k) + ".csv",
                          "results/raw_results_r1_80_10_" + str(k) + ".csv",
                          "results/probs_r1_80_10_" + str(k) + ".csv",
                          "results/cm_r1_80_10_" + str(k) + ".csv")
        create_splits(dataset_file="full_datasets/sl-r1-80-proficiency.csv", shot=10)
        training_iteration(hps=r180_hps)

    # 100, 10 shot
    for k in range(0, 5):
        update_file_paths("results/results_r1_100_10_" + str(k) + ".csv",
                          "results/raw_results_r1_100_10_" + str(k) + ".csv",
                          "results/probs_r1_100_10_" + str(k) + ".csv",
                          "results/cm_r1_100_10_" + str(k) + ".csv")
        create_splits(dataset_file="full_datasets/sl-r1-100-proficiency.csv", shot=10)
        training_iteration(hps=r180_hps)

    # 80, 20 shot
    for k in range(0, 5):
        update_file_paths("results/results_r1_80_20_" + str(k) + ".csv",
                          "results/raw_results_r1_80_20_" + str(k) + ".csv",
                          "results/probs_r1_80_20_" + str(k) + ".csv",
                          "results/cm_r1_80_20_" + str(k) + ".csv")
        create_splits(dataset_file="full_datasets/sl-r1-80-proficiency.csv", shot=20)
        training_iteration(hps=r180_hps)

    # 100, 20 shot
    for k in range(0, 5):
        update_file_paths("results/results_r1_100_20_" + str(k) + ".csv",
                          "results/raw_results_r1_100_20_" + str(k) + ".csv",
                          "results/probs_r1_100_20_" + str(k) + ".csv",
                          "results/cm_r1_100_20_" + str(k) + ".csv")
        create_splits(dataset_file="full_datasets/sl-r1-100-proficiency.csv", shot=20)
        training_iteration(hps=r180_hps)

    print(r180_hps)


if __name__ == "__main__":
    main()
