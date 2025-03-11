from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from setfit import utils
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, f1_score, \
    classification_report
from optuna import Trial
import optuna
import numpy
import csv
import torch
import random
import time
import gc
from matplotlib import pyplot as plt


# Since I can't directly call compute_metrics, but that's the place where I specify the names of the results
# files, I'm forced to make the file names global variables which I change in main() when I need to change the output
# file
results_file = "results.csv"
raw_results_file = "raw_results.csv"
raw_results_probs_file = "raw_results_probabilities.csv"
confusion_matrix_file = "confusion_matrix.png"


# compute_metrics is called internally when the model is evaluated. Note that I've made modifications to the SetFit
# source code to allow me to have access to the classification probabilities as well; probabilities cannot be passed to
# compute_metrics out of the box
def compute_metrics(y_pred, y_true, y_pred_probs) -> dict[str, float]:
    labels = ['IDE and Environment Setup', 'None ', 'Other',
              'Python and Coding', 'Time Management and Motivation']

    # Commented out: multi-label evaluation, used in Fall 24 semester poster
    """
    if not any(item in y_true for item in
               [i for i in range(2, max(y_true))]):  # MULTI-LABEL CASE if y_true doesn't contain numbers other than 0,1
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
    """
    # Below: single-label evaluation, what I've been using for synchronous experiments

    # Get the locations of the misclassified reflections for later.
    misclassified_idx = []

    for i in range(0, len(y_true)):
        if y_pred[i] != y_true[i]:
            misclassified_idx.append(i)

    # Save the training split used.
    with open("data-splits/train.csv", "r", encoding="utf-8") as train:
        trn = list(csv.reader(train))
        with open("results/train" + results_file[-15:], "w", encoding="utf-8", newline="") as t:
            c_w = csv.writer(t)
            c_w.writerows(trn)

    # Read the test split used for writing the raw results file (next).
    with open("data-splits/test.csv", "r", encoding="utf-8") as test:
        tst = list(csv.reader(test))
        tst_refs = [row[0] for row in tst]

    misclassified = []

    # Write the raw results file, which includes the predicted label and probabilities for each reflection, and the reflection text.
    with open(raw_results_file, "w", encoding="utf-8", newline="") as rr:
        c_w = csv.writer(rr)
        c_w.writerow(["", "", "", labels[0], labels[1], labels[2], labels[3], labels[4]])
        for pred, ref, probs, i in zip(y_pred, tst_refs[1:], y_pred_probs, range(0,len(y_pred))):
            row = [ref, labels[pred], ""]
            row.extend(probs)
            if i in misclassified_idx:
                misclassified.append(row)
            c_w.writerow(row)

    # Write the raw results file for only the misclassified reflections.
    with open("results/misclassified" + results_file[-15:], "w", encoding="utf-8", newline="") as m:
        c_w = csv.writer(m)
        c_w.writerows(misclassified)

    matrix = confusion_matrix(y_true, y_pred, labels=[i for i in range(0, len(labels))])
    report = classification_report(y_true, y_pred, labels=[i for i in range(0, len(labels))],
                                   target_names=labels, output_dict=True)
    f1 = f1_score(y_true, y_pred, average="weighted")

    # writes the metrics from the classification report to a results file
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

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot()
    plt.gca().set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.savefig(confusion_matrix_file, bbox_inches="tight")
    plt.cla()

    # Used when maximizing the F1 during the hyperparameter search.
    return {"F1": f1}


# Instantiates the model every time I create a new Trainer object.
def model_init(params):
    params = {  # "multi_target_strategy": "one-vs-rest",
        "device": torch.device("cuda")}
    return SetFitModel.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v2", **params)


# Performs one "fold" or one training iteration, with the hyperparameters specified by the hps parameter.
def training_iteration(hps=None, do_hp_search=False, train_file="data-splits/train.csv",
                       test_file="data-splits/test.csv"):
    # Supports multi-label and single-label text classification using Setfit
    # loosely followed https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb

    # Before running this code, create folders named data-splits, full_datasets, and results
    # full_datasets should contain the dataset(s) you are using, e.g. SL-R1-80P
    # Last, change the labels List in compute_metrics to match the labels you are using in your training set
    # (By default, they are the labels in the training sets of SL-R1-80P and Sl-R1-100P).

    print("Loading datasets...")
    dataset = load_dataset('csv', data_files={
        "train": train_file,
        "test": test_file
    })

    print("Processing datasets...")
    # extract the header column in the dataset
    labels = dataset["train"].column_names

    # Further preprocessing, only necessary if you are doing multi-label classification
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

    print("Loading model...")

    if not do_hp_search:
        # Either hardcode the hyperparameters or pass in an hps dictionary to specify hyperparameters
        """
        args = TrainingArguments(
            batch_size=8,
            num_epochs=1,
            body_learning_rate=1e-4
        )
        """

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

        # VRAM optimizations; clear out memory not used later
        del trainer
        gc.collect()

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
    else:  # Hyperparameter search
        def objective(trial):
            # Define the hyperparameter search space by altering the bounds of each hyperparameter
            # (also, add other hyperparameters if desired)
            body_learning_rate = trial.suggest_float("body_learning_rate", 1e-6, 1e-4, log=True)
            num_epochs = trial.suggest_int("num_epochs", 1, 3)
            batch_size = trial.suggest_categorical("batch_size", [8, 16])

            print(f"Learning rate: {body_learning_rate}")
            print(f"Epoch: {num_epochs}")
            print(f"Batch size: {batch_size}")

            args = TrainingArguments(
                body_learning_rate=body_learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )

            search_trainer = Trainer(
                model_init=model_init,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                metric=compute_metrics,
                args=args
            )

            search_trainer.train()

            # I chose to optimize the weighted F1 score.
            f1 = search_trainer.evaluate()["F1"]

            print(f"Trial result: {f1}")

            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        best_trial = study.best_trial
        print(study.best_params)
        best_params = utils.BestRun(str(best_trial.number), best_trial.value, best_trial.params, study)

        # best_params will be applied in the next training_iteration() call if hps=best_params.
        return best_params

    # model.push_to_hub("my-fantastic-model")


# Customized split creation method. dataset_file specifies what dataset to generate the splits from, and
# shot specifies how many examples per label class to generate. Currently, the default behavior if the shot
# exceeds the actual number of examples for a label class in the dataset is to exclude that label class altogether,
# though I've hardcoded in an exception for IDE and Environment Setup to allow for it to be in the splits.
def create_splits(dataset_file, shot, train_file="data-splits/train.csv", test_file="data-splits/test.csv"):
    # Open the full dataset (SL-R1-80P, for example).
    with open(dataset_file, "r", encoding="utf-8", newline="") as ds:
        c_r = list(csv.reader(ds))
        c_r = c_r[1:]  # exclude the header column [text, label].

        # Reset the seed and shuffle the dataset.
        random.seed()
        random.shuffle(c_r)

        # SetFit internally treats the string label "None" as None (as in the null value),
        # so circumvent that by changing the name of the label to "None ".
        for row in c_r:
            if row[1] == "None":
                row[1] = "None "

        labels = [row[1] for row in c_r]

        train = []
        for label in set(labels):
            # exclude labels that are not common enough and are not "IDE and Environment Setup"
            # (special case label where we don't have enough of them but still want to include it)
            if labels.count(label) < shot and label != "IDE and Environment Setup":
                continue
            count = 0

            # Fill the training split with shot number of examples for that label class.
            for row in c_r:
                if row[1] == label:
                    train.append(row)
                    count += 1
                if count == shot:
                    break

        train_labels = [row[1] for row in train]

        # Construct the test set; includes all reflections not in train, as long as the label
        # attached to the reflection is also in train.
        test = [row for row in c_r if (row not in train and row[1] in set(train_labels))]

        ### Test Cases ###
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


# Used to update the global results filenames -- this is an awkward solution, but
# I write the results files in compute_metrics, and I can't directly pass in file name parameters
# to that function.
def update_file_paths(results, raw_results, raw_results_probs, cm):
    global results_file
    global raw_results_file
    global raw_results_probs_file
    global confusion_matrix_file
    results_file = results
    raw_results_file = raw_results
    raw_results_probs_file = raw_results_probs
    confusion_matrix_file = cm


# main() defines the sequence of training experiments.
def main():
    # Instructions: before running the code, you have to alter the transformers library source code to fix a
    # backwards compatibility issue that exists for some reason -- all instances of the class attribute
    # "eval_strategy" must be changed to "evaluation_strategy" in trainer_callback.py.
    # Additionally, make the source code change mentioned in compute_metrics to allow for label class probabilities
    # to be passed to it.
    #
    # Make sure that you create empty directories data-splits, full-datasets, and results, with full-datasets
    # containing the complete dataset(s) you intend to use (e.g. SL-R1-80P).

    training_times = {}

    ### HP Search(es), comment out if undesired. ###
    # Edit the HP search space in training_iteration().
    do_hp_search = False
    hps = {}

    if do_hp_search:
        start_time = time.time()
        # creates train.csv and test.csv to be trained and tested on
        create_splits(dataset_file="full-datasets/sl-r1-80-proficiency.csv",
                      shot=10,
                      train_file="data-splits/train_80_hps.csv",
                      test_file="data-splits/test_80_hps.csv")
        hps = training_iteration(do_hp_search=True,
                                 train_file="data-splits/train_80_hps.csv",
                                 test_file="data-splits/test_80_hps.csv")
        training_times.update({"hp_search": start_time - time.time()})
        print(hps)
    else:
        # Values as defined in https://arxiv.org/pdf/2209.11055, except for the learning rate,
        # which is set to the default value specified in the SetFit documentation instead of the
        # paper's 1e-3, because too high of a learning rate caused the model not to converge
        hps = {'body_learning_rate': 2e-5, 'num_epochs': 1, 'batch_size': 16}

    start_time = time.time()

    ### K-Fold Experiments(s) ###
    # Regenerating train/testing splits on every iteration to account for evaluation noise

    k_hp = 10  # number of k-fold iterations.

    # 80P, 10 shot
    for k in range(0, k_hp):
        torch.cuda.empty_cache()

        update_file_paths("results/results_r1_80_10_" + str(k) + ".csv",
                          "results/raw_results_r1_80_10_" + str(k) + ".csv",
                          "results/probs_r1_80_10_" + str(k) + ".csv",
                          "results/cm_r1_80_10_" + str(k) + ".png")
        create_splits(dataset_file="full-datasets/sl-r1-80-proficiency.csv", shot=10)
        training_iteration(hps=hps)

    training_times.update({"80, 10 shot": time.time() - start_time})
    print(f"Experiment duration: {time.time() - start_time}")
    start_time = time.time()

    # 100P, 10 shot
    for k in range(0, k_hp):
        torch.cuda.empty_cache()

        update_file_paths("results/results_r1_100_10_" + str(k) + ".csv",
                          "results/raw_results_r1_100_10_" + str(k) + ".csv",
                          "results/probs_r1_100_10_" + str(k) + ".csv",
                          "results/cm_r1_100_10_" + str(k) + ".png")
        create_splits(dataset_file="full-datasets/sl-r1-100-proficiency.csv", shot=10)
        training_iteration(hps=hps)

    training_times.update({"100, 10 shot": time.time() - start_time})
    print(f"Experiment duration: {time.time() - start_time}")
    start_time = time.time()

    # 80P, 20 shot
    for k in range(0, k_hp):
        torch.cuda.empty_cache()

        update_file_paths("results/results_r1_80_20_" + str(k) + ".csv",
                          "results/raw_results_r1_80_20_" + str(k) + ".csv",
                          "results/probs_r1_80_20_" + str(k) + ".csv",
                          "results/cm_r1_80_20_" + str(k) + ".png")
        create_splits(dataset_file="full-datasets/sl-r1-80-proficiency.csv", shot=20)
        training_iteration(hps=hps)

    training_times.update({"80, 20 shot": time.time() - start_time})
    print(f"Experiment duration: {time.time() - start_time}")
    start_time = time.time()

    # 100P, 20 shot
    for k in range(0, k_hp):
        torch.cuda.empty_cache()

        update_file_paths("results/results_r1_100_20_" + str(k) + ".csv",
                          "results/raw_results_r1_100_20_" + str(k) + ".csv",
                          "results/probs_r1_100_20_" + str(k) + ".csv",
                          "results/cm_r1_100_20_" + str(k) + ".png")
        create_splits(dataset_file="full-datasets/sl-r1-100-proficiency.csv", shot=20)
        training_iteration(hps=hps)

    training_times.update({"100, 20 shot": time.time() - start_time})
    print(f"Experiment duration: {time.time() - start_time}")

    print(hps)
    print(training_times)

    """
    with open("hps_time.csv", "w", encoding="utf-8", newline="") as hpt:
        c_w = csv.writer(hpt)
        c_w.writerows(training_times)
        c_w.writerows(hps)
    """


if __name__ == "__main__":
    main()
