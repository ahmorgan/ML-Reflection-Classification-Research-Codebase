from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from setfit import utils
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
from optuna import Trial
import optuna
import numpy
import csv
import torch
import random
import time
import gc
import os
from matplotlib import pyplot as plt
import datetime


results_file = "results/setfit/results.csv"
raw_results_file = "results/setfit/raw_results.csv"
raw_results_probs_file = "results/setfit/raw_results_probabilities.csv"
confusion_matrix_file = "results/setfit/confusion_matrix.png"

current_model = "paraphrase-distilroberta-base-v2"

device_name = "cuda"

dataset_label_set = []


# Generate a confusion matrix for each label in the dataset. For each column/vector
# in the label_num by reflection_num matrix of predictions output by the model,
# one confusion matrix will be created. That will represent the confusion for
# that label. Repeat process for each label. Hopefully, with enough predictions
# for each class, a minimally noisy confusion matrix can be created for each label
def compute_metrics(y_pred, y_true, y_pred_probs) -> dict[str, float]:
    # Either manually set labels, or use automatic label set detection above
    # (dataset_label_set)
    # labels = ['IDE and Environment Setup', 'None ', 'Other',
    #          'Python and Coding', 'Time Management and Motivation']
    """
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
    """
    global current_model
    misclassified_idx = []

    for i in range(0, len(y_true)):
        if y_pred[i] != y_true[i]:
            misclassified_idx.append(i)

    with open("data-splits/train.csv", "r", encoding="utf-8") as train:
        trn = list(csv.reader(train))
        with open(f"results/setfit/{current_model}_train" + results_file[-15:], "w", encoding="utf-8", newline="") as t:
            c_w = csv.writer(t)
            c_w.writerows(trn)

    with open("data-splits/test.csv", "r", encoding="utf-8") as test:
        tst = list(csv.reader(test))
        tst_refs = [row[0] for row in tst]

    misclassified = []

    with open(raw_results_file, "w", encoding="utf-8", newline="") as rr:
        c_w = csv.writer(rr)
        header = ["", "", ""]
        header.extend(dataset_label_set)
        c_w.writerow(header)

        for pred, ref, probs, i in zip(y_pred, tst_refs[1:], y_pred_probs, range(0,len(y_pred))):
            row = [ref, dataset_label_set[pred], ""]
            row.extend(probs)
            if i in misclassified_idx:
                misclassified.append(row)
            c_w.writerow(row)

    with open("results/setfit/" + current_model + "_misclassified" + results_file[-15:], "w", encoding="utf-8", newline="") as m:
        c_w = csv.writer(m)
        c_w.writerows(misclassified)

    matrix = confusion_matrix(y_true, y_pred, labels=[i for i in range(0, len(dataset_label_set))])  # max(y_true) + 1)])
    report = classification_report(y_true, y_pred, labels=[i for i in range(0, len(dataset_label_set))],  # max(y_true) + 1)],
                                    target_names=dataset_label_set, output_dict=True)
    f1 = f1_score(y_true, y_pred, average="weighted")

    with open(results_file, "w", encoding="utf-8", newline="") as results:
        c_w = csv.writer(results)
        c_w.writerow(dataset_label_set)
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

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=dataset_label_set)
    display.plot()
    plt.gca().set_xticklabels(dataset_label_set, rotation=45, ha="right", rotation_mode="anchor")
    plt.savefig(confusion_matrix_file, bbox_inches="tight")
    plt.cla()

    return {"F1": f1}


# model instantiation for each trial run of the hyperparameter search
def model_init(params):
    global current_model
    global device_name
    params = {  # "multi_target_strategy": "one-vs-rest",
              "device": torch.device(device_name)}
    return SetFitModel.from_pretrained(f"sentence-transformers/{current_model}", **params)


"""
# hyperparameters to optimize during hp search
def hp_space(trial: Trial):
    return {
        "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-4, 1e-2, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 1),
        "batch_size": trial.suggest_categorical("batch_size", [2])
    }
"""


def _training_iteration(hps=None, do_hp_search=False, train_file="data-splits/train.csv", test_file="data-splits/test.csv"):
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
        trainer.evaluate()  # evaluate() calls compute_metrics()
        del trainer
        gc.collect()

        return None
    else:  # HP Search
        def objective(trial):
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
            f1 = search_trainer.evaluate()["F1"]

            print(f"Trial result: {f1}")

            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        best_trial = study.best_trial
        best_params = utils.BestRun(str(best_trial.number), best_trial.value, best_trial.params, study)

        return best_params


def _create_splits(dataset_file, shot, train_file="data-splits/train.csv", test_file="data-splits/test.csv"):
    # create 80/20 train and test splits
    with open(dataset_file, "r", encoding="utf-8", newline="") as ds:
        c_r = list(csv.reader(ds))
        c_r = c_r[1:]
        random.seed()
        random.shuffle(c_r)

        prev_labels = [row[1] for row in c_r]

        other_labels = []
        for label in set(prev_labels):
            if prev_labels.count(label) < shot:
                other_labels.append(label)

        # SetFit internally treats the string label "None" as None (as in the null value),
        # so circumvent that by changing the name of the label to "None "
        for row in c_r:
            if row[1] in other_labels:
                row[1] = "Other"
            if row[1] == "None":
                row[1] = "None "
            if row[0] in ["N/A", "n/a", "N/a", "na", "NA", "None", "none"]:
                row[0] = row[0] + " "
            if row[0] is None:
                row[0] = "None "

        labels = [row[1] for row in c_r]

        train = []
        for label in set(labels):
            # exclude labels that are not common enough and are not "IDE and Environment Setup"
            # (special case label where we don't have enough of them but still want to include)
            if labels.count(label) < shot and label not in ["IDE and Environment Setup", "MySQL"]:
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
            if label not in ["IDE and Environment Setup", "MySQL"]:
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


        global dataset_label_set
        dataset_label_set = list(set(train_labels))
        dataset_label_set.sort()  # sort alphabetically so the order matches the
        # order SetFit uses internally


def _update_file_paths(results, raw_results, raw_results_probs, cm):
    global results_file
    global raw_results_file
    global raw_results_probs_file
    global confusion_matrix_file
    results_file = results
    raw_results_file = raw_results
    raw_results_probs_file = raw_results_probs
    confusion_matrix_file = cm


def inference(inference_dataset=None, inference_hps=None, inference_model="stsb-roberta-base-v2", inference_variation="r1_100_20", device="cuda"):
    """
    WIP

    :param inference_dataset:
    :param inference_hps:
    :param inference_model:
    :param inference_variation:
    :param device:
    :return:
    """
    if device == "mps":
        global device_name
        device_name = device

    if not inference_dataset:
        raise ValueError("Please specify at least a dataset for inference.")
    if not inference_hps:
        inference_hps = {'body_learning_rate': 2e-5, 'num_epochs': 1, 'batch_size': 16}
    training_times = {}
    start_time = time.time()
    variation = inference_variation.split("_")
    reflection_set = variation[0]
    agreement = variation[1]
    shot = variation[2]

    global current_model
    current_model = inference_model
    _create_splits(dataset_file=f"full-datasets/sl-{reflection_set}-{agreement}-proficiency.csv", shot=shot)
    _update_file_paths(f"results/setfit/{inference_dataset}-results_{reflection_set}_{agreement}_{shot}.csv",
                       f"results/setfit/{inference_dataset}-raw_results_{reflection_set}_{agreement}_{shot}.csv",
                       f"results/setfit/{inference_dataset}-probs_{reflection_set}_{agreement}_{shot}.csv",
                       f"results/setfit/{inference_dataset}-cm_{reflection_set}_{agreement}_{shot}.png")
    _training_iteration(hps=inference_hps, test_file=f"{inference_dataset}.csv")
    training_times.update({f"{inference_dataset}_{reflection_set}_{agreement}_{shot}": datetime.timedelta(seconds=time.time() - start_time)})
    torch.cuda.empty_cache()

    with open(f"results/setfit/{inference_dataset}_time.csv", "w", encoding="utf-8", newline="") as ifd:
        c_w = csv.writer(ifd)
        c_w.writerows(list(training_times.items()))

    return None


def setfit_experiment(dataset_file_name, shot, k_hp, hps, models, do_hp_search, device="cuda"):
    """
    Runs a single k-fold SetFit experiment based on the specified parameters. To input datasets, make sure you have the dataset
    as a csv file saved locally, and pass in the dataset's file name to dataset_file_name.

    All experiment results are written to a folder "results", which is created automatically.
    Result file names are coded by, "results/setfit/{model}_{results_type}_{reflection_set}_{agreement}_{shot}_{k-fold_iteration}". Training
    times and hyperparameters used are also saved to the results folder under the filename "hps_time.csv".

    Can also run a hyperparameter search, in which case the best run's hyperparameters are returned.

    :param dataset_file_name: path to dataset, must have file name format: "sl-{ref_set}-{agreement}.csv" (e.g., "sl-r1-80.csv").
    :param shot: number of examples per label class in training split
    :param k_hp: k hyperparameter for k-fold
    :param hps: dictionary of structure {'body_learning_rate': _, 'num_epochs': _, 'batch_size': _}, respective suggestions: 2e-5, 1, 16
    :param do_hp_search: whether or not to do a hyperparameter search
    :param models: list of names of Sentence Transformers available through Hugging Face
    :param device: "cuda" by default, can be "mps"
    :return: None or a dictionary of hyperparameters if do_hp_search=True
    """
    # Experiment sequence
    # HP searches -> apply hyperparameters
    # k-fold experiments

    if device == "mps":
        print(f"Running on Apple silicon GPU: {torch.backends.mps.is_available()}")
        global device_name
        device_name = device
    else:
        print(f"Running on CUDA GPU: {torch.cuda.is_available()}")

    if not os.path.isdir("results"):
        os.mkdir("results")
    if not os.path.isdir("results/setfit"):
        os.mkdir("results/setfit")
    if not os.path.isdir("data-splits"):
        os.mkdir("data-splits")

    dataset_name = dataset_file_name[dataset_file_name.index("sl-"):dataset_file_name.index(".csv")]

    training_times = {}

    start_time = time.time()

    ### HP Search(es) ###

    if do_hp_search:
        start_time = time.time()
        # creates train.csv and test.csv to be trained and tested on
        _create_splits(dataset_file=dataset_file_name,
                       shot=shot,
                       train_file="data-splits/train_80_hps.csv",
                       test_file="data-splits/test_80_hps.csv")
        hps = _training_iteration(do_hp_search=True,
                                  train_file="data-splits/train_80_hps.csv",
                                  test_file="data-splits/test_80_hps.csv")
        training_times.update({"hp_search": datetime.timedelta(seconds=start_time - time.time())})
        return hps

    global current_model

    ### K-Fold Evaluation(s) ###
    # Regenerating train/testing splits on every iteration to account for evaluation noise
    for model in models:
        current_model = model
        for k in range(0, k_hp):
            torch.cuda.empty_cache()

            # update paths used in compute_metrics() to log results
            # would pass these as parameters to compute_metrics() but
            # compute_metrics() is called by setfit internally. these have to be global variables
            _update_file_paths(f"results/setfit/{current_model}_results_{dataset_name}_{str(shot)}_" + str(k) + ".csv",
                               f"results/setfit/{current_model}_raw_results_{dataset_name}_{str(shot)}_" + str(k) + ".csv",
                               f"results/setfit/{current_model}_probs_{dataset_name}_{str(shot)}_" + str(k) + ".csv",
                               f"results/setfit/{current_model}_cm_{dataset_name}_{str(shot)}_" + str(k) + ".png")
            _create_splits(dataset_file=dataset_file_name, shot=shot)
            _training_iteration(hps=hps)

        training_times.update({f"{dataset_name}, {current_model}, 80, 10 shot": datetime.timedelta(seconds=time.time() - start_time)})
        print(f"Experiment duration: {datetime.timedelta(seconds=time.time() - start_time)}")

    with open(f"results/setfit/hps_time.csv", "w", encoding="utf-8", newline="") as hpt:
        c_w = csv.writer(hpt)
        c_w.writerows(list(training_times.items()))
        c_w.writerows(list(hps.items()))

