from datasets import load_dataset
from fastfit import FastFitTrainer, FastFit
from transformers import AutoTokenizer, pipeline
import random
import csv
import torch
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import optuna
import os


# Since I can't directly call compute_metrics, but that's the place where I specify the names of the results
# files, I'm forced to make the file names global variables which I change in main() when I need to change the output
# file
results_file = "results.csv"
raw_results_file = "raw_results.csv"
raw_results_probs_file = "raw_results_probabilities.csv"
confusion_matrix_file = "confusion_matrix.png"


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


# compute_metrics is called internally every time a model is trained -- note that, to use this function,
# you will have to modify the FastFit source code to allow you to pass in compute_metrics as a Callable when
# the FastFitTrainer is instantiated, because out of the box this is impossible (even though the FastFit devs
# claim it is possible -- it looks like they accidentally hardcoded in their own internal compute_metrics function).
def compute_metrics(p) -> dict[str, float]:
    # Predictions are passed in after training as EvalPrediction objects, which contain the predictions as logits
    predictions = (p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions)

    # Get the prediction probabilities by taking the softmax of the logits.
    probs = torch.softmax(torch.tensor(predictions), dim=1)

    # Get the predictions themselves by getting the index of the largest logit in each prediction.
    predictions = np.argmax(predictions, axis=1)

    # True labels.
    references = p.label_ids

    # Save the locations of the misclassified reflections for later.
    misclassified_idx = []
    for i in range(0, len(references)):
        if predictions[i] != references[i]:
            misclassified_idx.append(i)

    print(probs)
    print(predictions)

    labels = ['IDE and Environment Setup', 'None ', "Other",
              'Python and Coding', 'Time Management and Motivation']

    # Save the training splits used.
    with open("data-splits/train.csv", "r", encoding="utf-8") as train:
        trn = list(csv.reader(train))
        with open("results/train" + results_file[-15:], "w", encoding="utf-8", newline="") as t:
            c_w = csv.writer(t)
            c_w.writerows(trn)

    # Read the reflections from the test split used for the raw results file.
    with open("data-splits/test.csv", "r", encoding="utf-8") as test:
        tst = list(csv.reader(test))
        tst_refs = [row[0] for row in tst]

    misclassified = []

    # Write the raw results, which include the predicted label and probabilities for each reflection, and each reflection's text.
    with open(raw_results_file, "w", encoding="utf-8", newline="") as rr:
        c_w = csv.writer(rr)
        c_w.writerow(["", "", "", labels[0], labels[1], labels[2], labels[3], labels[4]])
        for pred, ref, probs, i in zip(predictions, tst_refs[1:], probs.numpy().tolist(), range(0,len(predictions))):
            row = [ref, labels[pred], ""]
            row.extend(probs)
            if i in misclassified_idx:
                misclassified.append(row)
            c_w.writerow(row)

    # Raw results file for only misclassified reflections.
    with open("results/misclassified" + results_file[-15:], "w", encoding="utf-8", newline="") as m:
        c_w = csv.writer(m)
        c_w.writerows(misclassified)

    matrix = confusion_matrix(references, predictions, labels=[i for i in range(0, len(labels))])
    report = classification_report(references, predictions, labels=[i for i in range(0, len(labels))], target_names=labels, output_dict=True)
    f1 = f1_score(references, predictions, average="weighted")

    # Write the complete set of metrics plus the confusion matrix for that training iteration.
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

    # Used when maximizing the F1 during the hyperparameter search.
    return {"F1": f1}


def training_iteration(hps=None, do_hp_search=False, train_file="data-splits/train.csv", test_file="data-splits/test.csv"):
    # If this prints False, make sure you have CUDA installed + a CUDA capable GPU + the CUDA version of PyTorch
    print(torch.cuda.is_available())

    dataset = load_dataset('csv', data_files={
        "train": train_file,
        "test": test_file
    })

    dataset["validation"] = dataset["test"]

    if do_hp_search:
        def objective(trial):
            lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
            epochs = trial.suggest_categorical("epochs", [30, 40, 50])
            batch_size = trial.suggest_categorical("batch_size", [8, 16])
            # repeats is a major bottleneck to training time at >4
            # repeats = trial.suggest_categorical("repeats", [4, 5, 6, 7])

            print(f"Learning rate: {lr}")
            print(f"Epoch: {epochs}")
            print(f"Batch size: {batch_size}")
            # print(f"Repeats: {repeats}")

            search_trainer = FastFitTrainer(
                model_name_or_path="sentence-transformers/paraphrase-distilroberta-base-v2",
                learning_rate=lr,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                dataset=dataset,
                optim="adafactor",
                label_column_name="label",
                text_column_name="text",
                max_text_length=128,
                dataloader_drop_last=False,
                num_repeats=4,
                compute_metrics=compute_metrics
            )

            search_trainer.train()
            f1 = search_trainer.evaluate()["eval_F1"]

            print(f"Trial result: {f1}")

            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        best_params = study.best_params
        print(best_params)

        return best_params
    else:
        trainer = FastFitTrainer(
            model_name_or_path="sentence-transformers/paraphrase-distilroberta-base-v2",
            learning_rate=hps["learning_rate"],
            num_train_epochs=hps["num_epochs"],
            per_device_train_batch_size=hps["batch_size"],
            per_device_eval_batch_size=16,
            dataset=dataset,
            optim="adafactor",
            label_column_name="label",
            text_column_name="text",
            max_text_length=128,  # 128 used in FastFit paper
            dataloader_drop_last=False,
            num_repeats=4,  # 4 used in FastFit paper
            compute_metrics=compute_metrics  # <-- see instructions at top of main(), does not work out of the box
        )

        model = trainer.train()

        trainer.evaluate()

        return None


def inference(model):
    pass
    # Currently doesn't work because you can't pass a FastFit model to pipeline() TODO - Fix
    """
    print(model.config_args)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v2")
    classifier = pipeline("text-classification", model=model, config=model.config, tokenizer=tokenizer)
    latest_ref_results = []

    with open("latest_reflection_1.csv", "r", encoding="utf-8") as lr:
        refs = list(csv.reader(lr))
        for ref in refs:
            latest_ref_results.append(classifier(ref))
        with open("latest_reflection_1_results_fastfit.csv", "w", encoding="utf-8", newline="") as abcdefg:
            c_w = csv.writer(abcdefg)
            c_w.writerows(latest_ref_results)
    """


def update_file_paths(results, raw_results, raw_results_probs, cm):
    global results_file
    global raw_results_file
    global raw_results_probs_file
    global confusion_matrix_file
    results_file = results
    raw_results_file = raw_results
    raw_results_probs_file = raw_results_probs
    confusion_matrix_file = cm


# main() defines the experiment sequence.
def main():
    # Instructions: First, you must alter the FastFit source code in a couple ways, as
    # it's outdated in some spots (and really not the most robust library).
    # First, under set_trainer() in the FastFitTrainer class, add the parameter "trust_remote_code=True"
    # to the load_metric function call, which is needed to run a custom compute_metrics.
    # Next, the source code comments insist that you can define your own custom compute_metrics
    # function, but you can't off the shelf, the source is hardcoded to only accept their compute_metrics function
    # by default. Add an attribute compute_metrics to the FastFitTrainer class, which is a Callable.
    # Then, when the Trainer is instantiated in set_trainer(), change class parameter "compute_metrics=compute_metrics"
    # to "compute_metrics=self.compute_metrics" to pass in the custom compute_metrics.
    #
    # Make sure you create directories data-splits, full-datasets, and results in the same directory as model.py.
    # full-datasets should contain the datasets you want to use to create your splits (e.g. SL-R1-80P), and the rest
    # of them should be empty.

    training_times = {}

    ### HP Search(es), comment out if undesired. ###
    # Edit the HP search space in training_iteration().
    do_hp_search = False
    hps = {}

    if do_hp_search:
        start_time = time.time()
        # creates train.csv and test.csv to be trained and tested on
        create_splits(dataset_file="full_datasets/sl-r1-80-proficiency.csv",
                      shot=10,
                      train_file="data-splits/train_80_hps.csv",
                      test_file="data-splits/test_80_hps.csv")
        hps = training_iteration(do_hp_search=True,
                                 train_file="data-splits/train_80_hps.csv",
                                 test_file="data-splits/test_80_hps.csv")
        training_times.update({"hp_search": start_time - time.time()})
        print(hps)
    else:
        # values as defined in FastFit paper https://aclanthology.org/2024.naacl-demo.18.pdf
        hps = {'learning_rate': 1e-5, 'num_epochs': 40, 'batch_size': 16}

    # k = number of times to train/test for experiment evaluation
    k_hp = 10

    start_time = time.time()

    ### K-Fold Evaluation(s) ###
    # Regenerating train/testing splits on every iteration to account for evaluation noise

    # 80, 10 shot
    for k in range(0, k_hp):
        torch.cuda.empty_cache()

        update_file_paths("results/results_r1_80_10_" + str(k) + ".csv",
                          "results/raw_results_r1_80_10_" + str(k) + ".csv",
                          "results/probs_r1_80_10_" + str(k) + ".csv",
                          "results/cm_r1_80_10_" + str(k) + ".png")
        create_splits(dataset_file="full-datasets/sl-r1-80-proficiency.csv", shot=10)
        training_iteration(hps=hps)

    training_times.update({"80, 10 shot": start_time - time.time()})
    print(f"Experiment duration: {start_time - time.time()}")
    start_time = time.time()

    # 100, 10 shot
    for k in range(0, k_hp):
        torch.cuda.empty_cache()

        update_file_paths("results/results_r1_100_10_" + str(k) + ".csv",
                          "results/raw_results_r1_100_10_" + str(k) + ".csv",
                          "results/probs_r1_100_10_" + str(k) + ".csv",
                          "results/cm_r1_100_10_" + str(k) + ".png")
        create_splits(dataset_file="full-datasets/sl-r1-100-proficiency.csv", shot=10)
        training_iteration(hps=hps)

    training_times.update({"100, 10 shot": start_time - time.time()})
    print(f"Experiment duration: {start_time - time.time()}")

    start_time = time.time()

    # 80, 20 shot
    for k in range(0, k_hp):
        torch.cuda.empty_cache()

        update_file_paths("results/results_r1_80_20_" + str(k) + ".csv",
                          "results/raw_results_r1_80_20_" + str(k) + ".csv",
                          "results/probs_r1_80_20_" + str(k) + ".csv",
                          "results/cm_r1_80_20_" + str(k) + ".png")
        create_splits(dataset_file="full-datasets/sl-r1-80-proficiency.csv", shot=20)
        training_iteration(hps=hps)

    training_times.update({"80, 20 shot": start_time - time.time()})
    print(f"Experiment duration: {start_time - time.time()}")
    start_time = time.time()

    # 100, 20 shot
    for k in range(0, k_hp):
        torch.cuda.empty_cache()

        update_file_paths("results/results_r1_100_20_" + str(k) + ".csv",
                          "results/raw_results_r1_100_20_" + str(k) + ".csv",
                          "results/probs_r1_100_20_" + str(k) + ".csv",
                          "results/cm_r1_100_20_" + str(k) + ".png")
        create_splits(dataset_file="full-datasets/sl-r1-100-proficiency.csv", shot=20)
        training_iteration(hps=hps)

    training_times.update({"100, 20 shot": start_time - time.time()})
    print(f"Experiment duration: {start_time - time.time()}")
    start_time = time.time()

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
