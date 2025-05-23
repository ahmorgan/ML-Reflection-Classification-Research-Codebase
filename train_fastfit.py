from datasets import load_dataset
from fastfit import FastFitTrainer, FastFit
from transformers import AutoTokenizer, pipeline
import random
import csv
import torch
import time
import numpy as np
import gc
import datetime
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import optuna
import os


results_file = "results/fastfit/results.csv"
raw_results_file = "results/fastfit/raw_results.csv"
raw_results_probs_file = "results/fastfitraw_results_probabilities.csv"
confusion_matrix_file = "results/fastfit/confusion_matrix.png"

current_model = ""

dataset_label_set = []


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

        # FastFit internally treats the string label "None" as None (as in the null value),
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
        # order FastFit uses internally


def compute_metrics(p) -> dict[str, float]:
    global current_model
    predictions = (p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions)
    probs = torch.softmax(torch.tensor(predictions), dim=1)
    predictions = np.argmax(predictions, axis=1)

    references = p.label_ids

    misclassified_idx = []

    for i in range(0, len(references)):
        if predictions[i] != references[i]:
            misclassified_idx.append(i)

    print(probs)
    print(predictions)

    # labels = ['IDE and Environment Setup', 'None ', "Other",
    #           'Python and Coding', 'Time Management and Motivation']

    with open("data-splits/train.csv", "r", encoding="utf-8") as train:
        trn = list(csv.reader(train))
        with open(f"results/fastfit/{current_model}_train" + results_file[-15:], "w", encoding="utf-8", newline="") as t:
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

        for pred, ref, probs, i in zip(predictions, tst_refs[1:], probs.numpy().tolist(), range(0,len(predictions))):
            row = [ref, dataset_label_set[pred], ""]
            row.extend(probs)
            if i in misclassified_idx:
                misclassified.append(row)
            c_w.writerow(row)

    with open("results/fastfit/" + current_model + "_misclassified" + results_file[-15:], "w", encoding="utf-8", newline="") as m:
        c_w = csv.writer(m)
        c_w.writerows(misclassified)

    matrix = confusion_matrix(references, predictions, labels=[i for i in range(0, len(dataset_label_set))])
    report = classification_report(references, predictions, labels=[i for i in range(0, len(dataset_label_set))], target_names=dataset_label_set, output_dict=True)
    f1 = f1_score(references, predictions, average="weighted")

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


def _training_iteration(hps=None, do_hp_search=False, train_file="data-splits/train.csv", test_file="data-splits/test.csv", device="cuda"):
    # Instructions: First, you must alter the FastFit source code in a couple ways, as
    # it's outdated in some spots (and really not the most robust library).
    # First, under set_trainer() in the FastFitTrainer class, add the parameter "trust_remote_code=True"
    # to the load_metric function call, which is needed to run a custom compute_metrics.
    # Next, the source code comments insist that you can define your own custom compute_metrics
    # function, but you can't off the shelf, the source is hardcoded to only accept their compute_metrics function
    # by default. Add an attribute compute_metrics to the FastFitTrainer class, which is a Callable.
    # Then, when the Trainer is instantiated in set_trainer(), change class parameter "compute_metrics=compute_metrics"
    # "compute_metrics=self.compute_metrics" to pass in the custom compute_metrics.
    #
    # After that, make sure that you have the full dataset in the same directory as model.py, which
    # should be called "low_disagreement_dataset.csv". create_splits will divide the dataset into
    # train and test splits based on the shot variable, which is how many examples per label class will
    # be in train (ie a shot of 10 means 10 example reflections for each label class in train.csv).
    # The rest of the reflections in the dataset will go to the test split.
    # You can also run a hyperparameter search by uncommenting the code below objective() -- if needed,
    # alter the search space by changing the arguments to suggest_float() and suggest_categorical() in objective().
    # Hyperparameters can also be set manually in the FastFitTrainer constructor call.

    # If this prints False, make sure you have CUDA installed + a CUDA capable GPU + the CUDA version of PyTorch
    if device == "mps":
        print(f"Running on Apple silicon GPU: {torch.backends.mps.is_available()}")
    else:
        print(f"Running on CUDA GPU: {torch.cuda.is_available()}")

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
                num_repeats=4,  # number suggested by the FastFit developers
                compute_metrics=compute_metrics,
                device=device
            )

            search_trainer.train()
            f1 = search_trainer.evaluate()["eval_F1"]

            print(f"Trial result: {f1}")

            return f1

        # In a study of 20 trials, I found that 7e-5 base learning rate and 50 epochs was optimal
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        best_params = study.best_params
        print(best_params)

        return best_params
    else:
        trainer = FastFitTrainer(
            model_name_or_path=f"sentence-transformers/{current_model}",
            learning_rate=hps["body_learning_rate"],  # best_params["lr"],
            num_train_epochs=hps["num_epochs"],  # best_params["epochs"],
            per_device_train_batch_size=hps["batch_size"],
            per_device_eval_batch_size=hps["batch_size"],
            dataset=dataset,
            optim="adafactor",
            label_column_name="label",
            text_column_name="text",
            max_text_length=128,  # 128 suggested by FastFit developers
            dataloader_drop_last=False,
            num_repeats=4,  # best_params["repeats"]
            compute_metrics=compute_metrics,  # <-- see instructions at top of main()
            device=device
        )

        trainer.train()

        trainer.evaluate()

        del trainer
        del dataset
        gc.collect()

        return None


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
    if not inference_dataset:
        raise ValueError("Please specify at least a dataset for inference.")
    if not inference_hps:
        inference_hps = {'body_learning_rate': 1e-5, 'num_epochs': 40, 'batch_size': 16}
    training_times = {}
    start_time = time.time()
    variation = inference_variation.split("_")
    reflection_set = variation[0]
    agreement = variation[1]
    shot = variation[2]

    global current_model
    current_model = inference_model
    _create_splits(dataset_file=f"full-datasets/sl-{reflection_set}-{agreement}-proficiency.csv", shot=shot)
    _update_file_paths(f"results/fastfit/{inference_dataset}-results_{reflection_set}_{agreement}_{shot}.csv",
                       f"results/fastfit/{inference_dataset}-raw_results_{reflection_set}_{agreement}_{shot}.csv",
                       f"results/fastfit/{inference_dataset}-probs_{reflection_set}_{agreement}_{shot}.csv",
                       f"results/fastfit/{inference_dataset}-cm_{reflection_set}_{agreement}_{shot}.png")
    _training_iteration(hps=inference_hps, test_file=f"{inference_dataset}.csv", device=device)
    training_times.update({f"{inference_dataset}_{reflection_set}_{agreement}_{shot}": datetime.timedelta(
        seconds=time.time() - start_time)})
    if device == "cuda":
        torch.cuda.empty_cache()

    with open(f"results/fastfit/{inference_dataset}_time.csv", "w", encoding="utf-8", newline="") as ifd:
        c_w = csv.writer(ifd)
        c_w.writerows(list(training_times.items()))

    return None


def _update_file_paths(results, raw_results, raw_results_probs, cm):
    global results_file
    global raw_results_file
    global raw_results_probs_file
    global confusion_matrix_file
    results_file = results
    raw_results_file = raw_results
    raw_results_probs_file = raw_results_probs
    confusion_matrix_file = cm


def fastfit_experiment(dataset_file_name, shot, k_hp, hps, models, do_hp_search, device="cuda"):
    """
    Runs a single k-fold FastFit experiment based on the specified parameters. To input datasets, make sure you have the dataset
    as a csv file saved locally, and pass in the dataset's file name to dataset_file_name.

    All experiment results are written to a folder "results", which is created automatically.
    Result file names are coded by, "results/fastfit/{model}_{results_type}_{reflection_set}_{agreement}_{shot}_{k-fold_iteration}". Training
    times and hyperparameters used are also saved to the results folder under the filename "hps_time.csv".

    Can also run a hyperparameter search, in which case the best run's hyperparameters are returned.

    :param dataset_file_name: path to dataset, must have file name format: "sl-{reflection_set}-{agreement}.csv" (e.g., "sl-r1-80.csv").
    :param shot: number of examples per label class in training split
    :param k_hp: k hyperparameter for k-fold
    :param hps: dictionary of structure {'body_learning_rate': _, 'num_epochs': _, 'batch_size': _}, respective suggestions: 1e-5, 40, 16
    :param do_hp_search: whether or not to do a hyperparameter search
    :param models: list of names of Sentence Transformers available through Hugging Face
    :param device: "cuda" by default, can be "mps"
    :return: None or a dictionary of hyperparameters if do_hp_search=True
    """
    # Experiment sequence
    # HP searches -> apply hyperparameters
    # k-fold experiments

    if not os.path.isdir("results"):
        os.mkdir("results")
    if not os.path.isdir("results/setfit"):
        os.mkdir("results/setfit")
    if not os.path.isdir("data-splits"):
        os.mkdir("data-splits")

    if "sl" not in dataset_file_name:
        raise ValueError("Please format your dataset file name in the structure, 'sl-{reflection_set}-{agreement}'!")
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
                                  test_file="data-splits/test_80_hps.csv",
                                  device=device)
        training_times.update({"hp_search": datetime.timedelta(seconds=start_time - time.time())})
        return hps

    global current_model

    ### K-Fold Evaluation(s) ###
    # Regenerating train/testing splits on every iteration to account for evaluation noise
    for model in models:
        current_model = model
        for k in range(0, k_hp):
            if device == "cuda":
                torch.cuda.empty_cache()

            # update paths used in compute_metrics() to log results
            # would pass these as parameters to compute_metrics() but
            # compute_metrics() is called by setfit internally. these have to be global variables
            _update_file_paths(f"results/fastfit/{current_model}_results_{dataset_name}_{str(shot)}_" + str(k) + ".csv",
                               f"results/fastfit/{current_model}_raw_results_{dataset_name}_{str(shot)}_" + str(k) + ".csv",
                               f"results/fastfit/{current_model}_probs_{dataset_name}_{str(shot)}_" + str(k) + ".csv",
                               f"results/fastfit/{current_model}_cm_{dataset_name}_{str(shot)}_" + str(k) + ".png")
            _create_splits(dataset_file=dataset_file_name, shot=shot)
            _training_iteration(hps=hps, device=device)

        training_times.update({f"{dataset_name}, {current_model}, 80, 10 shot": datetime.timedelta(seconds=time.time() - start_time)})
        print(f"Experiment duration: {datetime.timedelta(seconds=time.time() - start_time)}")

    with open(f"results/fastfit/hps_time.csv", "w", encoding="utf-8", newline="") as hpt:
        c_w = csv.writer(hpt)
        c_w.writerows(list(training_times.items()))
        c_w.writerows(list(hps.items()))

