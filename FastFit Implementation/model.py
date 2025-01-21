from datasets import load_dataset
from fastfit import FastFitTrainer
import random
import csv
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import optuna
import os


def create_splits(shot):
    # create 80/20 train and test splits
    with open("low_disagreement_dataset.csv", "r", encoding="utf-8", newline="") as ds:
        c_r = list(csv.reader(ds))
        c_r = c_r[1:]
        random.shuffle(c_r)

        # FastFit internally treats the string label "None" as None (as in the null value),
        # so circumvent that by changing the name of the label to No Issue
        for row in c_r:
            if row[1] == "None":
                row[1] = "No Issue"

        labels = [row[1] for row in c_r]

        train = []
        for label in set(labels):
            if labels.count(label) < shot:
                continue
            count = 0
            for row in c_r:
                if row[1] == label:
                    train.append(row)
                    count += 1
                if count == 10:
                    break
        train_labels = [row[1] for row in train]
        test = [row for row in c_r if (row not in train and row[1] in set(train_labels))]

        for row in train:
            assert row not in test, "Test contains reflections from train!"

        for label in set(train_labels):
            assert train_labels.count(label) == 10, "Train does not contain ten of each label!"

        test_labels = [row[1] for row in test]
        for label in set(test_labels):
            print(f"{label} label count in test: {test_labels.count(label)}")

        with open("test.csv", "w", encoding="utf-8", newline="") as tst:
            c_w = csv.writer(tst)
            c_w.writerow(["text", "label"])
            c_w.writerows(test)

        with open("train.csv", "w", encoding="utf-8", newline="") as trn:
            c_w = csv.writer(trn)
            c_w.writerow(["text", "label"])
            c_w.writerows(train)


def compute_metrics(p) -> dict[str, float]:
    predictions = (p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions)
    predictions = np.argmax(predictions, axis=1)

    references = p.label_ids

    labels = ["API", 'Course Structure and Materials', 'Github', 'Group Work', 'MySQL', 'No Issue',
              'Python and Coding', 'Time Management and Motivation']

    matrix = confusion_matrix(predictions, references, labels=[i for i in range(0, max(references)+1)])
    report = classification_report(predictions, references, labels=[i for i in range(0, max(references)+1)], target_names=labels, output_dict=True)
    f1 = f1_score(predictions, references, average="macro")

    with open("results.csv", "w", encoding="utf-8", newline="") as results:
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
    plt.show()

    return {"F1": f1}


def main():
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
    print(torch.cuda.is_available())

    # how many samples to select per label class, ie "10-shot" or "5-shot"
    shot = 10

    if not os.path.exists("test.csv") and not os.path.exists("train.csv"):
        print("Generating splits...")
        # create_splits(shot)

    dataset = load_dataset('csv', data_files={
        "train": "train.csv",
        "test": "test.csv"
    })

    dataset["validation"] = dataset["test"]

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        epochs = trial.suggest_categorical("epochs", [40, 50, 60])
        # repeats is a major bottleneck to training time at >4
        # repeats = trial.suggest_categorical("repeats", [4, 5, 6, 7])

        print(f"Learning rate: {lr}")
        print(f"Epoch: {epochs}")
        # print(f"Repeats: {repeats}")

        search_trainer = FastFitTrainer(
            model_name_or_path="sentence-transformers/all-mpnet-base-v2",
            learning_rate=lr,
            num_train_epochs=epochs,
            dataset=dataset,
            optim="adafactor",
            label_column_name="label",
            text_column_name="text",
            max_text_length=128,
            dataloader_drop_last=False,
            num_repeats=4,  # number suggested by the FastFit developers
            compute_metrics=compute_metrics
        )

        search_trainer.train()
        f1 = search_trainer.evaluate()["eval_F1"]

        print(f"Trial result: {f1}")

        return f1

    """
    # Uncomment to run hyperparameter search (optimizing the f1 score)
    # In a study of 20 trials, I found that 7e-5 base learning rate and 50 epochs was optimal
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    """

    # Looking at the FastFit source code, the device is set to cuda internally
    # We don't have to set it ourselves like with SetFit
    trainer = FastFitTrainer(
        model_name_or_path="sentence-transformers/all-mpnet-base-v2",
        learning_rate=7e-5,  # best_params["lr"],
        num_train_epochs=50,  # best_params["epochs"],
        dataset=dataset,
        optim="adafactor",
        label_column_name="label",
        text_column_name="text",
        max_text_length=128,  # 128 suggested by FastFit developer
        dataloader_drop_last=False,
        num_repeats=4,  # best_params["repeats"]
        compute_metrics=compute_metrics  # <-- see instructions at top of main()
    )

    trainer.train()

    print(torch.cuda.memory_summary())

    trainer.evaluate()


if __name__ == "__main__":
    main()
