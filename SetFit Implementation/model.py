from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import multilabel_confusion_matrix
from optuna import Trial
import numpy
import csv
import torch


# Generate a confusion matrix for each label in the dataset. For each column/vector
# in the label_num by reflection_num matrix of predictions output by the model,
# one confusion matrix will be created. That will represent the confusion for
# that label. Repeat process for each label. Hopefully, with enough predictions
# for each class, a minimally noisy confusion matrix can be created for each label
def compute_metrics(y_pred, y_true) -> dict[str, float]:
    # initialize labels
    labels = ["Python and Coding",
              "Github",
              "Assignments",
              "Time Management and Motivation",
              ]

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
    for label in labels:
        # len(y_true) is the number of reflections used in evaluation
        # acc = num_of_correct_classifications / num_reflections
        accuracy += (result[f"{label}-tp"] + result[f"{label}-tn"]) / len(y_true)
    accuracy /= len(labels)
    result.update({"accuracy": accuracy})
    return result


# model instantiation for each trial run of the hyperparameter search
def model_init(params):
    params = {"multi_target_strategy": "one-vs-rest", "device": torch.device("cuda")}
    # all-MiniLM-L12-v2 is 33.6M params
    return SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2", **params)


# hyperparameters to optimize during hp search
def hp_space(trial: Trial):
    return {
        "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-6, 1e-3, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 3)
    }


def main():
    # Multi-label text classification using Setfit
    # loosely followed https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb

    # Instructions: create a folder called "data-splits" containing "setfit-dataset-train.csv" and setfit-dataset-test.csv", which are generated from the Dataset Construction script
    # Uncomment hyperparameter search code block and comment TrainingArguments code block and "args=args" to run a hyperparameter search
    # Last, change the labels List in compute_metrics if running experiments with different labels than "Python and Coding", "GitHub", "Assignments", and "Time Management"

    # Datasets are generated using the consensus data parser script

    print("Loading datasets...")
    # load two datasets from csv files in dataset dictionary
    dataset = load_dataset('csv', data_files={
        "train": "data-splits/setfit-dataset-train.csv",
        "test": "data-splits/setfit-dataset-test.csv"
    })

    print("Processing datasets...")
    # extract labels
    labels = dataset["train"].column_names
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
    eight_examples_of_each = numpy.concatenate([numpy.random.choice(numpy.where(dataset["train"][label])[0], 10) for label in labels])
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

    # tokenization as specified in the "Fine tuning BERT (and friends)" notebook is not necessary or worthwhile
    # (as far as I know) working with SetFit models. SetFit must tokenize the data behind the scene

    print("Loading model...")

    # only setting initial batch size, hyperparameter search will cover learning rate and num epochs
    args = TrainingArguments(
        batch_size=16,
        body_learning_rate=0.00017039160219643772,  # optimal lr determined through hp search
        num_epochs=1
    )

    # fine tune pretrained model using datasets using default hyperparameters (will change as I run experiments with
    # varying hyperparameters, only running default hps for debugging right now)
    trainer = Trainer(
        model_init=model_init,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        metric=compute_metrics,
        args=args
    )

    print("Training...")
    # optimizing sentence transformer learning rate and num of epochs with hyperparameter search
    """
    best_run = trainer.hyperparameter_search(
        # compute_objective is the overall accuracy of all labels
        direction="maximize",  # maximize accuracy
        hp_space=hp_space,
        compute_objective=lambda result: result.get("accuracy"),
        n_trials=20
    )
    trainer.apply_hyperparameters(best_run.hyperparameters)
    """

    trainer.train()

    print("Testing...")
    metrics = trainer.evaluate()  # confusion data

    # DON'T push to hub for initial pass of experiment
    # model.push_to_hub("setfit-multilabel-test")

    print(metrics)

    with open("metrics.csv", "w") as m:
        c_w = csv.writer(m)
        for key in metrics.keys():
            arr = [key, metrics[key]]
            c_w.writerow(arr)
    print("Metrics data written to metrics.csv")


if __name__ == "__main__":
    main()
