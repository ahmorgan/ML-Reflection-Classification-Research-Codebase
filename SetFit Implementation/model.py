from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
from optuna import Trial
import numpy
import csv
import torch
from matplotlib import pyplot as plt


# Generate a confusion matrix for each label in the dataset. For each column/vector
# in the label_num by reflection_num matrix of predictions output by the model,
# one confusion matrix will be created. That will represent the confusion for
# that label. Repeat process for each label. Hopefully, with enough predictions
# for each class, a minimally noisy confusion matrix can be created for each label
def compute_metrics(y_pred, y_true) -> dict[str, float]:
    # initialize labels
    labels = ["API", 'Course Structure and Materials', 'Github', 'Group Work', 'MySQL', 'No Issue',
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
        matrix = confusion_matrix(y_true, y_pred, labels=[i for i in range(0, max(y_true) + 1)])
        report = classification_report(y_true, y_pred, labels=[i for i in range(0, max(y_pred) + 1)],
                                       target_names=labels, output_dict=True)
        f1 = f1_score(y_true, y_pred, average="macro")

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

        with open("raw_results.csv", "w", encoding="utf-8", newline="") as rr:
            c_w = csv.writer(rr)
            for pred in y_pred:
                c_w.writerow([labels[pred]])

        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
        display.plot()
        plt.show()
        return {"F1": f1}


# model instantiation for each trial run of the hyperparameter search
def model_init(params):
    params = {  # "multi_target_strategy": "one-vs-rest",
              "device": torch.device("cuda")}
    # all-MiniLM-L12-v2 is 33.6M params
    return SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2", **params)


# hyperparameters to optimize during hp search
def hp_space(trial: Trial):
    return {
        "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-5, 1e-3, log=True),
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

    # In the single label case, the data is already prepared for classification

    # tokenization as specified in the "Fine tuning BERT (and friends)" notebook is not necessary or worthwhile
    # (as far as I know) working with SetFit models. SetFit must tokenize the data behind the scene

    print("Loading model...")

    # only setting initial batch size, hyperparameter search will cover learning rate and num epochs
    args = TrainingArguments(
        batch_size=8,
        body_learning_rate=0.0001037,  # optimal lr determined through hp search
        num_epochs=2
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
    """
    # optimizing sentence transformer learning rate and num of epochs with hyperparameter search
    best_run = trainer.hyperparameter_search(
        # compute_objective is the overall accuracy of all labels
        direction="maximize",  # maximize accuracy
        hp_space=hp_space,
        compute_objective=lambda result: result.get("F1"),
        n_trials=20
    )
    """
    # trainer.apply_hyperparameters(best_run.hyperparameters)

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

    print(torch.cuda.memory_summary())


if __name__ == "__main__":
    main()
