import csv
import dataset_construction as dc
import dataset_filtering as df
import dataset_utilities as du
import train_fastfit
import train_setfit
import weighted_kfold_results as wkf

import pandas as pd
import copy

"""
Reflection Classification Library

written by Andrew Morgan
a part of Dr. Mohsen Dorodchi's Text Analytics Lab at UNC Charlotte, for the student reflection classification project

To run the demo, please put the data here in your working directory (private to only current reflection classification
researchers as of 5/20/25): https://drive.google.com/drive/folders/1Nicswyyy57tw3YmIXx4gTX3QxdmXB6TD?usp=sharing

A complete implementation of our reflection classification pipeline. Provides a simple interface for dataset construction/preprocessing 
and SetFit/FastFit training. Could be a sort of neat, operationalized final product attached to our paper.
We could add Eunyoung's clustering/transformers code and Sandra's GPT-4o code in the future.
"""

exclude_labels = ["Assignments", "Quizzes", "Learning New Material", "Personal Issue"]
label_category = "Primary"

# Reads unfiltered Excel file annotations from directory raw_data
full_multi_label_dataset, individual_reflection_datasets, annotation_label_sets, subreflections = dc.construct_multilabel_datasets(exclude_labels=exclude_labels,
                                                                                                                                   label_category=label_category)

# Reflection set construction
print(individual_reflection_datasets.keys())
print()
r1 = []
r2 = []
for ref_set in individual_reflection_datasets.keys():
    if ref_set[-1] == "1":  # reflection 1
        if not r1:
            r1.extend(individual_reflection_datasets[ref_set])  # individual_reflection_datasets maps reflection names to the consensus dataset for that reflection
        else:  # avoid writing the header twice
            r1.extend(individual_reflection_datasets[ref_set][1:])
    if ref_set[-1] == "2":  # reflection 2
        if not r2:
            r2.extend(individual_reflection_datasets[ref_set])
        else:
            r2.extend(individual_reflection_datasets[ref_set][1:])

# Agreement will be increased to at least the threshold
r1_80 = df.filter_dataset(kalpha_agreement_threshold=0.8,
                          single_label=True,
                          full_dataset=r1,
                          label_sets=annotation_label_sets)  # filter_dataset() automatically matches label_sets to full_dataset,
                          # as long as the reflections in full_dataset are a subset of those in label_sets

r2_80 = df.filter_dataset(kalpha_agreement_threshold=0.8,
                          single_label=True,
                          full_dataset=r2,
                          label_sets=annotation_label_sets)

### Handling new annotations

# Generate datasets with consensus labels for the none and controversial annotations
none_consensus = du.single_label_consensus(input_folder="none_annotations")
contr_consensus = du.single_label_consensus(input_folder="controversial_annotations")

# Fiddly nuance: our none and controversial annotations contains reflections from r1,2,3, and 4, so get only the reannotated reflections from r1
# Ideally, re-annotations would already be exclusive to a single reflection set
r1_text = [row[0] for row in r1_80]
r1_none_consensus = [row for row in none_consensus if row[0] in r1_text]
r1_contr_consensus = [row for row in contr_consensus if row[0] in r1_text]

### Integrate the none and controversial reannotations into the dataset
r1_80 = du.add_new_refs_to_dataset(input_dataset=r1_80, integrate_data=r1_none_consensus)
r1_80 = du.add_new_refs_to_dataset(input_dataset=r1_80, integrate_data=r1_contr_consensus)

# Do the same thing for r2
r2_text = [row[0] for row in r2_80]
r2_none_consensus = [row for row in none_consensus if row[0] in r2_text]
r2_contr_consensus = [row for row in contr_consensus if row[0] in r2_text]

r2_80 = du.add_new_refs_to_dataset(input_dataset=r2_80, integrate_data=r2_none_consensus)
r2_80 = du.add_new_refs_to_dataset(input_dataset=r2_80, integrate_data=r2_contr_consensus)

### Alter reflection text to include the questions the student is responding to, and the student's subresponses

questions = [
    "How do you feel about the course so far?",
    "Explain why you selected the above choice(s).",
    "What was your biggest challenge(s) for the past modules?",
    "How did you overcome this challenge(s)?",
    "Do you have any current challenges in the course? If so, what are they?"
]

print("Constructing split reflection dataset...")
# Split the reflections in the target dataset into sub-responses (basically, integrate the subreflections into the dataset)
r1_80 = du.construct_split_reflection_dataset(input_dataset=r1_80, subreflections=subreflections, questions=questions)
r2_80 = du.construct_split_reflection_dataset(input_dataset=r2_80, subreflections=subreflections, questions=questions)

print("Converting split reflection dataset into a question prepended dataset...")
# Then prepend the questions each sub-response is answering to each sub-response (convert the multiple columns with student responses
# into one integrated text column in the format "{question} {answer} {question} {answer} ... ")
r1_80 = du.construct_question_prepended_refs(input_dataset=r1_80)
r2_80 = du.construct_question_prepended_refs(input_dataset=r2_80)

print("Converting dataset to proficiency label dataset (converting labels to 'Other')...")

other_labels = ["Understanding requirements and instructions", "SDLC", "MySQL", "Github",
                "Course Structure and Materials", "Other (Secondary Issue)", "HTML"]

r1_80, r1_80_other_reflections = du.collapse_into_other(input_dataset=r1_80, other_labels=other_labels)

other_labels.remove("Github")  # reflection two does not have "Github" as "Other"

r2_80, r2_80_other_reflections = du.collapse_into_other(input_dataset=r2_80, other_labels=other_labels)

other_reflections = pd.concat([r1_80_other_reflections, r2_80_other_reflections])

print(other_reflections.head())

supplement_labels = ["IDE and Package Installation", "Time Management and Motivation",
                     "API", "Python and Coding", "None", "Group Work", "Other"]

# Though you could supplement from all reflection sets if you wanted
all_datasets = {
    "r1_80": copy.deepcopy(r1_80),
    "r2_80": copy.deepcopy(r2_80)
}

other_labels.append("Github")
# "max-shot": adds labels from other reflections to each label class from supplement_labels for each reflection
r1_80 = du.supplement_label_classes(all_datasets=all_datasets,
                                              supplement_dataset_name="r1_80",
                                              supplement_labels=supplement_labels,
                                              increase_to="max",  # "shot"
                                              other_refs=other_reflections,
                                              other_labels=other_labels)

# configure for reflection 2 (Github != "Other" in r2)
other_labels.remove("Github")
supplement_labels.append("Github")

r2_80 = du.supplement_label_classes(all_datasets=all_datasets,
                                              supplement_dataset_name="r2_80",
                                              supplement_labels=supplement_labels,
                                              increase_to="max",  # "shot"
                                              other_refs=other_reflections,
                                              other_labels=other_labels)

r1_80 = du.challenge_column_only(input_dataset=r1_80)
r2_80 = du.challenge_column_only(input_dataset=r2_80)

with open("sl-r1-80.csv", "w", encoding="utf-8", newline="") as d:
    writer = csv.writer(d)
    writer.writerows(r1_80)

# You can also write r2_80 to a file and run experiments with it too; just r1_80 for demonstration

# Executes a single k-fold experiment.
# Results written to automatically created /results/ directory.
train_fastfit.fastfit_experiment(
    dataset_file_name="sl-r1-80.csv",
    shot=10,
    k_hp=10,
    hps={
        "body_learning_rate": 1e-5,
        "num_epochs": 40,
        "batch_size": 16
    },
    models=[
        "paraphrase-distilroberta-base-v2",
        "stsb-roberta-base-v2"
    ],
    do_hp_search=False
)

train_setfit.setfit_experiment(
    dataset_file_name="sl-r1-80.csv",
    shot=10,
    k_hp=10,
    hps={
        "body_learning_rate": 2e-5,
        "num_epochs": 1,
        "batch_size": 16
    },
    models=[
        "paraphrase-distilroberta-base-v2",
        "stsb-roberta-base-v2"
    ],
    do_hp_search=False
)

# Calculate the weighted kfold metrics for each set of raw kfold results
wkf.weighted_kfold_metrics(results_folder="results/setfit",
                           models=[
                               "paraphrase-distilroberta-base-v2",
                               "stsb-roberta-base-v2"
                           ],
                           reflection_sets=[
                               "r1"
                           ],
                           variations=[
                               "80_10"
                           ],
                           k=2
                           )

wkf.weighted_kfold_metrics(results_folder="results/fastfit",
                           models=[
                               "paraphrase-distilroberta-base-v2",
                               "stsb-roberta-base-v2"
                           ],
                           reflection_sets=[
                               "r1"
                           ],
                           variations=[
                               "80_10"
                           ],
                           k=2
                           )




