# <h1 style="text-align:center">Reflection Classification Toolkit<h1>
## A user-friendly library for student reflection datasets and classification
#### Library written by Andrew Morgan, working with Sandra Wiktor, Eunyoung Kim, and Dr. Mohsen Dorodchi. A part of the student reflection classification project in Dr. Dorodchi's Text Analytics lab at the University of North Carolina at Charlotte.

(preliminary): train_setfit and train_fastfit use CUDA (device="cuda") by default, and requirements.txt attempts to install torch 2.4.1+cu124. If you're on a Mac and using Apple silicon/mps, pass the parameter device="mps" to calls to train_setfit and train_fastfit to change to PyTorch's mps backend. If on a Mac, you shouldn't have to edit requirements.txt to use GPU acceleration with PyTorch.

All functions have docstrings that explain correct usage; use them for guidance and demo.py for examples. Note that this code requires the use of proprietary data.

Unfortunately, due to shortcomings with FastFit and SetFit's interfaces (and a backwards compatibility bug with the transformers library), this code requires specially altered versions of those libraries' source codes to run. Provided in altered_dependency_files are the files that must be replaced in the necessary libraries (after installing them in requirements.txt). 

# Experiment replication instructions:
These are complete instructions on how to implement every part of our methodology that I implemented using this library and high-level descriptions of each important function. This should provide everything you need to easily replicate all experiments with FastFit and SetFit in a small number of function calls.

Please also refer to demo.py for a complete implementation of our entire pipeline.

## Adding reflection sets
    # r1, 2, and 3 (for example)
    reflection_sets = dc.construct_reflection_sets(individual_reflection_datasets=individual_reflection_datasets, reflection_sets=["r1", "r2", "r3"])
    r1 = reflection_sets["r1"]
    r2 = reflection_sets["r2"]
    r3 = reflection_sets["r3"]
Add the desired reflection sets to the "reflection_sets" parameter in the call to construct_reflection_sets() to get access to them. They'll be in the returned dictionary, and you can use them as with r1 and r2 in the demo.
## Changing agreement
    # 100% agreement
    r1_100 = df.filter_dataset(kalpha_agreement_threshold=1.0,
                          single_label=True,
                          full_dataset=r1,
                          label_sets=annotation_label_sets)
Simply change the agreement threshold. This function will approximate the MASI per-reflection agreement threshold such that the Krippendorff's alpha agreement is at least and close to the passed value.
## Adding re-annotated reflections to the reflection sets
    r1_text = [row[0] for row in r1_80]  # row[0] is the text, and row[1] is the label in all single-label datasets
    r1_none_consensus = [row for row in none_consensus if row[0] in r1_text]
    r1_contr_consensus = [row for row in contr_consensus if row[0] in r1_text]

    r1_80 = du.add_new_refs_to_dataset(input_dataset=r1_80, integrate_data=r1_none_consensus)
    r1_80 = du.add_new_refs_to_dataset(input_dataset=r1_80, integrate_data=r1_contr_consensus)
Since our none and controversial reflection reannotations encompass all the reflection sets, you'll have to filter for only the corresponding reannotations.
#### For 100% agreement annotations/reflection sets:
    none_consensus_100 = du.single_label_consensus(input_folder="none_annotations", only_100=True)
    contr_consensus_100 = du.single_label_consensus(input_folder="controversial_annotations", exclude_controversial=True, only_100=True)

    # Then the same as above
This will only include reannotations were the _re-annotators_ all agreed on the new label.
## Augmenting dataset text with reflection questions
    questions = [
    "How do you feel about the course so far?",
    "Explain why you selected the above choice(s).",
    "What was your biggest challenge(s) for the past modules?",
    "How did you overcome this challenge(s)?",
    "Do you have any current challenges in the course? If so, what are they?"
    ]
    r1_80 = du.construct_split_reflection_dataset(input_dataset=r1_80, subreflections=subreflections, questions=questions)
    r1_80 = du.construct_question_prepended_refs(input_dataset=r1_80)
First, specify the questions that the students responded to in the original reflection survey and integrate them into the dataset's text column. 
construct_split_reflection_dataset will split the single text column into n columns (one for each question) based on the subreflections (the students' responses to each question) generated in the early call to construct_multilabel_dataset (at the top of demo.py). 
Then, construct_question_prepended_refs will merge the text columns of the split reflection dataset back into one dataset

## Converting labels to "Other" / Excluding non-proficiency labels
    other_labels = ["Understanding requirements and instructions", "SDLC", "MySQL", "Github",
                "Course Structure and Materials", "Other (Secondary Issue)", "HTML"]

    r1_80, r1_80_other_reflections = du.collapse_into_other(input_dataset=r1_80, other_labels=other_labels)

    other_labels.remove("Github")

    r2_80, r2_80_other_reflections = du.collapse_into_other(input_dataset=r2_80, other_labels=other_labels)
Specify which labels should become "Other" and use collapse_into_other() to alter the dataset so. collapse_into_other() also returns r1_80_other_reflections, which are all reflections converted to "Other" with their original labels. This is used in the next step (label supplementing). 
Remove labels from other_labels as necessary (in the above, remove Github, which isn't considered "Other" in r2).

## Max-shot or partial label supplementing
    other_reflections = pd.concat([r1_80_other_reflections, r2_80_other_reflections])

    supplement_labels = ["IDE and Package Installation", "Time Management and Motivation",
                     "API", "Python and Coding", "None", "Group Work", "Other"]

    all_datasets = {
        "r1_80": copy.deepcopy(r1_80),
        "r2_80": copy.deepcopy(r2_80)
    }

    other_labels.append("Github")

    ## Max-shot supplementing
    r1_80 = du.supplement_label_classes(all_datasets=all_datasets,
                                              supplement_dataset_name="r1_80",
                                              supplement_labels=supplement_labels,
                                              increase_to="max",  # "shot"
                                              other_refs=other_reflections,
                                              other_labels=other_labels)

    ## Partial supplementing
    r1_80 = du.supplement_label_classes(all_datasets=all_datasets,
                                              supplement_dataset_name="r1_80",
                                              supplement_labels=supplement_labels,
                                              increase_to=10,  # "shot"
                                              other_refs=other_reflections,
                                              other_labels=other_labels)
First, we need the un-converted "Other" reflections to maximize the bank of reflections available for supplementing label classes. Also pass in all_datasets to specify where our main bank of supplement reflections comes from. 
Again, you should change what labels are being supplemented based on the reflection set. You should also update other_labels as you update the reflection sets as well.

Please note that we used all reflection sets to supplement labels in our original experiments, and you should do the same when replicating. 
## Converting text to challenge-column only
    r1_80 = du.challenge_column_only(input_dataset=r1_80)
Very simple, just a call to challenge_column_only(). Edits the dataset text in-place.

## Executing kfold experiments and getting raw results
    with open("sl-r1-80.csv", "w", encoding="utf-8", newline="") as d:
        writer = csv.writer(d)
        writer.writerows(r1_80)

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
        do_hp_search=False,
        # device="mps"  # --> uncomment to use Apple silicon GPUs (probably)
    )
Here, you can specify the hyperparameters, k for kfold, models to use, etc. Straightforward, but just make sure you write the dataset you want to experiment with to a csv file, and you use the filename formatting "sl-{ref_name}-{agreement}".
The hyperparameters specified in demo.py are the ones I used for my experiments. Make sure to use those ones if replicating, and also know that the hps parameter only supports specifying those hyperparameters as of now.

Results will automatically be written to a "results" directory, which itself contains "setfit" and "fastfit" directories where the raw results files (including experiment time) will be. All directories are created for you. Here are the results files breakdown:

#### Time log

hps_time.csv: hyperparameters used and experiment time

#### Main files; general file name formatting: {model}-{results-type}-{dataset}-{shot}-{kfold-iteration}.csv

cm: that iteration's confusion matrix \
misclassified: reflections which were misclassified by the model in that iteration, with label class probabilities \
raw_results: raw model predictions in that iteration, with label class probabilities \
results: an sklearn classification_report() for that iteration \
train: the train split for that iteration

## Cleaning up results / getting weighted kfold results
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
This writes a cleaned-up version of the above results to an automatically created directory "weighted-fastfit" or "weighted-setfit". These are the results I ended up reporting to the research group.

This results file has the weighted average results across the kfold iterations (though not necessarily arithmetic mean results; results are calculated using a particular procedure described and justified here: https://docs.google.com/document/d/1ghEzr1rUZz5XLtzdqhWTAHMABWhW-5M1KpfBjqUUBO4/edit?usp=sharing).

\
\
\
Replicating all of my experiments should be just a matter of varying demo.py slightly (e.g., commenting out the label supplementing part for the pure dataset experiments). Feel free to email me with any problems or if anything is unclear.

