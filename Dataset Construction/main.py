# DATASET PREPROCESSOR: goes from unprocessed annotation data to multi-label dataset
# for training a model for multi-label classification

import csv
import pandas as pd
import numpy as np
from collections import Counter
import organize
from pathlib import Path
import os, os.path


# there might be a cleaner way to do the following that doesn't involve
# 2 dicts but this works
issue2integer = {
    "None": 0,
    "Python and Coding": 1,
    "Github": 2,
    "MySQL": 3,
    "Assignments": 4,
    "Quizzes": 5,
    "Understanding requirements and instructions": 6,
    "Learning New Material": 7,
    "Course Structure and Materials": 8,
    "Time Management and Motivation": 9,
    "Group Work": 10,
    "IDE and Package Installation": 11,
    "API": 12,
    "FastAPI": 12,
    "Personal Issue": 13,
    "HTML": 14,
    "SDLC": 15,
}

integer2issue = {
    0: "None",
    1: "Python and Coding",
    2: "Github",
    3: "MySQL",
    4: "Assignments",
    5: "Quizzes",
    6: "Understanding requirements and instructions",
    7: "Learning New Material",
    8: "Course Structure and Materials",
    9: "Time Management and Motivation",
    10: "Group Work",
    11: "IDE and Package Installation",
    12: "API",
    13: "Personal Issue",
    14: "HTML",
    15: "SDLC",
}

primary_labels = []


# METHOD PARAMETERS
# files: the intermediary annotations obtained from the datasets in raw_data, generated in organize.py
# output_file: name to assign to file which contains the final consensus dataset
# include_other: whether or not to include the "other" column. If not, any reflection with the "other"
#   label will remain in the dataset, just stripped of the "other" label
def construct_dataset(files, output_file, dataset_name):
    print(f"Processing files in {dataset_name}...\n")
    # CONSENSUS LABELING METHODOLOGY
    # **************************************
    # data will be a dictionary where each reflection is mapped to a set of every label
    # assigned to it by every labeler that labeled that reflection. e.g. [0,0,0,1,1,2]
    # where 0 == Python and Coding, 1 == GitHub, and 2 == Assignments, though label-integer
    # mapping depend on what labels are excluded. After every label for every reflection has been
    # tallied, we will resolve each list of labels to n "consensus labels", where n is the length of
    # consensus label set. n is calculated as the mean length of each of the label sets created by
    # every labeler, i.e. if labeler one said [0] and labeler two said [0,1,2], n would be two. The
    # top n labels are chosen from the complete list, e.g. with [0,0,0,1,1,2], the top 2 would be
    # [0,1] and that is determined to be the "consensus label set" for that reflection. In the case
    # of a tie when determining the top n labels, i.e. [0,0,0,1,1,3,3] with n=2, just choose either,
    # so for that example, the consensus label set would be [0,1] or [0,3] (both are assumed to be
    # equally valid. top_n_labels keeps track of the length of each individual label set to determine n.
    # **************************************
    data = {}

    # keeping track of all reflections w/o duplicates to add to
    # final processed dataset.
    # reflections instantiated every time a new
    # reflection is iterated over, except for the first one
    # which is added in a special case
    reflections = []

    # Iterate through all annotation files, encode the labels to integers, and concatenate the single-label rows into
    # label lists for each reflection
    for file in files:
        with open(file, "r", encoding="utf-8") as annotation:
            c_r = csv.reader(annotation)
            c_r = list(c_r)
            # replace issue strings with mapped integers
            for i in range(0, len(c_r)):
                if c_r[i][1] in issue2integer.keys() and c_r[i][1] != "Other":
                    c_r[i][1] = issue2integer[c_r[i][1].strip()]
                else:
                    c_r[i][1] = max(list(issue2integer.values()))+1  # issues in exclude_labels will be mapped to next consecutive integer
                    # and be removed later
                    # (16 in case of no excluded labels)
            c_r.append(["", -1])  # dummy list entry, so the last reflection is included in the loop below
            current_reflection = c_r[0][0]
            reflection_labels = []  # n labels chosen by annotator for that reflection
            i = 0
            for row in c_r:
                if current_reflection == "😍Excited <name omitted> is the best <name omitted> is the best <name omitted> is the best <name omitted> is the best":
                    print(f"found in {file}")
                # row[0] == reflection text, row[1] == label
                if current_reflection != row[0]:
                    # create a new entry in the dictionary if one doesn't exist for the current reflection
                    if current_reflection not in data.keys():
                        data.update({current_reflection: [reflection_labels]})  # new reflection_labels array
                        # refs_labelsets needed to calculate inter-annotator disagreement
                        reflections.append(current_reflection)  # current_reflection replaced, add to reflections
                    else:
                        data[current_reflection].append(reflection_labels)
                    i = 0
                    reflection_labels = []
                    current_reflection = row[0]
                reflection_labels.append(row[1])
                i += 1

    refs_labelsets = {}

    # resolve label lists to consensus label list for each reflection
    for key in data.keys():
        # https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list

        for i in range(0, len(data[key])):
            while 1+max(list(issue2integer.values())) in data[key][i]:
                data[key][i].remove(1+max(list(issue2integer.values())))  # remove all integers mapping to an excluded label
        data[key] = [list(set(l_set)) for l_set in data[key] if l_set]  # remove empty label sets after removing excluded labels

        label_sets = []
        for label_set in data[key]:
            label_sets.append([integer2issue[label] for label in label_set])
        refs_labelsets.update({key: label_sets})

        # choose the top n labels from the list of labels, where n is the mean number of labels for that reflection
        # 1 in the case of every label being excluded, just to avoid a divide by zero error (that reflection will not be used)
        n = 1 if len(data[key]) == 0 else round(sum([len(l_set) for l_set in data[key]]) / len(data[key]))

        # flatten 2d list of label sets
        data[key] = [label for l_set in data[key] for label in l_set]

        if data[key]:
            c = Counter(data[key]).most_common(n)
            data[key].clear()
            for i in range(0, n):
                if i >= len(c):
                    break
                data[key].append(c[i][0])
            for j in range(0, len(data[key])):
                data[key][j] = integer2issue[data[key][j]]
        # if the reflection corresponding to data[key] contains only excluded reflections, mark it with a -1
        else:
            data[key] = [-1]

    # array of zeroes of size num_labels*num_reflections
    # (pycharm is requiring me to give the type hint that the dataset is an ndarray
    # to allow me to index into it, no idea why.)
    dataset = np.zeros((len(data), len(integer2issue)), dtype=np.int8)  # type: np.ndarray

    # populate final dataset with the consensus labels
    i = 0
    for label_set in data.values():
        # case where all the labels are in exclude_labels, do not include in final dataset
        if label_set[0] == -1:
            i += 1
            continue
        for val in label_set:
            dataset[i][issue2integer[val.strip()]] = 1
        i += 1

    # reflections_new is every reflection that has at least one label in dataset
    reflections_new = []
    for i in range(0, len(dataset)):
        if dataset[i].any():
            reflections_new.append(reflections[i])
    # (https://stackoverflow.com/questions/11188364/remove-zero-lines-2-d-numpy-array)
    dataset = dataset[~np.all(dataset == 0, axis=1)]  # nice one liner to remove all lists of zeroes from 2d ndarray
    column_names = list(integer2issue.values())

    # https://stackoverflow.com/questions/20763012/creating-a-pandas-dataframe-from-a-numpy-array-how-do-i-specify-the-index-colum
    df = pd.DataFrame(data=dataset, columns=column_names)
    # append column for reflection associated with each set of labels
    df.insert(len(dataset[0]), "text", reflections_new)

    df.to_csv(output_file, index=False)

    # return new reflections for sanitize_gpt_reflections() and the the label sets
    # and their corresponding reflections
    package = [reflections_new, [list(item) for item in refs_labelsets.items()]]

    return package


# Remove reflections from gpt_reflections.csv that do not correspond to the excluded labels
# parameter reflection is list of reflections with corresponding labels
# method returns None
# METHOD PARAMETERS:
# reflections: reflections_new from process()
def align_subreflections(reflections):
    print(len(reflections))
    # full_refs is a list of every reflection with no label exclusions
    # which will be populated by gpt_reflections.csv
    full_refs = []
    # copy full reflection data w/o sub-responses
    # with no label exclusion from gpt_reflections.csv into full_refs
    with open("gpt_reflections.csv", "r", encoding="utf-8") as gpt:
        c_r = list(csv.reader(gpt))
        print(len(c_r))
        for row in c_r:
            ref = " ".join(row)
            # concatenate each part in full_refs
            """
            for i in range(0, len(row)):
                if i == len(row)-1:
                    ref += row[i]
                else:
                    ref += row[i] + ' '
            """
            print(f"Reflection: {ref}")
            if len(full_refs) > 0:
                print(f"Last in full_refs: {' '.join(full_refs[-1])}")
            else:
                print("First")
            if len(full_refs) == 0 or " ".join(full_refs[-1]) != ref:
                print(f"Appended ref: {ref}")
                full_refs.append(row)
            else:
                print("Ref skipped!")
            print()
    print(len(full_refs))
    # indices of rows to remove from dataframe of gpt_reflections
    # each row referenced in remove is one in full_refs
    remove = []
    # locate rows which are in full_refs but not in the label-truncated
    # reflections parameter, read their indices into remove
    for i in range(0, len(full_refs)):
        if " ".join(full_refs[i]) not in reflections:
            remove.append(i)
    # dataframe of original gpt_reflections, each row contains the 5 sub-responses making
    # up the entire reflections
    df = pd.DataFrame(full_refs)
    # drop undesired rows simultaneously and overwrite gpt_reflections.csv with new truncated data
    print(len(df))
    if remove:
        df.drop(remove, inplace=True)
    print(len(df))
    df.to_csv("reflection_substrings.csv", index=False, header=False)

    # Note: I can't just return a csv of reflections_sanitized because I want gpt_reflections.csv
    # to be divided into sub-parts (that is, each of the sub-responses to each question in the original
    # reflection form). I want gpt_reflections this way because I plan to run experiments with the
    # questions to each sub-response included


# Test method to make sure that full_dataset.csv and label_sets.csv contain the same reflections and labels,
# and also to verify that the consensus labels are correct
def validate_datasets():
    full = []
    label_sets = []
    with open("full_dataset.csv", "r", encoding="utf-8") as full:
        c_r = csv.reader(full)
        full = list(c_r)
    with open("label_sets.csv", "r", encoding="utf-8") as l_sets:
        c_r = csv.reader(l_sets)
        label_sets = list(c_r)

    assert len(full)-1 == len(label_sets), "full_dataset.csv and label_sets.csv different lengths!"

    for i in range(1, len(full)):
        assert full[i][-1] == label_sets[i-1][0], f"Mismatched reflection found at index {i}!"

    # check if the consensus labels were calculated correctly
    for i in range(0, len(label_sets)):
        # recalculate consensus labels for each reflection -- reminder: take the avg_len most common
        # labels from each list of labels, where avg_len is the average length of the label set
        # (e.g. if annotator one put 3 labels and annotator two put 1 label, the avg_len would be
        # (3+1)/2 rounded which is 2

        label_strs = eval(label_sets[i][1])
        # only include a label if it's not in exclude_labels, and by extension in issue2integer.keys()
        for j in range(0, len(label_strs)):
            label_strs[j] = [label for label in label_strs[j] if label in list(issue2integer.keys())]

        avg_len = 0
        for l_set in label_strs:
            avg_len += len(l_set)
        avg_len = round(avg_len / len(label_strs))
        # beefy list comprehension below flattens the 2d list of label sets into one list of labels
        label_strs = [label for label_list in label_strs for label in label_list]
        counter = Counter(label_strs).most_common(avg_len)
        consensus_labels_encoded = [issue2integer[counter[i][0]] for i in range(0, len(counter))]
        for label_enc in consensus_labels_encoded:
            assert full[i+1][label_enc] == "1", f"Consensus label mismatch at reflection number {i}, reflection {full[i+1][-1]}!"

    with open("reflection_substrings.csv", "r", encoding="utf-8") as gr:
        c_r = list(csv.reader(gr))
        gpt_refs = [' '.join(row) for row in c_r]
        refs_actual = [row[0] for row in label_sets]
        assert len(gpt_refs) == len(refs_actual), f"Substring refs and actual refs have differing lengths, {len(gpt_refs)}, {len(refs_actual)}"
        for actual, gpt in zip(gpt_refs, refs_actual):
            assert gpt.rstrip() == actual.rstrip()


def main():
    # Manual pre-processing for the datasets: get rid of "ID" column in each excel in raw_data
    # and remove annotation sets that are not complete
    # also make sure that column g is the issue column in every dataset.
    # also make sure that none of the "issue" labels are null (either fix or remove them if so)
    # also make sure none of the cells in any of the header rows are null.
    # i could probably do these programmatically but the datasets are so messy it's
    # just less of a pain to do these first steps manually

    # Instructions: identify what labels should be kept in the final dataset by altering "exclude_labels"
    # and/or "include_other" (which includes the "Other" label class)
    # Ensure that this folder -> https://drive.google.com/drive/folders/10g5msqE4sELGakqICucO9sVuxlXoIggM?usp=drive_link
    # exists as "raw_data" in the same directory as main.py and organize.py
    # Also, create an empty directory called "data"
    # Refer to the top of this file for instructions on minor manual dataset cleaning to be done first
    # This code will output a multi-label dataset for each sub-dataset (each D-ESX-X dataset) as well as
    # "gpt_reflections.csv", which is used in my GPT-4o implementation as part of the prompt

    # Superset of labels as of 1/28:
    # [None, Python and Coding, GitHub, MySQL, Assignments, Quizzes, Learning New Material, Understanding requirements and instructions,
    # Course Structure and Materials, Time Management, Group Work, IDE and Package Installation, Personal Issue, API, HTML, SDLC, Other Primary]
    # All labels in the data but not in the superset are excluded
    #
    # Choose which labels to exclude in the final generated multilabel training dataset
    exclude_labels = ["Assignments", "Quizzes", "Learning New Material", "Personal Issue"]
    # TODO write in support for secondary labels
    label_category = "Primary"  # CAUTION: as of 1/28 I have not written in full support for the secondary label category -- COMING SOON

    if exclude_labels:
        # remove unwanted labels
        for label in exclude_labels:
            val = issue2integer.pop(label)
            # only pop every val once
            try:
                integer2issue.pop(val)
            except KeyError:
                pass
        # change integer values to be consecutive
        i = 0
        in_2_is_changes = []
        for key in issue2integer.keys():
            # special case: a lot of people put "FastAPI" instead of "API" as their label
            # for issues with fastapi so "FastAPI" and "API" both map to the same integer
            # but map back to just "API"
            if key == "FastAPI":
                issue2integer.update({key: i-1})
            else:
                issue2integer.update({key: i})
                in_2_is_changes.append({i: key})
                i += 1
        integer2issue.clear()
        for change in in_2_is_changes:
            integer2issue.update(change)

    next_available_num = max(list(issue2integer.values())) + 1
    if label_category == "Primary":
        issue2integer.update({"Other Primary": next_available_num})
        integer2issue.update({next_available_num: "Other Primary"})
    else:
        issue2integer.update({"Other Secondary": next_available_num})
        integer2issue.update({next_available_num: "Other Secondary"})

    if not os.path.isdir("data"):
        os.makedirs("data")
    if len(os.listdir("data")) == 0:
        organize.prepare_raw_data(label_category=label_category)  # wrangle the raw datasets into a format that can be further processed
        # into a multi-label training dataset

    full_dataset = []
    label_sets = []
    # for each sub directory in data (one for each annotation set), generate a multi-label training dataset
    reflections_sanitized = []  # every reflection used, given label exclusion constraint
    for sub_dir in os.listdir("data"):
        path = "data/" + sub_dir
        files = Path(path).glob("*")  # glob() mentioned
        output_file = "consensus-" + sub_dir + ".csv"

        # process() generates multi-label training dataset for each ESU/P dataset
        # and then returns list of reflections for that dataset
        package = construct_dataset(files=files, output_file=output_file, dataset_name=sub_dir)

        # package is the [0] the list of reflections to undergo further preprocessing for GPT code
        # and [1] the label sets and their corresponding reflections for calculating Krippendorff's alpha
        reflections_sanitized.extend([ref for ref in package[0]])

        for ref_label_pair in package[1]:
            if ref_label_pair[0] in reflections_sanitized:
                label_sets.append(ref_label_pair)

        # concatenate the datasets for each D-ESX-X dataset into one csv file full_dataset.csv
        with open(output_file, 'r', encoding="utf-8") as of:
            c_r = csv.reader(of)
            for row in list(c_r)[1:]:  # don't include the header
                full_dataset.append(row)

    # Consensus datasets (D-ESP-4-1, D-ESP4-2, ...) concatenated into one csv
    with open("full_dataset.csv", 'w', encoding="utf-8", newline="") as fd:
        c_w = csv.writer(fd)
        header = list(integer2issue.values())
        header.append("text")
        c_w.writerow(header)
        c_w.writerows(full_dataset)

    # Label sets for each dataset concatenated into one csv
    with open("label_sets.csv", "w", encoding="utf-8", newline="") as l_s:
        c_w = csv.writer(l_s)
        c_w.writerows(label_sets)

    # gpt_reflections is every reflection in raw_data divided in it's requisite sub-parts by default
    # based on reflections_sanitized which has been molded in accordance with the label exclusions,
    align_subreflections(reflections_sanitized)

    # test method that ensures that full_dataset.csv and label_sets.csv contain the same reflections
    # can throw AssertionError
    validate_datasets()


if __name__ == "__main__":
    main()
