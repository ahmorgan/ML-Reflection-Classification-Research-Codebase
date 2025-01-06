# DATASET PREPROCESSOR: goes from unprocessed annotation data to multi-label dataset
# for training a model for multi-label classification

# Instructions for raw_data excel file specifications (to be done manually):
# for all datasets:
#    get rid of the "ID" column in each excel in raw_data
#    and remove annotation sets that are not complete
# for D-ESA4-1:
#    the above steps plus, depending on whether you want
#    to generate a dataset with primary or secondary labels,
#    remove either the primary or secondary label column
# After that, additionally,
#    Make sure that column g / column 5 is the label column in every dataset.
#    Make sure that none of the issue label cells are null (either fix or remove them if so)
#    Make sure none of the cells in any of the header rows are null.
# the input dataset in raw_data should mirror the template:
#    columns 0 -> 4: reflection sub-parts column (i.e. column 0 is "How do you feel about this class?",
#    column 1 is "Explain why you feel this way", etc.)
#    column 5: issue label column

import csv
import pandas as pd
import numpy as np
from collections import Counter
import organize
from pathlib import Path
import os, os.path
import copy


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
def process(files, output_file, dataset_name):
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
    top_n_labels = {}

    # keeping track of all reflections w/o duplicates to add to
    # final processed dataset.
    # reflections instantiated every time a new
    # reflection is iterated over, except for the first one
    # which is added in a special case
    reflections = []

    refs_labelsets = {}

    # populate matrix
    for file in files:
        with open(file, "r", encoding="utf-8") as annotation:
            c_r = csv.reader(annotation)
            c_r = list(c_r)
            string_c_r = list(copy.deepcopy(c_r))
            # replace issue strings with mapped integers
            for i in range(0, len(c_r)):
                if c_r[i][1] in issue2integer.keys() and c_r[i][1] != "Other":
                    c_r[i][1] = issue2integer[c_r[i][1].strip()]
                else:
                    c_r[i][1] = max(list(issue2integer.values()))+1  # issues in exclude_labels will be mapped to next consecutive integer
                    # and be removed later
                    # (16 in case of no excluded labels)
            c_r.append(["", -1])  # stopping point, so the last reflection is included in loop
            current_reflection = c_r[0][0]
            reflection_labels = []  # n labels chosen by annotator for that reflection
            reflection_labels_str = []  # unencoded label strings with no "other" abstraction
            i = 0
            count = 0
            for row in c_r:
                # row[0] == reflection text, row[1] == label
                if current_reflection != row[0]:
                    reflections.append(current_reflection)  # current_reflection replaced, add to reflections
                    # create a new entry in the dictionary if one doesn't exist for the current reflection
                    if current_reflection not in data.keys():
                        data.update({current_reflection: reflection_labels})  # new reflection_labels array
                        top_n_labels.update({current_reflection: [i]})
                        refs_labelsets.update({current_reflection: [copy.deepcopy(reflection_labels_str)]})  # list of lists of label sets
                        # refs_labelsets needed to calculate inter-annotator disagreement
                    else:
                        data[current_reflection].extend(reflection_labels)
                        top_n_labels[current_reflection].append(i)
                        refs_labelsets[current_reflection].append(reflection_labels_str)
                    i = 0
                    reflection_labels = []
                    reflection_labels_str = []
                    current_reflection = row[0]
                reflection_labels.append(row[1])
                if count < len(string_c_r):
                    reflection_labels_str.append(string_c_r[count][1])
                i += 1
                count += 1

    # reflections contains an empty string (the stopping point), remove it
    reflections = reflections[:len(reflections)-1]

    # resolve label lists to consensus label list for each reflection
    for key in data.keys():
        # https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
        offset = 0
        while 1+max(list(issue2integer.values())) in data[key]:
            data[key].remove(1+max(list(issue2integer.values())))  # remove all integers mapping to an excluded label
            offset += 1  # since one label is removed from data, remove one label from top_n_labels
        # choose the top n labels from the list of labels, where n is the mean number of labels for that reflection
        top_n_labels[key] = round((sum(top_n_labels[key])-offset) / len(top_n_labels[key]))
        # only exception is to not allow for zero labels, which can happen when the other column is excluded
        top_n_labels[key] = top_n_labels[key] if top_n_labels[key] != 0 else 1
        # if there are still labels after excluding the other column, proceed to resolving label lists
        if data[key]:
            c = Counter(data[key]).most_common(top_n_labels[key])
            data[key].clear()
            for i in range(0, top_n_labels[key]):
                if i >= len(c):
                    break
                data[key].append(c[i][0])
            for j in range(0, len(data[key])):
                data[key][j] = integer2issue[data[key][j]]
        else:
            data[key] = [-1]

    # array of zeroes of size num_labels*num_reflections
    # (pycharm is requiring me to give the type hint that the dataset is an ndarray
    # to allow me to index into it, no idea why.)
    dataset = np.zeros((len(data), len(integer2issue)), dtype=np.int8)  # type: np.ndarray

    # populate final dataset with the consensus labels
    i = 0
    for label_set in data.values():
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
    package = [reflections_new, list(refs_labelsets.items())]

    return package


# Remove reflections from gpt_reflections.csv that do not correspond to the excluded labels
# parameter reflection is list of reflections with corresponding labels
# method returns None
# METHOD PARAMETERS:
# reflections: reflections_new from process()
def sanitize_gpt_reflections(reflections):
    # full_refs is a list of every reflection with no label exclusions
    # which will be populated by gpt_reflections.csv
    full_refs = []
    # copy full reflection data w/o sub-responses
    # with no label exclusion from gpt_reflections.csv into full_refs
    with open("gpt_reflections.csv", "r", encoding="utf-8") as gpt:
        c_r = csv.reader(gpt)
        for row in c_r:
            ref = ""
            # concatenate each part in full_refs
            for i in range(0, len(row)):
                if i == len(row)-1:
                    ref += row[i]
                else:
                    ref += row[i] + ' '
            full_refs.append(ref)
    del full_refs[0]
    # indices of rows to remove from dataframe of gpt_reflections
    # each row referenced in remove is one in full_refs
    remove = []
    # locate rows which are in full_refs but not in the label-truncated
    # reflections parameter, read their indices into remove
    for ref in full_refs:
        if ref not in reflections:
            remove.append(full_refs.index(ref))
    # dataframe of original gpt_reflections, each row contains the 5 sub-responses making
    # up the entire reflections
    df = pd.read_csv("gpt_reflections.csv")
    # drop undesired rows simultaneously and overwrite gpt_reflections.csv with new truncated data
    if remove:
        df.drop(remove, inplace=True)
    df.to_csv("gpt_reflections.csv", index=False, header=False)

    # Note: I can't just return a csv of reflections_sanitized because I want gpt_reflections.csv
    # to be divided into sub-parts (that is, each of the sub-responses to each question in the original
    # reflection form). I want gpt_reflections this way because I plan to run experiments with the
    # questions to each sub-response included


def main():
    # Instructions: identify what labels should be kept in the final dataset by altering "exclude_labels".
    # Alter label_category, depending on whether your raw data contains primary or secondary labels. This
    # parameter is necessary because there are two "Other" categories, one for primary labels and one for
    # secondary labels, and I need to know if the data comes from primary or secondary labels.
    # Ensure that the raw_data_primary_labels folder from my MLCompare research journal
    # exists in the same directory as main.py and organize.py and is named "raw_data".
    # I plan to make a similar raw_data_secondary_labels folder for the secondary labels soon, as well as add support
    # for generating datasets that use the secondary labels.
    # There are also instructions for the specifications that the files in raw_data should meet at the top of this file.
    # This code will output a multi-label dataset for each sub-dataset (each D-ESX-X dataset) as well as
    # "gpt_reflections.csv", which is used in my GPT-4o implementation as part of the prompt
    # You will also get full_dataset.csv, which is each sub-dataset concatenated, and label_sets.csv,
    # which is each reflection paired with the corresponding label sets assigned by each annotator as a 2d list of strings
    # Email me at amorga94@charlotte.edu with any questions or issues.
    
    # Support for excluding labels in the final generated multilabel training dataset
    exclude_labels = ["Assignments", "Quizzes", "Learning New Material", "Understanding requirements and instructions", "Personal Issue"]
    # TODO write in support for secondary labels
    label_category = "Primary"  # CAUTION: as of 1/4 I have not written in full support for the secondary label category -- COMING SOON

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
        organize.organize(label_category=label_category)  # wrangle the raw datasets into a format that can be further processed
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
        package = process(files=files, output_file=output_file, dataset_name=sub_dir)
        # package is the [0] the list of reflections to undergo further preprocessing for GPT code
        # and [1] the label sets and their corresponding reflections for calculating Krippendorff's alpha
        reflections_sanitized.extend([ref for ref in package[0]])
        for ref_label_pair in package[1]:
            if ref_label_pair[0] in reflections_sanitized:
                label_sets.append(ref_label_pair)  # label sets for calculating Krippendorff's alpha
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
    sanitize_gpt_reflections(reflections_sanitized)


if __name__ == "__main__":
    main()



