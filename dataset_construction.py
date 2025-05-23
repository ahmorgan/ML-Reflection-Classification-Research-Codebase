# dataset construction: goes from unprocessed annotation data to multi-label dataset
# for training a model for multi-label classification
# written by Andrew Morgan
# a part of Dr. Mohsen Dorodchi's Text Analytics Lab at UNC Charlotte, for the student reflection classification project

import csv
import pandas as pd
import numpy as np
from collections import Counter
import raw_datasets
import glob
import copy


# Mapping label issues to integers so that the order of labels can be enforced internally (by encoding
# labels to integers and then sorting the encoded labels).
# This is also useful to combine multiple labels into one (e.g., FastAPI and API --> just API)
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

primary_labels = []  # list of labels considered "Primary", populated in construct_multilabel_datasets


def _construct_dataset(anns, dataset_name):
    """
    Calculates the consensus labels from each annotation for every reflection in that reflection set and organizes
    the data into the initial consensus multi-label dataset.
    :param anns: annotations to work from
    :param dataset_name: name of reflection set (e.g., D-ESA4-1) being constructed
    :return: multiple return: the raw reflections, formatted reflection annotation label sets, and the full
    multi-label dataset
    """
    print(f"Processing files in {dataset_name}...")
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
    for ann in anns:
        ann = ann.to_numpy().tolist()  # ann is a dataframe; it's easier here to work with lists
        # replace issue strings with mapped integers
        for i in range(0, len(ann)):
            if ann[i][1] in issue2integer.keys() and ann[i][1] != "Other":
                ann[i][1] = issue2integer[ann[i][1].strip()]
            else:
                ann[i][1] = max(list(issue2integer.values())) + 1  # issues in exclude_labels will be mapped to next consecutive integer
                # and be removed later
                # (16 in case of no excluded labels)
        ann.append(["", -1])  # dummy list entry, so the last reflection is included in the loop below
        current_reflection = ann[0][0]
        reflection_labels = []  # n labels chosen by annotator for that reflection
        i = 0
        for row in ann:
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

    # instantiate the labels with an array of zeros
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
    dataset = dataset[~np.all(dataset == 0, axis=1)]  # nice one liner to remove all lists of all zeros from dataset

    # TODO - below is inefficient (though works), refactor later
    df = pd.DataFrame(data=dataset)
    # append column for reflection associated with each set of labels
    df.insert(len(dataset[0]), "text", reflections_new)

    column_names = list(integer2issue.values())
    column_names.append("text")
    dataset_final = df.to_numpy().tolist()
    dataset_final.insert(0, column_names)

    # return new reflections for sanitize_gpt_reflections() and the the label sets
    # and their corresponding reflections
    package = [reflections_new, [list(item) for item in refs_labelsets.items()], dataset_final]

    return package


def _align_subreflections(reflections, sub_refs):
    """
    Alters sub_refs (unfiltered reflection sub-responses) to only have reflections present in param reflections.
    :param reflections: list of reflections that are included after excluding unwanted labels
    :param sub_refs: subreflections (reflection parts) to be filtered according to reflections.
    :return: DataFrame containing the filtered subreflections.
    """
    # full_refs is a list of every reflection with no label exclusions
    full_refs = []
    # copy full reflection data w/o sub-responses with no label exclusion
    for row in sub_refs:
        ref = " ".join(row)
        if len(full_refs) == 0 or " ".join(full_refs[-1]) != ref:
            full_refs.append(row)
    # indices of rows to remove from a dataframe of subreflections
    # each row referenced in remove is one in full_refs
    remove = []
    # locate rows which are in full_refs but not in the label-truncated
    # reflections parameter, read their indices into remove
    for i in range(0, len(full_refs)):
        if " ".join(full_refs[i]) not in reflections:
            remove.append(i)
    # dataframe of original gpt_reflections, each row contains the 5 sub-responses making
    # up the entire reflections
    # TODO - Again, it's inefficient to create a dataframe for just this one operation. refactor?
    df = pd.DataFrame(full_refs)
    # drop undesired rows simultaneously and overwrite gpt_reflections.csv with new truncated data
    if remove:
        df.drop(remove, inplace=True)

    return df


def _validate_datasets(full, label_sets, reflection_substrings):
    """
    Test method to make sure that full_dataset and label_sets contain the same reflections and labels,
    to verify that the consensus labels are correct, and to verify that the subreflections and full reflection sets
    have the same reflections (equivalently, that _align_subreflections() worked correctly).
    :param full: Full list of included reflections.
    :param label_sets: Full list of label sets (reflections paired with the set of annotations from each annotator).
    :param reflection_substrings: Final list of reflections split into subresponses
    :return: None
    """
    assert len(full)-1 == len(label_sets), "full_dataset.csv and label_sets.csv are different lengths!"

    for i in range(1, len(full)):
        # full contains the header row ("text", "label"); corresponding reflections in label sets are one index behind
        assert full[i][-1] == label_sets[i-1][0], f"Mismatched reflection found at index {i}!"

    # check if the consensus labels were calculated correctly
    for i in range(0, len(label_sets)):
        # recalculate consensus labels for each reflection -- methodology reminder: take the avg_len most common
        # labels from each list of labels, where avg_len is the average length of the label set
        # (e.g. if annotator one put 3 labels and annotator two put 1 label, the avg_len would be
        # (3+1)/2 rounded which is 2

        label_strs = label_sets[i][1]
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
            assert full[i+1][label_enc] == 1, f"Consensus label mismatch at reflection number {i}, reflection {full[i+1][-1]}!"

    sub_refs = [' '.join(row) for row in reflection_substrings]
    refs_actual = [row[0] for row in label_sets]
    assert len(sub_refs) == len(refs_actual), f"Substring refs and actual refs have differing lengths, {len(sub_refs)}, {len(refs_actual)}"
    for actual, sub in zip(sub_refs, refs_actual):
        assert sub.rstrip() == actual.rstrip()


def _update_label_integer_mappings(exclude_labels):
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
            issue2integer.update({key: i - 1})
        else:
            issue2integer.update({key: i})
            in_2_is_changes.append({i: key})
            i += 1
    integer2issue.clear()
    for change in in_2_is_changes:
        integer2issue.update(change)


def construct_multilabel_datasets(exclude_labels, label_category):
    """
    Construct a multi-label consensus dataset based on specifications defined in params exclude_labels and label_category,
    consensus meaning that the labels in the dataset are the top n labels specified by the annotators, where n is the
    average length of the annotators' label sets.

    Also returns each individual reflection set (e.g. D-ESA4-1) as its own multi-label consensus dataset
    (individual_reflection_set_datasets), a dataset of each reflection paired with all annotation label sets created by
    each annotator (annotation_label_sets), and a list of all reflections in the full dataset split into subresponses
    (reflection_subresponses). These have varied uses in our reflection classification pipeline:

    ***individual_reflection_set_datasets: useful for splitting the dataset by reflection type (e.g., reflection 1)
    ***annotation_label_sets: necessary for calculating reflection agreement and Krippendorff's alpha in
    dataset_filtering.py
    ***reflection_subresponses: useful for various dataset manipulations, notably splitting the dataset into
    challenge-only reflections and for LLM-prompt-based experiments

    IMPORTANT: Finally, this function requires a folder raw_data with the unprocessed sets of annotations. Currently, raw_datasets.py
    is hardcoded for our specific use case, so make sure to use the raw datasets from this Google Folder:
    https://drive.google.com/drive/folders/10g5msqE4sELGakqICucO9sVuxlXoIggM?usp=drive_link. We will want to determine a
    standardization for raw data input format if and when this code is deployed for further use.

    :param exclude_labels: the labels to exclude when constructing all output datasets
    :param label_category: "Primary" or "Secondary", the issue type to work under. Warning: "Secondary" has not been explicitly programmed yet and is likely unstable.
    :return: multiple return: (multilabel_consensus_dataset, individual_reflection_set_datasets, annotation_label_sets, reflection_subresponses).
    """
    # TODO - Improve or somehow generalize the below described hardcoding; fine for our purpose, but not sustainable
    # Manual pre-processing for the raw datasets: get rid of "ID" column in each excel in raw_data
    # and remove annotation sets that are not complete
    # also make sure that column g is the issue column in every dataset.
    # also make sure that none of the "issue" labels are null (either fix or remove them if so)
    # also make sure none of the cells in any of the header rows are null.
    # i could probably do these programmatically but the datasets are so messy it's
    # just less of a pain to do these first steps manually
    # These steps have already been completed in the raw datasets in https://drive.google.com/drive/folders/10g5msqE4sELGakqICucO9sVuxlXoIggM?usp=drive_link.

    # Instructions: identify what labels should be kept in the final dataset by altering "exclude_labels"
    # and label_category
    # Ensure that this folder -> https://drive.google.com/drive/folders/10g5msqE4sELGakqICucO9sVuxlXoIggM?usp=drive_link
    # exists as "raw_data" in the same directory as dataset_construction.py and raw_datasets.py

    # (primary) superset of labels as of 1/28:
    # [None, Python and Coding, GitHub, MySQL, Assignments, Quizzes, Learning New Material, Understanding requirements and instructions,
    # Course Structure and Materials, Time Management, Group Work, IDE and Package Installation, Personal Issue, API, HTML, SDLC, Other Primary]
    # All labels in the data but not in the superset are excluded

    if exclude_labels:
        # remove unwanted labels
        _update_label_integer_mappings(exclude_labels=exclude_labels)

    # also update other category based on the label category
    next_available_num = max(list(issue2integer.values())) + 1
    if label_category == "Primary":
        issue2integer.update({"Other Primary": next_available_num})
        integer2issue.update({next_available_num: "Other Primary"})
    else:
        issue2integer.update({"Other Secondary": next_available_num})
        integer2issue.update({next_available_num: "Other Secondary"})

    annotations, sub_refs = raw_datasets.prepare_raw_data(label_category=label_category)  # wrangle the raw datasets into a format that can be further processed
    sub_refs = sub_refs.to_numpy().tolist()
    # into a multi-label training dataset

    full_dataset = []  # all consensus datasets (consensus-D-ESA4-1, consensus-D-ESP4-1, etc.) concatenated
    label_sets = []  # label sets: reflections paired with annotations sets, necessary for calculating Krippendorff's alpha later
    # for each sub directory in data (one for each annotation set), generate a multi-label training dataset
    reflections_sanitized = []  # every reflection used, given label exclusion constraint
    individual_reflection_sets = {}
    for dataset in sorted(list(annotations.keys())):
        anns = annotations[dataset]
        output_file = "consensus-" + dataset + ".csv"

        # process() generates multi-label training dataset for each ESU/P dataset
        # and then returns list of reflections for that dataset
        package = _construct_dataset(anns=anns, dataset_name=dataset)

        refs = package[0]
        l_sets = package[1]
        d_set = package[2]


        reflections_sanitized.extend([ref for ref in refs])

        for ref_label_pair in l_sets:
            if ref_label_pair[0] in reflections_sanitized:
                label_sets.append(ref_label_pair)

        for row in d_set[1:]:  # don't include the header
            full_dataset.append(row)

        individual_reflection_sets.update({output_file: d_set})

    header = list(integer2issue.values())
    header.append("text")
    full_dataset.insert(0, header)

    # gpt_reflections is every reflection in raw_data divided in it's requisite sub-parts by default
    # based on reflections_sanitized which has been molded in accordance with the label exclusions,
    reflection_substrings = _align_subreflections(reflections_sanitized, sub_refs=sub_refs).to_numpy().tolist()

    # test method that ensures that full_dataset.csv and label_sets.csv contain the same reflections
    # can throw AssertionError
    _validate_datasets(full_dataset, label_sets, reflection_substrings)

    """
    manual_check = {"match\\label_sets.csv": label_sets,
                    "match\\full_dataset.csv": full_dataset,
                    "match\\reflection_substrings.csv": reflection_substrings,
                    "match\\gpt_reflections.csv": sub_refs}

    for name, item in individual_reflection_sets.items():
        name = "match\\" + name
        manual_check.update({name: item})

    test_match(manual_check=manual_check)
    """

    for name, item in copy.deepcopy(list(individual_reflection_sets.items())):
        del individual_reflection_sets[name]
        name = name.replace(".csv", "")
        individual_reflection_sets.update({name: item})

    return full_dataset, individual_reflection_sets, label_sets, reflection_substrings


"""
# refactoring test case to check that any change does not alter the output of the code
def test_match(manual_check):
    paths = glob.glob("match/*.csv")
    for manual in manual_check.keys():
        paths.remove(manual)
    for path in paths:
        orig_path_idx = path.index("\\")
        orig_path = path[orig_path_idx + 1:]  # the original path will be in the working directory
        with open(path, "r", encoding="utf-8") as p:
            check = list(csv.reader(p))
        with open(orig_path, "r", encoding="utf-8") as o:
            orig = list(csv.reader(o))
        assert check == orig, f"{path} and {orig_path} do not match!"

    for path, item in manual_check.items():
        with open(path, "r", encoding="utf-8") as p:
            check = list(csv.reader(p))
        path_idx = path.index("\\")
        path = path[path_idx+1:]
        with open(f"check_{path}", "w", encoding="utf-8", newline="") as c:
            c_w = csv.writer(c)
            c_w.writerows(item)
        with open(f"check_{path}", "r", encoding="utf-8") as c:
            item = list(csv.reader(c))
        assert check == item
"""


def construct_reflection_sets(individual_reflection_datasets, reflection_sets):
    """
    Extract and construct only the reflection sets specified in param reflection_sets from all the sub-datasets in
    individual_reflection_datasets.

    :param individual_reflection_datasets: dictionary of sub-datasets (dont really have a good name for these; e.g., D-ESA4-1). Returned by construct_multilabel_datasets(). All dictionary keys must have the reflection set number as the last character.
    :param reflection_sets: List of strings of what reflection sets to construct (e.g., ["r1", "r2" ...]). Must be formatted that way ('r' + ref_number).
    :return: A list of the constructed datasets.
    """

    refs = {ref_name: [] for ref_name in reflection_sets}
    for ref_set in individual_reflection_datasets.keys():
        if ref_set[-1] not in [ref_name[-1] for ref_name in reflection_sets]:
            continue
        if not refs['r' + ref_set[-1]]:
            refs['r' + ref_set[-1]].extend(individual_reflection_datasets[ref_set])  # individual_reflection_datasets maps reflection names to the consensus dataset for that reflection
        else:  # avoid writing the header twice
            refs['r' + ref_set[-1]].extend(individual_reflection_datasets[ref_set][1:])

    return refs

