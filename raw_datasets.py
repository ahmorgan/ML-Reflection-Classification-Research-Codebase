from pathlib import Path
from pandas import DataFrame
import pandas as pd
import openpyxl


# Quick and dirty conversion needed to convert D-ESA4-1's differently formatted labels to all other reflection sets' formatting
label_name_conversion = {
    "python_and_coding": "Python and Coding",
    "SDLC": "SDLC",
    "github": "Github",
    "mysql": "MySQL",
    "api": "API",
    "html": "HTML",
    "ide_package_software_installation": "IDE and Package Installation",
    "ide_and_environment_setup": "IDE and Package Installation",
    "course_structure_and_materials": "Course Structure and Materials",
    "understanding_requirements_and_instructions": "Understanding requirements and instructions",
    "time_management_and_motivation": "Time Management and Motivation",
    "group_work": "Group Work",
    "none": "None"
}


def handle_esa41_formatting(sheet):
    """
    D-ESA4-1 formatting nonsense: changing to D-ESA4-1's formatting where multiple labels are assigned to a reflection
    on one row in the Google Sheet conflicts with all other datasets we have (and my code in dataset_construction.py),
    because I wrote that code specifically to handle the old annotation formatting.
    This function converts D-ESA4-1 to the needed formatting.
    :param sheet: D-ESA4-1
    :return: the reformatted dataset
    """
    annotation_set = []
    for row in sheet.values:
        if not row[0]:  # skip the junk rows which contain the empty dropdowns
            break
        if "Primary_Label(s)" in row[-1]:
            continue
        # the values in the cells which contain the labels are represented as strings, regardless
        # of whether the cell contains one or multiple labels. e.g. the cell with the labels GitHub, Python and Coding
        # just becomes the string "GitHub, Python and Coding", not a list of strings (which is stupid)
        # the only way to differentiate single label cells from multi label ones is to check for a comma in the string
        labels = []  # goal: extract the labels from the string pseudo-list
        # Leetcode sliding window problem in the wild
        i = 0
        j = 0
        for char in row[-1]:
            if char == ",":
                labels.append(row[-1][i:j].strip())
                i = j+1
            j += 1
        labels.append(row[-1][i:].strip())  # append final label that doesn't precede a comma
        # for each label, create a new row
        for label in labels:
            entry = list(row[:-1])  # the reflection text
            entry.append(label_name_conversion[label])
            annotation_set.append(entry)
    frame = DataFrame(annotation_set)
    return frame


def prepare_raw_data(label_category):
    """
    Convert raw data into an organized dictionary mapping reflection sets to a list of DataFrames, each of which contain
    each annotator's annotations.
    :param label_category: "Primary" or "Secondary", the label category to work under. Warning: "Secondary" is not yet
    programmed and should not be used.
    :return: the referenced dictionary mapping reflection sets to annotations, as well as the reflection text divided into the student's response to each question.
    """
    if label_category == "Primary":
        label_name_conversion.update({"other": "Other Primary"})
    else:
        label_name_conversion.update({"other": "Other Secondary"})

    print("Sanitizing raw data...\n")

    # glob all of the files in raw_data
    paths = Path("raw_data").glob("*")  # glob() mentioned
    # annotations is of shape {dataset_name: [DataFrame]} where the
    # list of dataframes is the set of annotations for that dataset
    annotations = {}

    # Read each sheet in each Excel file in raw_data and filter out unnecessary columns (we only want the challenge and the
    # student's responses to each question).
    for path in paths:
        wb = openpyxl.load_workbook(path)  # TODO - reading large Excel files is likely an execution time bottleneck; find new solution?
        for sheet_name in wb.sheetnames:
            wb.active = wb[sheet_name]  # move to next excel sheet
            df = None  # non-pythonic but who cares
            if sheet_name == "D-ESA4-1":  # special case: handle the new formatting in D-ESA4-1 (unfortunate but necessary)
                df = handle_esa41_formatting(wb.active)
            else:  # standard case: handle the old formatting in every other dataset
                df = DataFrame(wb.active.values)
                df.drop(5, axis="columns", inplace=True)  # drop "Emotion" column
                df.drop([i for i in range(7, len(df.columns)+1)], axis="columns", inplace=True)  # drop every column past "Issue"
                df.dropna(inplace=True)  # drop rows containing null values (all of the junk rows) after the end of the dataset

            if sheet_name not in annotations.keys():
                annotations.update({sheet_name: [df]})
            else:
                annotations[sheet_name].append(df)

    # * continue sanitizing the dataset
    # * also create a csv of reflection text split into sub-responses for gpt experiments
    # * later, gpt_reflections will be truncated to only include reflections corresponding to certain labels
    # if exclude_labels is not None
    ref_parts_dfs = []
    for dataset_name in sorted(list(annotations.keys())):
        sanitized_annotations = []

        # write columns of reflection text parts to separate list of dataframes
        ref_parts_dfs.append(DataFrame(annotations[dataset_name][0].iloc[:, 0:5]))

        for df in annotations[dataset_name]:
            # TODO ***this code is slow and could probably be improved***
            new_df = DataFrame()
            # concatenate the first five columns, which make up the reflection text
            new_df["text"] = df[[0, 1, 2, 3, 4]].agg(' '.join, axis=1)
            # issue column
            new_df["label"] = df.iloc[:, 5]
            new_df.drop(index=0, inplace=True)  # drop header row
            sanitized_annotations.append(new_df)
        annotations.update({dataset_name: sanitized_annotations})
    # concatenate all dataframes row-wise into one large dataframe which has all reflections split into
    # sub-responses. This is useful for my GPT experiments, where I want to include question headers
    # i.e. instead of just "Anxious", the reflection should read "How are you feeling about this class?: Anxious"
    ref_parts = pd.concat(ref_parts_dfs, ignore_index=True)

    return annotations, ref_parts
