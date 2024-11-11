from pathlib import Path
from pandas import DataFrame
import pandas as pd
import openpyxl
import os

# this script converts raw, (minimally) unprocessed datasets from excel files
# to intermediate .csvs that will be used in main.py to create multi-label training datasets
# intermediate .csvs are of the form (reflection,[labels]) for each row
# reads data from /raw_data, does the necessary conversion, and writes the files to /data


def organize():
    # glob all of the files in raw_data
    paths = Path("raw_data").glob("*")  # glob() mentioned
    # annotations is of shape {dataset_name: [DataFrame]} where the
    # list of dataframes is the set of annotations for that dataset
    annotations = {}
    for path in paths:
        wb = openpyxl.load_workbook(path)
        for sheet_name in wb.sheetnames:
            wb.active = wb[sheet_name]  # move to next excel sheet
            df = DataFrame(wb.active.values)
            df.drop(5, axis="columns", inplace=True)  # drop "Emotion" column
            df.drop([i for i in range(7, len(df.columns)+1)], axis="columns", inplace=True)  # drop every column past "Issue"
            df.dropna(inplace=True)  # drop rows containing null values (all of the junk columns
            # after the end of the dataset that are still read to the dataframe for some reason)
            if sheet_name not in annotations.keys():
                annotations.update({sheet_name: [df]})
            else:
                annotations[sheet_name].append(df)

    # * continue sanitizing the dataset
    # * also create a csv of reflection text split into sub-responses for gpt experiments
    # * later, gpt_reflections will be truncated to only include reflections corresponding to certain labels
    # if exclude_labels is not None
    ref_parts_dfs = []
    for ann in sorted(list(annotations.keys())):
        new_df_list = []
        # write columns of reflection parts to separate list of dataframes
        ref_parts_dfs.append(DataFrame(annotations[ann][0].iloc[:, 0:5]))
        for df in annotations[ann]:
            new_df = DataFrame()
            # concatenate the first five columns, which make up the reflection text
            new_df["text"] = df[[0, 1, 2, 3, 4]].agg(' '.join, axis=1)
            # issue column
            new_df["label"] = df.iloc[:, 5]
            new_df_list.append(new_df)
        annotations.update({ann: new_df_list})
    # concatenate all dataframes row-wise into one large dataframe which has all reflections split into
    # sub-responses. This is useful for my GPT experiments, where I want to include question headers
    # i.e. instead of just "Anxious", the reflection should read "How are you feeling about this class?: Anxious"
    ref_parts = pd.concat(ref_parts_dfs, ignore_index=True)
    ref_parts.drop_duplicates(inplace=True)
    # gpt_reflections is a csv file and not a class member because I'm going to create the file
    # anyway later, so I might as well do it now and then overwrite it if and when it's necessary
    ref_parts.to_csv("gpt_reflections.csv", header=False, index=False)

    # write processed annotations to csvs organized in file structure:
    # data --> [annotation sets] --> [annotations]
    for ann_set in annotations.keys():
        try:
            directory = "data/" + ann_set
            os.mkdir(directory)
            i = 1
            for ann in annotations[ann_set]:
                path = directory + "/annotation" + str(i)
                ann.drop(index=0, inplace=True)  # remove first row which is just the header
                # also remove pandas' automatically generated headers and index
                ann.to_csv(path, header=False, index=False)
                i += 1
        except Exception as e:
            print(f"An error occurred: {e}")

    print(annotations)
