import pandas as pd
import glob
import csv
import os
import numpy as np
from collections import Counter
from nltk import AnnotationTask
from nltk.metrics import binary_distance
import copy
import random

ref_to_question_included = {}  # maps every reflection in our superset to question included version as a list
questions = []


### Relevant Functions ###
# The functions below have been streamlined for upgrading old reflection datasets to the latest version.


def single_label_consensus(input_folder, output_file, only_100=False, exclude_controversial=False, update_label_sets=False):
    """
    Process new annotations into one dataset of consensus labels.

    :param input_folder: str: directory from which to take annotations
    :param output_file: str: the file to output the resultant dataset into
    :param only_100: bool: if True, only consensus labels with 100% agreement are included
    :param exclude_controversial: bool: if True, reflections where the consensus label is
    "Controversial" are excluded from the resultant dataset
    :param update_label_sets: bool: if True, the label sets will be updated with the new consensus labels
    :return None:
    """
    paths = glob.glob(input_folder + "/*.csv")
    consensus = pd.DataFrame()
    consensus["text"] = pd.read_csv(paths[0])["text"]
    for path in paths:
        consensus[path] = pd.read_csv(path)["label"]

    print(f"Dataset length before: {len(consensus.to_numpy())}")

    # Get the consensus label based on each label annotation
    def con(row):
        labels = row.iloc[1:].tolist()
        if only_100 and len(Counter(labels).most_common()) > 1:  # Disagreement in the relabelings, exclude in 100 agreement
            return "remove"
        consensus_label = Counter(labels).most_common(1)[0][0]
        if consensus_label == "None (No Issue)":
            consensus_label = "None"
        return consensus_label

    consensus["label"] = consensus.apply(con, axis=1)
    consensus = consensus[consensus["label"] != 'remove']
    consensus_temp = consensus[["text", "label"]]

    if exclude_controversial:
        consensus_temp = consensus_temp[consensus_temp["label"] != "Controvercial"]

    consensus_temp.to_csv(output_file, index=False)

    """
    # save the reflections where the consensus is that the reflection is a secondary issue
    consensus_contr = consensus_temp[consensus_temp["label"] == target]
    consensus_contr = consensus_contr["text"]
    consensus_contr.to_csv("controversial_annotations/consensus_other_refs.csv", index=False)

    # save the actual none reflections separately
    consensus_temp = consensus_temp[consensus_temp["label"] != target]
    consensus_temp.to_csv("controversial_annotations/consensus_labels_none.csv", index=False)
    """

    if update_label_sets:
        # save the label sets
        consensus = consensus.drop("label", axis=1)
        consensus["label"] = consensus.apply(lambda row: [[label] for label in row.iloc[1:].tolist()], axis=1)
        # consensus["con_label"] = consensus_temp["label"]

        # include only the entries where the consensus label is in consensus_temp
        # consensus = consensus[~consensus["con_label"].isna()]
        consensus = consensus[["text", "label"]]
        consensus.to_csv("label_sets_none.csv", index=False, header=False)

    print(f"Dataset length after: {len(consensus.to_numpy())}\n")

    if not exclude_controversial:
        assert len(consensus) == len(consensus_temp), f"{len(consensus)}, {len(consensus_temp)}"
    assert [row[0] for row in consensus.to_numpy()] == [row[0] for row in consensus_temp.to_numpy()]


def add_new_refs_to_reflection(new_reflection, reflection: int, ref_agreement: str, output_folder: str, use_original_data=False, update_label_sets=False):
    """
    Integrates new reflections into a dataset by either appending them if they are new reflections (reflections outside
    of that reflection number, e.g. new reflections not in reflection 2) or replacing the reflection in-place if it
    already exists in the reflection

    :param new_reflection: list[tuple[str]]: the reflection to be integrated
    :param reflection: index of the reflection to be integrated into (e.g. 2 for reflection 2)
    :param ref_agreement: agreement level of the reflection to be integrated into (e.g. 100 for 100% agreement)
    :param output_folder: the folder to output the updated dataset to
    :param use_original_data: whether to use the original reflection data as the target reflection
    :param update_label_sets: whether to update the label sets with the new reflections
    :return None:
    """

    # the reflections to be integrated into the dataset
    with open(new_reflection, "r", encoding="utf-8") as nr:
        new_refs = list(csv.reader(nr))[1:]

    # superset dataset containing all reflections for that bin of reflections
    # (all-reflections, reflection 1, reflection 2, etc.)
    if use_original_data:
        path = "ref_1-4_singlelabel_unsanitized/ref_" + str(reflection) + "_unsanitized" + ".csv"
    else:
        path = "updated_datasets/ref_" + str(reflection) + "_unsanitized" + ".csv"
    with open(path, "r", encoding="utf-8") as ref_full:
        reflection_unsanitized_text = [row[0] for row in list(csv.reader(ref_full))[1:]]

    # dataset to be altered
    if use_original_data:
        path = "ref_1-4_singlelabel_" + ref_agreement + "/ref_" + str(reflection) + "_" + ref_agreement + ".csv"
    else:
        path = "updated_datasets/ref_" + str(reflection) + "_" + ref_agreement + ".csv"
    with open(path, "r", encoding="utf-8") as ref_target:
        reflection = list(csv.reader(ref_target))
        if reflection[0][0] == "text" and reflection[0][1] == "label":
            reflection = reflection[1:]
        target_reflection_text = [row[0] for row in reflection]

    print(f"Dataset length before: {len(reflection)}")

    if update_label_sets:
        # all label sets from all reflections with no filtering applied
        # used for running agreement on datasets and should be updated alongside
        # the target dataset
        with open("full_label_sets.csv", "r", encoding="utf-8") as fls:
            full_label_sets = list(csv.reader(fls))
            full_ref_text = [row[0] for row in full_label_sets]

        # label sets for new_refs
        with open("label_sets.csv", "r", encoding="utf-8") as ls:
            new_label_sets = list(csv.reader(ls))
            new_label_sets_text = [row[0] for row in new_label_sets]

    refs_appended_to_new_data = []
    refs_added_count = 0

    for row in new_refs:
        # append reflections from new_refs that are in the list of reflections for the unsanitized reflection
        if row[0] in reflection_unsanitized_text:
            if row[1] != "Controvercial":
                # case 1: reflection is in the superset of reflections for that dataset but not in the
                # dataset we are altering. Then, append the new reflection to the end of that dataset
                if row[0] not in target_reflection_text:
                    refs_added_count += 1
                    reflection.append(row)
                    refs_appended_to_new_data.append(row)
                # case 2: reflection is in the superset of reflections for that dataset and also is in the
                # dataset we are altering. Then, alter that reflection in place with the updated labels.
                else:
                    # In the case of duplicate reflections, just update all of those reflections
                    # with the same value all at once
                    if target_reflection_text.count(row[0]) > 1:
                        ref_indices = [index for index, value in enumerate(target_reflection_text) if value == row[0]]
                        for idx in ref_indices:
                            reflection[idx] = row
                    else:
                        ref_idx = target_reflection_text.index(row[0])
                        reflection[ref_idx] = row
            else:
                # Reflections in the target reflection that have been relabeled as "controversial"
                # (spelled wrong intentionally because it was spelled wrong in the dataset) should be
                # removed from the dataset if they are currently in it
                if row[0] in target_reflection_text:
                    ref_idx = target_reflection_text.index(row[0])
                    reflection[ref_idx] = None  # deleting the reflection now alters the reflection indices, which
                    # makes thing annoying, so instead, mark them as None and delete them later.

            # update the superset of label sets as we update the dataset
            if update_label_sets:
                idx = full_ref_text.index(row[0])
                new_label_set = new_label_sets[new_label_sets_text.index(row[0])]
                full_label_sets[idx] = new_label_set

    prev_len = len(reflection)
    reflection = [row for row in reflection if row]  # removing None reflections, which were marked for removal because
    # they were relabeled as "controversial"
    removed_len = prev_len - len(reflection)
    print(f"Removed {removed_len} controversial reflections.")
    print(f"Added {refs_added_count} relabeled reflections.")
    print(refs_appended_to_new_data)

    path_idx = path.index("/")

    test_new_dataset(reflection, new_refs, refs_appended_to_new_data, reflection_unsanitized_text)

    print(f"Dataset length after: {len(reflection)}\n")

    with open(f"{output_folder}/{path[path_idx+1:]}", "w", encoding="utf-8", newline="") as ur:
        c_w = csv.writer(ur)
        c_w.writerows(reflection)

    if update_label_sets:
        with open("updated_label_sets.csv", "w", encoding="utf-8", newline="") as ls:
            c_w = csv.writer(ls)
            c_w.writerows(full_label_sets)


def test_new_dataset(new_data, added_data, refs_appended_to_new_data, refs_unsanitized):
    contr_refs = [row[0] for row in added_data if row[1] == "Controvercial"]
    new_refs = [row[0] for row in new_data]

    # check no refs with the "Controvercial" label are in the new data
    for ref in contr_refs:
        assert ref not in new_refs

    # get intersection between refs_unsanitized and added_data and check that all of those reflections are correct
    should_be_added = [row for row in refs_unsanitized if row[0] in [row[0] for row in added_data]]

    # check that every reflection that should have been appended to new_data was, and that
    # the label is correct for that reflection
    for row in should_be_added:
        assert row[0] in new_refs
        assert added_data[added_data.index(row[0])][1] == new_refs[new_refs.index(row[0])][1]

    for row in added_data:
        # for reflections that were appended to new_data, check that they are actually in whatever
        # reflection number they should be in, and that their label matches that in added_data
        if row[0] in refs_appended_to_new_data:
            assert row[0] in refs_unsanitized
            assert row[1] == refs_appended_to_new_data[refs_appended_to_new_data.index(row[0])][1]

        # check if all labels from reflections in added_data that are in new_refs have been changed in new_refs
        if row[0] in new_refs and row[1] != "Controvercial":
            assert new_data[new_refs.index(row[0])][1] == row[1]


def construct_question_included_map():
    ref_to_question_included.clear()

    # split_refs is all of our reflections in our dataset split into sub-responses
    # based on the question the student is responding to
    with open("split_refs.csv", "r", encoding="utf-8") as sr:
        refs = list(csv.reader(sr))
        header_questions = refs[0]
        refs = refs[1:]
        refs_text = [" ".join(row) for row in refs]

    for text, ref in zip(refs_text, refs):
        ref_to_question_included.update({text: ref})

    for question in header_questions:
        questions.append(question)


def construct_split_reflection_dataset(input_folder, output_folder):
    construct_question_included_map()  # dictionary mapping reflection text to split reflection text
    paths = glob.glob(f"{input_folder}/*.csv")

    for path in paths:
        print(path)
        with open(path, "r", encoding="utf-8") as r:
            dataset = list(csv.reader(r))

            path_idx = path.index("\\")
            print(f"Dataset length of {path[path_idx+1:]} before: {len(dataset)}")
            dataset_text = [row[0] for row in dataset]
            prev_len = len(dataset)
            for row, i in zip(dataset, range(0, len(dataset))):
                # stupid edge case that happens exactly once where it's easier ot just hardcode it
                if row[0] == "😍Excited <name omitted> is the best <name omitted> is the best <name omitted> is the best <name omitted> is the best":
                    q_inc = copy.deepcopy([row[0], "", "", "", ""])
                else:
                    q_inc = copy.deepcopy(ref_to_question_included[row[0]])
                q_inc.append(row[1])  # q inc is the split refs as individual entries in a list plus the label
                dataset[i] = copy.deepcopy(q_inc)

            print(f"Dataset length of {path[path_idx + 1:]} after: {len(dataset)}\n")

            target = f"{output_folder}/{path[path_idx+1:]}"
            with open(target, "w", encoding="utf-8", newline="") as sd:
                c_w = csv.writer(sd)
                header = copy.deepcopy(questions)
                header.append("label")
                c_w.writerow(header)
                c_w.writerows(dataset)

            assert prev_len == len(dataset)
            for ref, ref_2 in zip(dataset_text, dataset):
                if ref_2[0] == "😍Excited <name omitted> is the best <name omitted> is the best <name omitted> is the best <name omitted> is the best":
                    continue
                assert ref == " ".join(ref_2[:5]), f"{ref}, \n {ref_2[:5]}"


def construct_question_prepended_refs(input_folder, output_folder):
    paths = glob.glob(f"{input_folder}/*.csv")

    for path in paths:
        print(path)
        df_labels = pd.read_csv(path)
        path_idx = path.index("\\")
        print(f"Dataset length of {path[path_idx+1:]} before: {len(df_labels)}")
        # pandas reads the string "None" as NaN, change back to None
        df_labels["label"] = df_labels["label"].fillna(value="None")
        check_len = len(df_labels)
        df_refs = df_labels.drop(columns=["label"])
        qstns = df_refs.columns
        # Similarly, pandas reads the common student sub-response "N/A"
        # or "NA" as NaN, change back to the string N/A
        for q in qstns:
            df_refs[q] = df_refs[q].fillna(value="N/A")

        def prepend_question(row):
            combined = []
            for ref, q in zip(row, qstns):
                combined.append(q + " " + ref)
            return "\n".join(combined)

        df_refs["text"] = df_refs.apply(prepend_question, axis=1)
        df_refs["label"] = df_labels["label"].values
        df_refs = df_refs[["text", "label"]]

        # assert check_len == len(df_refs), f"{check_len}, {len(df_refs)}"

        print(f"Dataset length of {path[path_idx + 1:]} after: {len(df_refs)}\n")
        df_refs.to_csv(f"{output_folder}/" + path[path_idx+1:], index=False)


# for creating the proficiency label dataset
def collapse_into_other(global_exclude, ref_num, ref_exclude, input_folder, output_folder):
    exclude = copy.deepcopy(global_exclude)
    exclude.extend(ref_exclude)
    print(f"Putting labels into 'Other' label class: {exclude}")
    paths = glob.glob(f"{input_folder}/*.csv")  # regex matches all path strings for that ref number
    paths = [path for path in paths if "ref_"+str(ref_num) in str(path)]
    other_refs = pd.DataFrame()
    for path in paths:
        print(path)
        df = pd.read_csv(path)

        path_idx = path.index("\\")
        print(f"Dataset length of {path[path_idx + 1:]} before: {len(df)}")
        # "None" will have been converted into NaN, convert it back
        df["label"] = df["label"].fillna(value="None")

        other_refs = df[df["label"].isin(exclude)]

        # For each label, if it's one of the labels that should be in "Other", make it so
        df["label"] = df["label"].apply(lambda label: "Other" if label in exclude else label)

        def check(label):
            assert label and label not in exclude, "Massive failure"
            return label

        # Test case: check if the intended behavior was performed
        df["label"].apply(check)

        def check_other_refs(label):
            assert label in exclude

        other_refs["label"].apply(check_other_refs)

        print(f"Dataset length of {path[path_idx+1:]} after: {len(df)}\n")
        print(f"{len(other_refs)} Other reflections present.")

        df.to_csv(f"{output_folder}/{path[path_idx+1:]}", index=False)

    return other_refs


def update_label_sets():
    with open("label_sets.csv", "r", encoding="utf-8") as ls:
        label_sets = list(csv.reader(ls))
    with open("label_sets_none.csv", "r", encoding="utf-8") as lsn:
        none_refs = list(csv.reader(lsn))
    none_refs_text = [row[0] for row in none_refs]
    label_sets_text = [row[0] for row in label_sets]

    def update(row):
        ref_text = row[0]
        labels = []
        none_ref_idx = none_refs_text.index(ref_text)
        labels = eval(none_refs[none_ref_idx][1])
        labels = [label[0] for label in labels]
        labels = "[" + ", ".join(labels) + "]"

        return labels

    for ref in label_sets_text:
        if ref in none_refs_text:
            idx = label_sets_text.index(ref)
            label_sets[idx][1] = update(label_sets[idx])

    with open("updated_label_sets.csv", "w", encoding="utf-8", newline="") as uls:
        c_w = csv.writer(uls)
        c_w.writerows(label_sets)


def supplement_label_classes(supplement_labels, increase_to, other_refs, other_labels, input_folder):
    """
    Supplement label classes with reflections of that label class from other reflection sets. By default,
    every reflection is supplemented. Reflection csv files are altered in place.

    :param supplement_labels: the label classes to supplement
    :param increase_to: the number of occurrences to increase each label class to (will either increase to that
    number or add as many as possible if that number cannot be reached). Any label class that already occurs more
    than increase_to times is excluded from supplementation
    :param other_refs: reflection text for all reflections with the Other label
    :param other_labels: all global underlying labels in the Other label class
    :param input_folder: the folder to source the reflection sets from
    :return: None
    """
    paths = glob.glob(f"{input_folder}/*.csv")

    for path in paths:
        print(path)
        start_ref = pd.read_csv(path, header=None)
        start_ref = start_ref.fillna(value="None")
        start_ref = start_ref.to_numpy().tolist()[1:]
        with open(path, "a", encoding="utf-8", newline="") as p:
            target_reflection = csv.writer(p)

            # only include label classes that are in that reflection set and occur less often than increase_to
            ref_labels = [row[1] for row in start_ref]
            ref_labels = [label for label in ref_labels if ref_labels.count(label) < increase_to]
            supplement = [label for label in supplement_labels if label in set(ref_labels)]

            # get all of the reflection sets that are not the current reflection
            ref_file_name_idx = path.index("\\")+1
            current_ref = path[ref_file_name_idx:ref_file_name_idx+5]
            supplement_sources = ["final_datasets/" + path for path in os.listdir("final_datasets") if current_ref not in path]

            # only supplement 100% agreement reflection sets with reflections from other 100% agreement sets
            if "100" in path:
                supplement_sources = [source for source in supplement_sources if "100" in source]

            other_labels_edit = copy.deepcopy(other_labels)
            if current_ref == "ref_1":
                other_labels_edit.extend(["MySQL", "Github"])
            if current_ref == "ref_2":
                other_labels_edit.append("MySQL")
            print(other_labels_edit)

            # In the case that the underlying label of "Other" is in the included labels for a dataset,
            # get those reflections and add them to the bank of reflections to choose from
            include_other_refs = [ref for ref in other_refs.to_numpy().tolist() if ref[1] not in other_labels_edit]

            # Only the text of "Other" reflections with underlying labels that match the underlying labels
            # for "Other" for that reflection set. e.g., for reflection 1, mysql and github as well as the global
            # other labels make up the set of underlying labels for the Other label class, so we only want Other
            # reflections that have labels from that underlying label set
            other_refs_text = [ref[0] for ref in other_refs.to_numpy().tolist() if ref[1] in other_labels_edit]

            print(f"Sourcing from {supplement_sources}")

            total_added_refs = []
            exception_other_refs_added = 0

            for label in supplement:
                supplement_refs = []
                for supplement_path in supplement_sources:
                    with open(supplement_path, "r", encoding="utf-8") as sp:
                        reflection = list(csv.reader(sp))[1:]
                        reflection_text = [ref[0] for ref in reflection]

                        # avoid duplicate reflections
                        if label == "Other":
                            # Since reflections in supplement_path already have reflections converted to "Other",
                            # we can't directly know the underlying label for "Other" reflections. Instead, I've
                            # saved the text of the "Other" reflections and am only sourcing reflections from there.
                            # I can't source directly from other_refs_edit because other_refs_edit contains all
                            # Other reflections, including ones from unsanitized datasets, so we first only look
                            # in reflection and then check if that reflection is in other_refs_edit.
                            supplement_refs.extend([ref for ref in reflection if (ref[0] in other_refs_text or ref[1] in other_labels_edit)
                                                    and ref not in supplement_refs and ref not in start_ref])
                        else:
                            supplement_refs.extend([ref for ref in reflection if ref[1] == label and ref not in supplement_refs and ref not in start_ref])

                        before = len(supplement_refs)
                        # To avoid adding exception Other reflections that are not 100% agreement, the candidate reflection must also be in one of the source reflections
                        supplement_refs.extend([ref for ref in include_other_refs if ref[1] == label and ref not in supplement_refs and ref not in start_ref and ref[0] in reflection_text])
                        after = len(supplement_refs)
                        exception_other_refs_added += after - before
                if label == "Other":
                    for ref in supplement_refs:
                        assert (ref[0] in other_refs_text or ref[1] in other_labels_edit or ref in include_other_refs)

                label_count = ref_labels.count(label)
                start = label_count
                random.seed()
                random.shuffle(supplement_refs)
                added_refs = []
                for ref in supplement_refs:
                    if label == "Other":
                        ref[1] = "Other"
                    target_reflection.writerow(ref)
                    added_refs.append(ref)
                    total_added_refs.append(ref)
                    label_count += 1
                    if label_count == increase_to:
                        break

                if label == "Other":
                    for ref in added_refs:
                        assert ref[1] == "Other", ref[1]

                print(f"Appended {label_count-start} reflections with label class {label} to {current_ref}")
            print(f"Added {exception_other_refs_added} exception other refs.")

        # test cases
        end_ref = pd.read_csv(path)
        end_ref = end_ref.fillna(value="None")
        end_ref = list(end_ref.to_numpy().tolist())
        assert set([row[1] for row in end_ref]) == set([row[1] for row in start_ref]) # no unexpected labels added
        if "100" in path:  # no reflections from non-100% reflection sets added if current reflection is 100%
            for source in supplement_sources:
                assert "100" in source
        assert end_ref[:len(start_ref)] == start_ref  # no unexpected reflections added
        assert end_ref[len(start_ref):] == total_added_refs  # expected reflections appended on
        assert set([ref[1] for ref in added_refs]).issubset(set(supplement))  # no labels added not from supplement
        for ref in end_ref:  # ensure there are no duplicate reflections
            if ref[0].count("N/A") < 4:
                assert end_ref.count(ref) == 1, f"{start_ref.count(ref)}, {end_ref.count(ref)}, {ref}"
        print("\n")


def challenge_column_only(input_folder, output_folder):
    """
    Alter the text of a reflection set to only have the student's response to "What challenges are you currently facing?"
    :param input_folder: the place to source reflections from
    :param output_folder: the place to write the new reflections to
    :return: None
    """
    paths = glob.glob(f"{input_folder}/*.csv")

    for path in paths:
        dataset = pd.read_csv(path)
        dataset = dataset.fillna(value="None")

        def isolate_challenge(text):
            target = "What was your biggest challenge(s) for these past modules?"
            target_idx = text.index(target)
            target_idx = target_idx + len(target) + 1  # String index of the beginning of the target student response
            end_target = "How did you overcome this challenge(s)?"
            end_idx = text.index(end_target)-1
            response = text[target_idx:end_idx]
            print(response)
            return response

        dataset["text_alter"] = dataset["text"].apply(isolate_challenge)
        dataset = dataset[["text_alter", "label"]]

        dataset_name_idx = path.index("\\")
        dataset.to_csv(f"{output_folder}/{path[dataset_name_idx:]}")


def prepare_challenge_only_annotations(input_folder, output_file):
    """
    One-time use function to distill multiple reflection sets containing duplicate
    reflections into one reflection set with no duplicates
    :param input_folder: the folder to source the reflection sets from
    :param output_file: where to write the distilled reflection set to
    :return: None
    """
    paths = glob.glob(f"{input_folder}/*.csv")
    full_refs = {}

    for path in paths:
        file_idx = path.index("\\")+1
        file_name = path[file_idx:]
        file_name_stop_idx = file_name.index("-")
        ref_name = file_name[:file_name_stop_idx]
        full_refs.update({ref_name: pd.read_csv(path, keep_default_na=False)})

    final = pd.DataFrame(pd.concat(list(full_refs.values())))

    # for each reflection, identify where it comes from
    def determine_source(text):
        ref_names = []
        for name, ref in full_refs.items():
            if ref["text"].isin([text]).any():
                ref_names.append(name)
        return ref_names

    final["label"] = None  # placeholder label column
    final["source"] = final["text"].apply(determine_source)  # identify which reflection sets each reflection belongs to
    final = final.drop_duplicates(subset=["text"], keep="first")  # drop rows with duplicate text columns

    final.to_csv(f"{output_file}", index=False)

    test_challenge_only(full_refs, output_file)


def test_challenge_only(full_refs, prev_output_file):
    """
    Test method to ensure that r1 and r2 can be reconstructed from the output challenge_column_only.csv.
    :param full_refs: the unprocessed r1,r2 variations
    :return:
    """

    # Reconstruct R1-80,100 and R2-80,100 using the output file
    full = pd.read_csv(prev_output_file, keep_default_na=False)
    full = full.to_numpy().tolist()
    refs = {"r1_100": [], "r1_80": [], "r2_100": [], "r2_80": []}

    for entry in full:
        for source in eval(entry[-1]):
            refs[source].append(entry[0])

    for name, ref in list(refs.items()):
        print(f"{name}: {len(ref)}")

    print("\nTest")
    # Reconstruct R1-80,100 and R2-80,100 from unprocessed reflections
    for name, df in full_refs.items():
        print(f"{name} len before: {len(df)}")
        df_altered = df.drop_duplicates(subset=["text"], keep="first")
        assert len(refs[name]) == len(df_altered)
        assert set(refs[name]) == set(df_altered["text"].to_numpy().tolist())
        print(f"{name}: {len(df_altered)}")


# Dataset utilities
def main():
    # prepare_challenge_only_annotations(input_folder="challenge-only-refs", output_file="challenge_column_only.csv")

    # This code updates reflection sets from the unprocessed 1/27 versions on our MLCompare Datasets page
    # to the latest proficiency label reflection set version.

    # TODO add print method with the folder being output to
    # TODO add verbose switch that displays what I already have
    # TODO refactor for operationalizability in general

    # Generate datasets with consensus labels for the none and controversial annotations
    print("Processing none annotations...")
    single_label_consensus(input_folder="none_annotations", output_file="consensus_annotations/none_all.csv", update_label_sets=True)
    single_label_consensus(input_folder="none_annotations", output_file="consensus_annotations/none_100.csv", only_100=True)
    print("Processing controversial annotations...")
    single_label_consensus(input_folder="controversial_annotations", output_file="consensus_annotations/controversial_all.csv")
    single_label_consensus(input_folder="controversial_annotations", output_file="consensus_annotations/controversial_100.csv", only_100=True)

    use_original_data = True
    # Integrate the controversial and none consensus labels into the reflection 2,3,4 datasets
    for variation in ["controversial", "none"]:
        print(f"Adding {variation} reflection re-annotations to reflection 1, unsanitized...")
        add_new_refs_to_reflection(new_reflection=f"consensus_annotations/{variation}_all.csv",
                                   reflection=1,
                                   ref_agreement="unsanitized",
                                   output_folder="updated_datasets",
                                   use_original_data=use_original_data)
        print(f"Adding {variation} reflection re-annotations to reflection 2, unsanitized...")
        add_new_refs_to_reflection(new_reflection=f"consensus_annotations/{variation}_all.csv",
                                   reflection=2,
                                   ref_agreement="unsanitized",
                                   output_folder="updated_datasets",
                                   use_original_data=use_original_data)
        print(f"Adding {variation} reflection re-annotations to reflection 3, unsanitized...")
        add_new_refs_to_reflection(new_reflection=f"consensus_annotations/{variation}_all.csv",
                                   reflection=3,
                                   ref_agreement="unsanitized",
                                   output_folder="updated_datasets",
                                   use_original_data=use_original_data)
        print(f"Adding {variation} reflection re-annotations to reflection 4, unsanitized...")
        add_new_refs_to_reflection(new_reflection=f"consensus_annotations/{variation}_all.csv",
                                   reflection=4,
                                   ref_agreement="unsanitized",
                                   output_folder="updated_datasets",
                                   use_original_data=use_original_data)
        print(f"Adding {variation} reflection re-annotations to reflection 1, 100%...")
        add_new_refs_to_reflection(new_reflection=f"consensus_annotations/{variation}_100.csv",
                                   reflection=1,
                                   ref_agreement="100",
                                   output_folder="updated_datasets",
                                   use_original_data=use_original_data)
        print(f"Adding {variation} reflection re-annotations to reflection 2, 100%...")
        add_new_refs_to_reflection(new_reflection=f"consensus_annotations/{variation}_100.csv",
                                   reflection=2,
                                   ref_agreement="100",
                                   output_folder="updated_datasets",
                                   use_original_data=use_original_data)
        print(f"Adding {variation} reflection re-annotations to reflection 3, 100%...")
        add_new_refs_to_reflection(new_reflection=f"consensus_annotations/{variation}_100.csv",
                                   reflection=3,
                                   ref_agreement="100",
                                   output_folder="updated_datasets",
                                   use_original_data=use_original_data)
        print(f"Adding {variation} reflection re-annotations to reflection 4, 100%...")
        add_new_refs_to_reflection(new_reflection=f"consensus_annotations/{variation}_100.csv",
                                   reflection=4,
                                   ref_agreement="100",
                                   output_folder="updated_datasets",
                                   use_original_data=use_original_data)
        use_original_data = False

    print("Constructing split reflection dataset...")
    # Split the reflections in the target dataset into sub-responses via split_refs.csv
    construct_split_reflection_dataset(input_folder="updated_datasets", output_folder="split_datasets")

    print("Converting split reflection dataset into a question prepended dataset...")
    # Then prepend the questions each sub-response is answering to each sub-response
    construct_question_prepended_refs(input_folder="split_datasets", output_folder="prepended_datasets")

    print("Converting dataset to proficiency label dataset...")
    # Last, collapse non-proficiency labels into an "Other" column. Exclude these labels globally;
    # there are some edge cases (e.g. Group Work shouldn't really be a proficiency label in Reflection 3,
    # but could feasibly occur in it). For those, it's much easier to manually edit them.
    global_exclude_labels = ["Understanding requirements and instructions", "SDLC",
                             "Course Structure and Materials", "Other (Secondary Issue)", "HTML"]
    r1_exclude_labels = ["MySQL", "Github"]
    r2_exclude_labels = ["MySQL"]
    r3_exclude_labels = []
    r4_exclude_labels = []
    # TODO - add csv file for global_exclude_labels to allow the user to easily change it
    r1_other_refs = collapse_into_other(global_exclude=global_exclude_labels,
                        ref_num=1,
                        ref_exclude=r1_exclude_labels,
                        input_folder="prepended_datasets",
                        output_folder="final_datasets_2")
    r2_other_refs = collapse_into_other(global_exclude=global_exclude_labels,
                        ref_num=2,
                        ref_exclude=r2_exclude_labels,
                        input_folder="prepended_datasets",
                        output_folder="final_datasets_2")
    r3_other_refs = collapse_into_other(global_exclude=global_exclude_labels,
                        ref_num=3,
                        ref_exclude=r3_exclude_labels,
                        input_folder="prepended_datasets",
                        output_folder="final_datasets_2")
    r4_other_refs = collapse_into_other(global_exclude=global_exclude_labels,
                        ref_num=4,
                        ref_exclude=r4_exclude_labels,
                        input_folder="prepended_datasets",
                        output_folder="final_datasets_2")

    other_refs = pd.concat([r1_other_refs, r2_other_refs, r3_other_refs, r4_other_refs])
    assert len(other_refs.drop_duplicates()) == len(other_refs)  # ensure no duplicates
    other_refs_labels = other_refs["label"]

    supplement_labels = ["MySQL", "IDE and Package Installation", "Time Management and Motivation",
                         "API", "Python and Coding", "None", "Group Work", "Github", "Other"]

    # adds labels from other reflections to each label class from supplement_labels for each reflection
    supplement_label_classes(supplement_labels=supplement_labels,
                             increase_to=10000,
                             other_refs=other_refs,
                             other_labels=global_exclude_labels,
                             input_folder="final_datasets")

    update_label_sets()  # only updates label sets with the altered none label sets for now

    challenge_column_only(input_folder="final_datasets", output_folder="final_datasets_challenge_only")


if __name__ == "__main__":
    main()

### Currently Unused / Old Functions ###
# The functions below were used for debugging and/or the construction of the latest reflection 1 (the methodology
# being the same, just that the process wasn't as streamlined so more code was needed. We determined the dataset
# methodology concurrently as we updated reflection 1, whereas the methodology was already laid out when it came
# time to update reflections 2,3,4).


def dataset_count():
    path_names = glob.glob("sanitized_datasets" + '\\*')

    with open("results.csv", "w", encoding="utf-8", newline="") as r:
        c_w = csv.writer(r)

        for path in path_names:
            c_w.writerow([path[-12:-4]])
            df = pd.read_csv(path)
            df = df["label"]
            labels = df.to_numpy().tolist()
            count = {}
            for label in set(labels):
                count.update({label: labels.count(label)})

            for item in count.items():
                c_w.writerow(item)
            c_w.writerow([])


def filter_refs_controversial():
    df_san = pd.read_csv("refs_sanitized.csv")
    df_unsan = pd.read_csv("refs_unsanitized.csv")

    df_controversial = df_unsan[~df_unsan["text"].isin(df_san["text"])]

    print(len(df_controversial))
    assert len(df_controversial) == 138, len(df_controversial)
    df_controversial.to_csv("controversial.csv", index=False)


def filter_ref_substrings():
    contr = refs_substr = contr_substrings = []
    with open("controversial.csv", "r", encoding="utf-8") as c:
        contr = list(csv.reader(c))
        contr = [row[1] for row in contr]

    with open("reflection_substrings.csv", "r", encoding="utf-8") as rs:
        refs_substr = list(csv.reader(rs))

    for ref_sub in refs_substr:
        ref = " ".join(ref_sub)
        if ref in contr:
            contr_substrings.append(ref_sub)

    assert len(contr_substrings) == 138, f"massive failure, {len(contr_substrings)}"
    assert [" ".join(ref) for ref in contr_substrings] == contr[1:]

    pd.DataFrame(contr_substrings).to_csv("controversial_substrings.csv", header=False, index=False)


def convert_no_issue_label_sets():
    with open("updated_label_sets.csv", "r", encoding="utf-8") as uls:
        reflections = list(csv.reader(uls))
        refs = [row[0] for row in reflections]
        len_check = len(reflections)

    for row, i in zip(reflections, range(0, len(reflections))):
        labels = eval(row[1])
        for l_set, j in zip(labels, range(0, len(labels))):
            if "No Issue" in l_set:
                labels[j][0] = "None"  # no issue will only ever be the first label

        reflections[i] = (reflections[i][0], labels)

    assert len(reflections) == len_check
    assert [row[0] for row in reflections] == refs

    with open("final_label_sets.csv", "w", encoding="utf-8", newline="") as fls:
        c_w = csv.writer(fls)
        c_w.writerows(reflections)


# update ref labels when the only case is to update existing refs
def update_ref_labels_simple(dataset_path):
    paths = glob.glob("Updated Datasets/no_controversial_ref_datasets/*.csv")
    if not os.path.isdir("Updated Datasets/updated_none_ref_datasets"):
        os.mkdir("Updated Datasets/updated_none_ref_datasets")

    with open(dataset_path, "r", encoding="utf-8") as p:
        none_refs = list(csv.reader(p))
        none_refs_text = [row[0] for row in none_refs]

    for path in paths:
        with open(path, "r", encoding="utf-8") as p:
            dataset = list(csv.reader(p))[1:]
            check_len = len(dataset)

        for entry, i in zip(dataset, range(0, len(dataset))):
            if entry[0] in none_refs_text:
                dataset[i] = none_refs[none_refs_text.index(entry[0])]

        assert len(dataset) == check_len

        target = "Updated Datasets/updated_none_ref_datasets/" + path[47:]
        with open(target, "w", encoding="utf-8", newline="") as p:
            c_w = csv.writer(p)
            c_w.writerow(["text", "label"])
            c_w.writerows(dataset)

    check_dataset_correctness("updated_none_ref_datasets")


def filter_by_reflection():
    # would use pandas here for a massive code simplification but there's a weird bug in pandas
    # where reading a new line character causes ...something... to happen such that the string containing
    # the new line character in the dataframe does not evaluate as equivalent to the same string in either
    # another dataframe or a list of strings. Since a few of our reflections contain new line characters, I'm
    # avoiding the bug and just not using pandas here. I also don't use pandas in a couple other places in this
    # code for the same reason
    paths = glob.glob("Updated Datasets/*.csv")
    with open("consensus_controversial_refs.csv", "r", encoding="utf-8") as ccr:
        filter_by = list(csv.reader(ccr))  # reflection text of reflections
        filter_by = [ref[0] for ref in filter_by[1:]]
    # where the re-annotated consensus label was Controversial
    check_len = len(filter_by)
    if not os.path.isdir("Updated Datasets/no_controversial_ref_datasets"):
        os.mkdir("Updated Datasets/no_controversial_ref_datasets")

    removed = []
    i = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as p:
            dataset = list(csv.reader(p))

        filtered = []
        for row in dataset:
            if row[0] not in filter_by:
                filtered.append(row)
            else:
                removed.append(row[0])

        print(f"new_len: {len(filtered)}, old_len: {len(dataset)} for {path}")
        assert len(filtered) <= len(dataset), f"new_len: {len(filtered)}, old_len: {len(dataset)} on {path}"

        target_path = "Updated Datasets/no_controversial_ref_datasets/fixed_" + path[17:]
        with open(target_path, "w", encoding="utf-8", newline="") as tp:
            c_w = csv.writer(tp)
            c_w.writerows(filtered)

    for ref in removed:
        print(ref)

    print(set(filter_by).difference(set(removed)))
    assert list(set(filter_by)) == list(set(removed))
    assert check_len == len(set(removed)), f"should have removed {check_len} but removed {len(removed)}"
    check_dataset_correctness("no_controversial_ref_datasets")


def get_refs_by_label(label):
    with open("Updated Datasets/no_controversial_ref_datasets/fixed_sl-all-80.csv", "r", encoding="utf-8") as xyz:
        dataset = list(csv.reader(xyz))
    count = [row[1] for row in dataset[1:]].count(label)
    filtered_by_label = [row for row in dataset if row[1] == label]

    assert count == len(filtered_by_label)
    assert list(set([row[1] for row in filtered_by_label])) == ["None"]

    filtered_by_label = [[row[0]] for row in filtered_by_label]

    name = "Updated Datasets/" + label + "_reflections" + ".csv"
    with open(name, "w", encoding="utf-8", newline="") as new_refs:
        c_w = csv.writer(new_refs)
        c_w.writerows(filtered_by_label)


def check_dataset_correctness(folder):
    sl_all_hundred = pd.read_csv("Updated Datasets/" + folder + "/fixed_sl-all-100.csv")
    sl_all_eighty = pd.read_csv("Updated Datasets/" + folder + "/fixed_sl-all-80.csv")
    sl_r1_hundred = pd.read_csv("Updated Datasets/" + folder + "/fixed_sl-r1-100.csv")
    sl_r1_eighty = pd.read_csv("Updated Datasets/" + folder + "/fixed_sl-r1-80.csv")

    should_be_empty = sl_all_hundred[~sl_all_hundred["text"].isin(sl_all_eighty["text"])]
    assert len(should_be_empty) == 0, f"{len(should_be_empty)} at allhundred vs alleighty"
    should_be_empty = sl_r1_hundred[~sl_r1_hundred["text"].isin(sl_r1_eighty["text"])]
    assert len(should_be_empty) == 0, f"{len(should_be_empty)} at r1hundred vs r1eighty"
    should_be_empty = sl_r1_hundred[~sl_r1_hundred["text"].isin(sl_all_hundred["text"])]
    assert len(should_be_empty) == 0, f"{len(should_be_empty)} at r1hundred vs allhundred"
    should_be_empty = sl_r1_eighty[~sl_r1_eighty["text"].isin(sl_all_eighty["text"])]
    assert len(should_be_empty) == 0, f"{len(should_be_empty)} at r1hundred vs allhundred"

    print("All clear g")


def nltk_annotation_formatting(dataset):
    ret_dataset = []
    i = 0
    coder_id = 0
    for row in dataset:
        for ann_set in eval(row[1]):
            annotator = "coder_" + str(coder_id)
            ret_dataset.append((annotator, i, frozenset(ann_set)))
            coder_id += 1
        i += 1
        coder_id = 0
    return ret_dataset


# This function calculates and prints the Krippendorff's alpha value for the
# previous unfiltered dataset and the newly filtered dataset to validate that
# filtering the dataset improved the inter-annotator agreement
def validation(prev_dataset, new_dataset):
    formatted_annts_new = nltk_annotation_formatting(new_dataset)
    formatted_annts_prev = nltk_annotation_formatting(prev_dataset)

    task_prev = AnnotationTask(data=formatted_annts_prev, distance=binary_distance)
    task_new = AnnotationTask(data=formatted_annts_new, distance=binary_distance)

    return task_prev.alpha(), task_new.alpha()


def convert_to_no_issue_from_dataset(dataset):
    for entry, i in zip(dataset, range(0, len(dataset))):
        if entry[1] == "None":
            dataset[i][1] = "No Issue"
    return dataset


def convert_from_no_issue_from_file():
    with open("no_issue_refs.csv", "r", encoding="utf-8") as nir:
        reflections = list(csv.reader(nir))

    for i in range(1, len(reflections)):
        if reflections[i][1] == "No Issue":
            reflections[i] = (reflections[i][0], "None")

    with open("fixed_refs.csv", "w", encoding="utf-8", newline="") as fr:
        c_w = csv.writer(fr)
        c_w.writerows(reflections)
