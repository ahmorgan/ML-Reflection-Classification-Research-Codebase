import pandas as pd
import glob
from collections import Counter
import copy
import random

ref_to_question_included = {}  # maps every reflection in our superset to question included version as a list


### Relevant Functions ###
# The functions below have been streamlined for upgrading old reflection datasets to the latest version.


def single_label_consensus(input_folder, only_100=False, exclude_controversial=False,
                           update_label_sets=False):
    """
    Process a set of new single-label annotations into one dataset of consensus labels.

    :param input_folder: path to annotation csvs
    :param only_100: bool: if True, only consensus labels with 100% agreement are included
    :param exclude_controversial: bool: if True, reflections where the consensus label is
    "Controversial" are excluded from the resultant dataset
    :param update_label_sets: bool: if True, the label sets will be updated with the new consensus labels
    :return consensus dataset created from input_annotations
    """
    paths = glob.glob(f"{input_folder}/*.csv")

    consensus = pd.DataFrame()
    annotation = pd.DataFrame(pd.read_csv(paths[0], keep_default_na=False), columns=["text", "label"])  # just to get the reflection texts
    consensus["text"] = annotation["text"]
    for path in paths:
        consensus[path] = pd.read_csv(path, keep_default_na=False)["label"]  # instantiate each annotation set as a column of labels
    if "sources" in annotation.columns:
        consensus["sources"] = annotation["sources"]

    # Get the consensus label based on each label annotation
    def con(row):
        labels = row.iloc[1:1+len(paths)].tolist()
        if only_100 and len(Counter(labels).most_common()) > 1:  # Disagreement in the relabelings, exclude in 100 agreement
            return "remove"
        # in the case of a complete disagreement, avoid a bias towards labels that occur earlier in the alphabet by shuffling
        if len(set(labels)) == len(labels):
            if "Multi-label" in labels:
                labels.remove("Multi-label")  # in the case of a 3-way disagreement, don't choose multi-label if it occurs.
                # since two people didn't say multi-label and one did, it makes more sense to assume the reflection is not
                # multi-label
            label_counts = Counter(labels).most_common()
            random.shuffle(label_counts)
            consensus_label = label_counts[0][0]
        # Normal case
        else:
            consensus_label = Counter(labels).most_common(1)[0][0]

        if consensus_label == "None (No Issue)":
            consensus_label = "None"
        if consensus_label == "Multi-label":
            return "remove"

        return consensus_label

    consensus["label"] = consensus.apply(con, axis=1)
    consensus = consensus[consensus["label"] != 'remove']

    # TODO:
    """
    if not only_100 and input_folder == "challenge_annotations":
        label_sets = pd.DataFrame(consensus["text"])
        # assuming single-label
        label_sets["label"] = consensus.apply(lambda row: [[label] for label in row.iloc[1:1 + len(paths)].tolist()],
                                              axis=1)
        label_sets.to_csv("challenge_annts_label_sets_80.csv", index=False, header=False)
    """

    consensus_temp = consensus[["text", "label"]]
    if "sources" in consensus.columns:
        consensus_temp["sources"] = consensus["sources"]

    assert [row[0] for row in consensus.to_numpy()] == [row[0] for row in consensus_temp.to_numpy()]
    if exclude_controversial:
        consensus_temp = consensus_temp[consensus_temp["label"] != "Controvercial"]

    """ OLD
    # save the reflections where the consensus is that the reflection is a secondary issue
    consensus_contr = consensus_temp[consensus_temp["label"] == target]
    consensus_contr = consensus_contr["text"]
    consensus_contr.to_csv("controversial_annotations/consensus_other_refs.csv", index=False)

    # save the actual none reflections separately
    consensus_temp = consensus_temp[consensus_temp["label"] != target]
    consensus_temp.to_csv("controversial_annotations/consensus_labels_none.csv", index=False)
    """

    # TODO:
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
    """

    if not exclude_controversial:
        assert len(consensus) == len(consensus_temp), f"{len(consensus)}, {len(consensus_temp)}"

    return consensus_temp.to_numpy().tolist()


def add_new_refs_to_dataset(input_dataset, integrate_data):
    """
    Integrates new reflections into a dataset by either appending them if they are new reflections (reflections outside
    of that reflection number, e.g. new reflections not in reflection 2) or replacing the reflection in-place if it
    already exists in the reflection

    :param input_dataset: iterable: the reflection dataset to be updated
    :param integrate_data: iterable: the reflection data to be integrated into another reflection dataset
    """

    # the reflections to be integrated into the dataset
    if integrate_data[0] == ["text", "label"]:
        new_refs = integrate_data[1:]
    else:
        new_refs = integrate_data

    # dataset to be altered
    if input_dataset[0][0] == "text" and input_dataset[0][1] == "label":
        input_dataset = input_dataset[1:]
    input_dataset_text = [row[0] for row in input_dataset]

    # TODO:
    """
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
    """

    refs_appended_to_new_data = []
    refs_added_count = 0

    for row in new_refs:
        # row[0] = text, row[1] = label
        if row[1] != "Controvercial":
            # case 1: reflection is in the superset of reflections for that dataset but not in the
            # dataset we are altering. Then, append the new reflection to the end of that dataset
            if row[0] not in input_dataset_text:
                refs_added_count += 1
                input_dataset.append(row)
                refs_appended_to_new_data.append(row)
            # case 2: reflection is in the superset of reflections for that dataset and also is in the
            # dataset we are altering. Then, alter that reflection in place with the updated labels.
            else:
                # In the case of duplicate reflections, just update all of those reflections
                # with the same value all at once
                if input_dataset_text.count(row[0]) > 1:
                    ref_indices = [index for index, value in enumerate(input_dataset_text) if value == row[0]]
                    for idx in ref_indices:
                        input_dataset[idx] = row
                else:
                    ref_idx = input_dataset_text.index(row[0])
                    input_dataset[ref_idx] = row
        else:
            # Reflections in the target reflection that have been relabeled as "controversial"
            # (spelled wrong intentionally because it was spelled wrong in the dataset) should be
            # removed from the dataset if they are currently in it
            if row[0] in input_dataset_text:
                ref_idx = input_dataset_text.index(row[0])
                input_dataset[ref_idx] = None  # deleting the reflection now alters the reflection indices, which
                # makes things annoying, so instead, mark them as None and delete them later.

        # update the superset of label sets as we update the dataset
        # TODO:
        """
        if update_label_sets:
            idx = full_ref_text.index(row[0])
            new_label_set = new_label_sets[new_label_sets_text.index(row[0])]
            full_label_sets[idx] = new_label_set
        """

    prev_len = len(input_dataset)
    input_dataset = [row for row in input_dataset if row]  # removing None reflections, which were marked for removal because
    # they were relabeled as "controversial"
    removed_len = prev_len - len(input_dataset)
    print(f"Removed {removed_len} controversial reflections.")
    print(f"Added {refs_added_count} relabeled reflections.")

    _test_new_dataset(input_dataset, new_refs, refs_appended_to_new_data)

    # TODO:
    """
    if update_label_sets:
        with open("updated_label_sets.csv", "w", encoding="utf-8", newline="") as ls:
            c_w = csv.writer(ls)
            c_w.writerows(full_label_sets)
    """

    input_dataset.insert(0, ["text", "label"])
    return input_dataset


def _test_new_dataset(new_data, added_data, refs_appended_to_new_data):
    added_data_text = [row[0] for row in added_data]
    contr_refs = [row[0] for row in added_data if row[1] == "Controvercial"]
    new_refs = [row[0] for row in new_data]

    # check no refs with the "Controvercial" label are in the new data
    for ref in contr_refs:
        assert ref not in new_refs

    should_be_added = [row for row in added_data if row[1] != "Controvercial"]

    # check that every reflection that should have been appended to new_data was, and that
    # the label is correct for that reflection
    for row in should_be_added:
        assert row[0] in new_refs, row[0]  # reflection text is in the updated dataset
        assert added_data[added_data_text.index(row[0])][1] == new_data[new_refs.index(row[0])][1], f"{added_data[added_data_text.index(row[0])][1]} does not match {new_data[new_refs.index(row[0])][1]}"   # labels match

    for row in added_data:
        # for reflections that were appended to new_data, check that they are actually in whatever
        # reflection number they should be in, and that their label matches that in added_data
        if row[0] in refs_appended_to_new_data:
            assert row[1] == refs_appended_to_new_data[refs_appended_to_new_data.index(row[0])][1]

        # check if all labels from reflections in added_data that are in new_refs have been changed in new_refs
        if row[0] in new_refs and row[1] != "Controvercial":
            assert new_data[new_refs.index(row[0])][1] == row[1]


# Obsolete code below, see construct_question_prepended_refs()
def _construct_question_included_map(subreflections):
    ref_to_question_included.clear()

    # split_refs is all of our reflections in our dataset split into sub-responses
    # based on the question the student is responding to
    # subreflections[0] should be the header
    subreflections = subreflections[1:]
    subreflections_text = [" ".join(row) for row in subreflections]

    for text, ref in zip(subreflections_text, subreflections):
        ref_to_question_included.update({text: ref})


def construct_split_reflection_dataset(input_dataset, subreflections, questions):
    """
    Alter the reflection text of a dataset to be split into reflection subresponses, with one column for each subresponse.

    :param input_dataset: the dataset to be updated
    :param subreflections: reflections split into subresponses
    :param questions: ordered list of the questions the students are responding to (e.g., ["How do you feel about this class so far?", "..."]
    :return: updated dataset
    """
    _construct_question_included_map(subreflections)  # dictionary mapping reflection text to split reflection text

    prev_len = len(input_dataset)
    dataset = copy.deepcopy(input_dataset[1:])  # need to deepcopy so input_dataset isn't also updated (unaltered input_dataset used for testing later)
    for row, i in zip(dataset, range(0, len(dataset))):
        # stupid edge case that happens exactly once where it's easier ot just hardcode it
        if row[0] == "😍Excited <name omitted> is the best <name omitted> is the best <name omitted> is the best <name omitted> is the best":
            q_inc = copy.deepcopy([row[0], "", "", "", ""])
        else:
            q_inc = copy.deepcopy(ref_to_question_included[row[0]])
        q_inc.append(row[1])  # q inc is the split refs as individual entries in a list plus the label
        dataset[i] = copy.deepcopy(q_inc)

    if questions[-1] != "label":
        questions.append("label")
    dataset.insert(0, questions)

    dataset_text = [row[0] for row in input_dataset]
    assert prev_len == len(dataset), f"{prev_len}, {len(dataset)}"
    for ref, ref_2 in zip(dataset_text[1:], dataset[1:]):
        if ref_2[0] == "😍Excited <name omitted> is the best <name omitted> is the best <name omitted> is the best <name omitted> is the best":
            continue
        assert ref == " ".join(ref_2[:5]), f"{ref}, \n {ref_2[:5]}"

    return dataset


def construct_question_prepended_refs(input_dataset):
    """
    Update the text of a reflection dataset to include the reflection questions.

    SPECIAL CASE: the input dataset must already in some way subdivide the reflections into the student's subresponses in each row,
    should have the label column as the last column, and should have a header row with the format [ref_question1, ref_question2, ..., label].
    This is necessary so that the reflection questions can be matched with their corresponding responses.

    :param input_dataset: the dataset to be updated
    :return: the updated dataset
    """
    df = pd.DataFrame(input_dataset[1:], columns=input_dataset[0])

    # pandas reads the string "None" as NaN, change back to None
    df["label"] = df["label"].fillna(value="None")
    # check_len = len(df)
    df_refs = df.drop(columns=["label"])
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
    df_refs["label"] = df["label"].values
    df_refs = df_refs[["text", "label"]]

    # assert check_len == len(df_refs), f"{check_len}, {len(df_refs)}"

    return df_refs.to_numpy().tolist()


# for creating the proficiency label dataset
def collapse_into_other(input_dataset, other_labels):
    """
    Alter the label class of specified reflections in a single reflection dataset to "Other".
    :param input_dataset: iterable, the dataset to update
    :param other_labels: which labels are to be altered to "Other" for that reflection dataset
    :return: the updated dataset and the "Other" reflections as they where before being updated
    """
    print(f"Putting labels into 'Other' label class: {other_labels}")
    df = pd.DataFrame(input_dataset, columns=["text", "label"])

    other_refs = df[df["label"].isin(other_labels)]  # for test case later

    # For each label, if it's one of the labels that should be in "Other", make it so
    df["label"] = df["label"].apply(lambda label: "Other" if label in other_labels else label)

    def check(label):
        assert label and label not in other_labels, "Massive failure"
        return label

    # Test case: check if the intended behavior was performed
    df["label"].apply(check)

    def check_other_refs(label):
        assert label in other_labels

    other_refs["label"].apply(check_other_refs)

    assert len(other_refs.drop_duplicates()) == len(other_refs)

    return df.to_numpy().tolist(), other_refs


# TODO:
"""
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
"""


def supplement_label_classes(all_datasets, supplement_dataset_name, supplement_labels, increase_to, other_refs, other_labels):
    """
    Supplement the label classes of a single reflection dataset with reflections of that label class from other reflection datasets.

    :param all_datasets: dictionary of format {name: dataset}: all datasets to use to supplement the target dataset supplement_dataset_name. Should include the dataset to be supplemented.
    :param supplement_dataset_name: the reflection dataset to supplement. Name should be present in all_datasets
    :param supplement_labels: the label classes to supplement
    :param increase_to: the number of occurrences to increase each label class to (will either increase to that
    number or add as many as possible if that number cannot be reached). Any label class that already occurs more
    than increase_to times is excluded from supplementation. "max" for max-shot label supplementing.
    :param other_refs: reflection text for all reflections with the Other label
    :param other_labels: all labels in the Other label class for the target reflection dataset
    :return: supplemented reflection dataset
    """
    if increase_to == "max":
        increase_to = 0
        for dataset in all_datasets.values():
            increase_to += len(dataset)  # upper bound on maximum possible label supplementation

    if not set(supplement_labels).isdisjoint(set(other_labels)):
        raise ValueError("Cannot supplement label that is considered 'Other' in this reflection (besides 'Other' itself)! Make sure supplement_labels and other_laels do not overlap.")

    # path = supplement_dataset_name
    start_ref = copy.deepcopy(all_datasets[supplement_dataset_name][1:])  # for testing later

    target_reflection = all_datasets[supplement_dataset_name][1:]

    # only include label classes that are in that reflection set and occur less often than increase_to
    ref_labels = [row[1] for row in target_reflection]
    ref_labels = [label for label in ref_labels if ref_labels.count(label) < increase_to]
    supplement = [label for label in supplement_labels if label in set(ref_labels)]

    # get all of the reflection sets that are not the current reflection
    supplement_sources = [name for name in all_datasets.keys() if name != supplement_dataset_name]

    # only supplement 100% agreement reflection sets with reflections from other 100% agreement sets
    if "100" in supplement_dataset_name:
        supplement_sources = [source for source in supplement_sources if "100" in source]

    # In the case that the underlying label of "Other" is in the included labels for a dataset,
    # get those reflections and add them to the bank of reflections to choose from
    include_other_refs = [ref for ref in other_refs.to_numpy().tolist() if ref[1] not in other_labels]

    # Only the text of "Other" reflections with underlying labels that match the underlying labels
    # for "Other" for that reflection set. e.g., for reflection 1, mysql and github as well as the global
    # other labels make up the set of underlying labels for the Other label class, so we only want Other
    # reflections that have labels from that underlying label set
    other_refs_text = [ref[0] for ref in other_refs.to_numpy().tolist() if ref[1] in other_labels]

    print(f"Sourcing from {supplement_sources}")

    total_added_refs = []
    exception_other_refs_added = 0

    for label in supplement:
        supplement_refs = []
        for supplementer_name in supplement_sources:
            reflection = all_datasets[supplementer_name][1:]
            reflection_text = [ref[0] for ref in reflection]

            # avoid duplicate reflections
            if label == "Other":
                # Since reflections in supplementer_name might already have reflections converted to "Other",
                # we can't directly know the underlying label for "Other" reflections. Instead, I've
                # saved the text of the "Other" reflections and am only sourcing reflections from there.
                # I can't source directly from other_refs because other_refs contains all
                # Other reflections, including ones from unsanitized datasets, so we first only look
                # in reflection and then check if that reflection is in other_refs.
                supplement_refs.extend(
                    [ref for ref in reflection if (ref[0] in other_refs_text or ref[1] in other_labels)
                     and ref not in supplement_refs and ref not in start_ref])
            else:
                supplement_refs.extend([ref for ref in reflection if ref[1] == label and ref not in supplement_refs and ref not in start_ref])

            before = len(supplement_refs)
            # To avoid adding exception Other reflections that are not 100% agreement, the candidate reflection must also be in one of the source reflections
            supplement_refs.extend([ref for ref in include_other_refs
                                    if ref[1] == label and ref not in supplement_refs and ref not in start_ref and ref[0] in reflection_text])
            after = len(supplement_refs)
            exception_other_refs_added += after - before
        if label == "Other":
            for ref in supplement_refs:
                assert (ref[0] in other_refs_text or ref[1] in other_labels or ref in include_other_refs)

        label_count = ref_labels.count(label)
        start = label_count
        random.seed()
        random.shuffle(supplement_refs)
        added_refs = []
        for ref in supplement_refs:
            if label == "Other":
                ref[1] = "Other"
            target_reflection.append(ref)
            added_refs.append(ref)
            total_added_refs.append(ref)
            label_count += 1
            if label_count == increase_to:
                break

        if label == "Other":
            for ref in added_refs:
                assert ref[1] == "Other", ref[1]

        print(f"Appended {label_count - start} reflections with label class {label} to {supplement_dataset_name}")

    # test cases
    end_ref = target_reflection
    assert set([row[1] for row in end_ref]) == set([row[1] for row in start_ref])  # no unexpected labels added
    if "100" in supplement_dataset_name:  # no reflections from non-100% reflection sets added if current reflection is 100%
        for source in supplement_sources:
            assert "100" in source
    assert end_ref[:len(start_ref)] == start_ref  # no unexpected reflections added
    assert end_ref[len(start_ref):] == total_added_refs  # expected reflections appended on
    assert set([ref[1] for ref in total_added_refs]).issubset(set(supplement))  # no labels added not from supplement

    return target_reflection


def challenge_column_only(input_dataset):
    """
    Alter the text of a reflection dataset to only have the student's response to "What challenges are you currently facing?".
    The input reflection dataset should have a text column where all entries include the text "What was your biggest challenge(s) for these past modules?"
    with the student's response following, and then should also include the following "How did you overcome this challenge(s)?"
    question to bound the student's challenge column response.

    :param input_dataset: iterable, dataset to update
    :return: updated input_dataset
    """
    input_dataset = pd.DataFrame(input_dataset, columns=["text", "label"])

    def isolate_challenge(text):
        target = "What was your biggest challenge(s) for the past modules?"
        target_idx = text.index(target)
        target_idx = target_idx + len(target) + 1  # String index of the beginning of the target student response
        end_target = "How did you overcome this challenge(s)?"
        end_idx = text.index(end_target) - 1
        response = text[target_idx:end_idx]
        return response

    input_dataset["text_alter"] = input_dataset["text"].apply(isolate_challenge)
    dataset = input_dataset[["text_alter", "label"]]

    return dataset.to_numpy().tolist()


def prepare_challenge_only_annotations(input_datasets, output_file):
    """
    Distills multiple reflection sets containing duplicate
    reflections into one reflection set with no duplicates. Used to prepare challenge-column
    only annotation templates from reflection sets with overlapping reflections

    :param input_datasets: dictionary of format {dataset_name: dataset} to distill/concatenate into one annotation set
    :param output_file: where to write the distilled reflection set to
    :return: None
    """
    dataset_dfs = [pd.DataFrame(dataset, columns=["text", "label"]) for dataset in list(input_datasets.values())]
    final = pd.concat(dataset_dfs, axis=1)

    # for each reflection, identify where it comes from
    def determine_source(text):
        ref_names = []
        for name, ref in input_datasets.items():
            if ref["text"].isin([text]).any():
                ref_names.append(name)
        return ref_names

    final["label"] = None  # placeholder label column
    final["source"] = final["text"].apply(determine_source)  # identify which reflection sets each reflection belongs to
    final = final.drop_duplicates(subset=["text"], keep="first")  # drop rows with duplicate text columns

    final.to_csv(f"{output_file}", index=False)

    _test_challenge_only(input_datasets, final.copy())


def _test_challenge_only(input_datasets, final):
    """
    Test method to ensure that the reflections can be reconstructed from the output annotation set
    :param input_datasets: the unprocessed r1,r2 variations
    :return: None
    """

    # Reconstruct R1-80,100 and R2-80,100 using the output file
    final = final.to_numpy().tolist()
    refs = {source: [] for source, _ in input_datasets.items()}

    for entry in final:
        for source in entry[-1]:
            refs[source].append(entry[0])

    print("\nTest")
    # Reconstruct datasets from unprocessed reflections
    for name, df in input_datasets.items():
        print(f"{name} len before: {len(df)}")
        df_altered = df.drop_duplicates(subset=["text"], keep="first")
        assert len(refs[name]) == len(df_altered)
        assert set(refs[name]) == set(df_altered["text"].to_numpy().tolist())
        print(f"{name}: {len(df_altered)}")


def reconstruct_dataset(input_dataset, target_variation):
    """
    If input_file is a dataset that contains reflections from multiple other datasets, provided a column
    noting the dataset(s) that reflection came from, reconstruct the original datasets

    :param input_dataset: iterable, dataset with a "source" column from which to reconstruct
    :param target_variation: string of the target reflection set/dataset to construct, e.g., "r1-80".
    :return: input_dataset filtered down to the target variation
    """
    reconstructed_dataset = []

    for entry in input_dataset:
        # In the case of the challenge reflections specifically, the reflections are the same for r1 and r2
        if target_variation in entry[2]:
            reconstructed_dataset.append([entry[0], entry[1]])

    return reconstructed_dataset


def update_labels_with_new_annotation(input_dataset, master_dataset):
    """
    Updates the labels in input with the labels of the corresponding reflections in master_dataset

    :param input_dataset: dataset to update
    :param master_dataset: source of updates
    :return: input_dataset with updated labels
    """
    input_dataset = pd.DataFrame(input_dataset, columns=["text", "label"])
    start_ref = input_dataset.copy()  # for testing later

    master_dataset = pd.DataFrame(master_dataset, columns=["text", "label"])
    df_master_ref = master_dataset.copy()  # for testing later

    master_dataset = master_dataset.to_numpy().tolist()
    master_dataset_map = {}
    for reflection in master_dataset:
        master_dataset_map.update({reflection[0]: reflection[1]})  # ref to label map

    def update_label(row):
        row = row.to_numpy().tolist()  # listify series to attmept to appease the neverending desires of pandas
        if row[0] not in list(master_dataset_map.keys()):
            return "remove"  # reflection was re-annotated as multi label or reannotated as not 100% agreement, remove
        return master_dataset_map[row[0]]

    input_dataset["new_label"] = input_dataset.apply(update_label, axis=1)

    input_dataset = input_dataset[["text", "new_label"]]

    input_dataset = input_dataset[input_dataset["new_label"] != "remove"]

    input_dataset.columns = ["text", "label"]
    input_dataset = input_dataset.drop_duplicates(subset=["text"], keep="first")

    print(f"Dataset length before: {len(start_ref)}, length after: {len(input_dataset)}\n")

    master_ref_text = [row[0] for row in input_dataset]

    # check if labels match in datasets
    def check(row):
        check_row = row.to_numpy().tolist()
        assert master_dataset[master_ref_text.index(check_row[0])][1] == check_row[1]
        return row

    input_dataset.apply(check, axis=1)

    assert len(input_dataset) <= len(start_ref)  # did increase the number of reflections
    assert len(input_dataset[~input_dataset["text"].isin(df_master_ref["text"])]) == 0  # no reflections in updated ref not in superset
    assert set(input_dataset["text"].to_numpy().tolist()).issubset(set(df_master_ref["text"].to_numpy().tolist()))

    return input_dataset.to_numpy().tolist()


def find_unique_reflections(find_unique: pd.DataFrame, compare_to: pd.DataFrame):
    """
    Find the reflections in find_unique that are not in compare_to
    :param find_unique:
    :param compare_to:
    :return:
    """
    find_unique = pd.DataFrame(find_unique, columns=["text", "label"])
    compare_to = pd.DataFrame(compare_to, columns=["text", "label"])
    return find_unique[~find_unique["text"].isin(compare_to["text"])].to_numpy().tolist()
