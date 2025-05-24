import copy
import csv
from itertools import combinations
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import binary_distance
from matplotlib import pyplot as plt
import pandas as pd


# filters out reflections that, by consensus of number of labels, have only
# one label. That is, the average label set length is one
# example: three annotators provide the label sets, [Python and Coding], [Python and Coding],
# [Python and Coding, GitHub] -- the average label set length is (1+1+2)/3 = 4/3 = rounded to 1
# ***NOTE: in this method, we don't care what the actual label is (i.e. it is OK if the three annotators each provided
# three single labels that are all different) because low disagreement reflections (or those with
# varying label sets) will get filtered out anyway
def _single_label_filter(dataset):
    ret_dataset = []

    # dataset is a list of tuples which each contain a reflection and its corresponding label sets
    # both items in the tuple are strings since dataset was read from a csv file
    for row in dataset:
        labels = row[1]
        avg_len = 0
        for label_set in labels:
            avg_len += len(label_set)
        avg_len = round(avg_len / len(labels))

        # Since label_setsOLD.csv / dataset contains only the labels used to calculate the consensus labels,
        # if the average length of the labels in dataset is 1, there is only one consensus label attached to
        # that reflection in full_dataset.csv
        if avg_len == 1:
            ret_dataset.append(row)

    return ret_dataset


# Fixed version of NLTK's masi_distance()
# MASI = jaccard_distance * m, the original NLTK implementation
# had (1-jaccard) * m (jaccard = cardinality(intersection)/cardinality(union))
# also per the original MASI paper:
# "If two sets Q and P are identical, M is 1. If one
# set is a subset of the other, M is 2/3. If the intersection
# and the two set differences are all non-null, then M is 1/3.
# If the sets are disjoint, M is 0." (Passonneau, 2006)
# http://www.lrec-conf.org/proceedings/lrec2006/pdf/636_pdf.pdf
def _masi_distance(label1, label2):
    len_intersection = len(label1.intersection(label2))
    len_union = len(label1.union(label2))
    if label1 == label2:  # two sets identical, m = 1
        m = 1
    elif label1.issubset(label2) or label2.issubset(label1):
        m = 0.67
    elif len_intersection > 0 and len(label1.difference(label2)) > 0 and len(label2.difference(label1)) > 0:
        m = 0.33
    else:  # two sets disjoint, m = 0
        m = 0
    return (len_intersection / float(len_union)) * m


def _nltk_annotation_formatting(dataset):
    ret_dataset = []
    i = 0
    coder_id = 0
    for row in dataset:
        for ann_set in row[1]:
            annotator = "coder_" + str(coder_id)
            ret_dataset.append((annotator, i, frozenset(ann_set)))
            coder_id += 1
        i += 1
        coder_id = 0
    return ret_dataset


# This function calculates and prints the Krippendorff's alpha value for the
# previous unfiltered dataset and the newly filtered dataset to validate that
# filtering the dataset improved the inter-annotator agreement
def _validation(prev_dataset, new_dataset):
    formatted_annts_new = _nltk_annotation_formatting(new_dataset)
    formatted_annts_prev = _nltk_annotation_formatting(prev_dataset)

    task_prev = AnnotationTask(data=formatted_annts_prev, distance=binary_distance)
    task_new = AnnotationTask(data=formatted_annts_new, distance=binary_distance)

    return task_prev.alpha(), task_new.alpha()


# To avoid having to regenerate label_setsOLD.csv every time I want to filter a D-ESX-X sub-dataset,
# just start with the full label_setsOLD.csv and filter out reflections in it that aren't in
# full_dataset.csv
def match_to_full_dataset(full_dataset, label_sets):
    text_col = list(full_dataset[0]).index("text")
    full_dataset = [row[text_col] for row in full_dataset[1:]]
    all_refs = [row[0] for row in label_sets]

    # dataset_refs = pd.read_csv("full_dataset.csv")["text"].to_numpy().tolist()
    # all_refs = pd.DataFrame(label_sets)[0].to_numpy().tolist()

    # would just filter a dataframe but that encounters a tricky bug
    # where reflections with exactly the same text are wrongly included
    ret_dataset = []
    first_ref = full_dataset[0]
    # https://stackoverflow.com/questions/6294179/how-to-find-all-occurrences-of-an-element-in-a-list
    i = 0
    while len(ret_dataset) < len(full_dataset):
        idx = all_refs.index(first_ref)
        while i < len(full_dataset) and full_dataset[i] == all_refs[idx]:
            ret_dataset.append(label_sets[idx])
            idx += 1
            i += 1
        if i < len(full_dataset):
            first_ref = full_dataset[i]

    assert len(full_dataset) == len(ret_dataset), f"{len(full_dataset)}, {len(ret_dataset)}"
    assert full_dataset == [row[0] for row in ret_dataset], "Datasets contain differing reflections"

    return ret_dataset


# Calculates the agreement for each reflection based on label_setsOLD.csv (see below) and filter out reflections
# from a provided dataset with
def _filter_dataset(masi_agreement_threshold, single_label, full_dataset, label_sets):
    # Instructions: place a file called "full_dataset.csv" containing all of your reflections and a file
    # called "label_setsOLD.csv" containing all of your reflections with a list of label sets into the same
    # directory as main.py before running main
    # see my MLCompare journal for examples of full_dataset.csv and label_setsOLD.csv or email me for help
    # if you can't run the code at amorga94@charlotte.edu
    # ***Use my dataset generation code under Dataset Construction to create a full_dataset.csv
    # and label_setsOLD.csv for any labels you wish
    # Last, alter threshold to change the agreement threshold for inclusion
    # in the final dataset.

    # dist_to_ref will map agreement values to lists of reflections, i.e. every reflection
    # with a 1.0 (perfect) agreement is placed into the 1.0 agreement list bucket in the dictionary
    dist_to_ref = {}

    label_sets = match_to_full_dataset(full_dataset=full_dataset, label_sets=label_sets)

    unfiltered_label_sets = copy.deepcopy(label_sets)  # copy of the original dataset is saved for later validation

    # filter out reflections that (based on the consensus length) don't have only one label
    # this is useful and necessary for running experiments with FastFit
    if single_label:
        label_sets = _single_label_filter(label_sets)

    for elem in label_sets:  # elem is a tuple containing the reflection-labelset pair
        labels = elem[1]  # TODO check that this still works

        # calculating reflection agreement by taking the averaged masi distance across
        # all possible unique subsets of the labels for every reflection
        dist = 0
        all_combinations = list(combinations([i for i in range(0, len(labels))], 2))
        if not all_combinations:
            continue
        for combin in all_combinations:
            dist += _masi_distance(set(labels[combin[0]]), set(labels[combin[1]]))
        dist = dist / len(all_combinations)

        # print(f"{dist} for label set: {labels}")

        # closed addressing collision handling is just easier to work with
        if dist not in dist_to_ref.keys():
            dist_to_ref.update({dist: [elem[0]]})
        else:
            dist_to_ref[dist].append(elem[0])

    dists = list(dist_to_ref.keys())
    dists.sort()
    # filter out distances less than the threshold
    dists = [dist for dist in dists if dist >= masi_agreement_threshold]

    desired_reflections = []
    for d in dists:
        desired_reflections.extend(dist_to_ref[d])

    # write the newly filtered dataset based on desired_reflections
    low_disagreement_dataset = []
    labels = [lb for lb in full_dataset[0][:-1]]  # every label column except "text" for single label dataset; only used when the starting dataset is multi-label
    if single_label:
        low_disagreement_dataset.append(["text", "label"])  # write header row
        for row in full_dataset[1:]:
            # Check which column "text" is in -- if we are going from a multi-label to single-label
            # dataset, the text column will be last, otherwise it will be first
            text_col = list(full_dataset[0]).index("text")

            # Case: Multi-label --> single-label
            # Convert the multi-label formatted entries to single-label ones
            if text_col != 0:
                if row[-1] in desired_reflections:
                    # search for the index of the "1" in the multi-label dataset and
                    # find the corresponding label in the labels array using that index
                    i = 0
                    while row[i] == 0:
                        i += 1
                    # row[-1] is the reflection text and labels[i] is the label
                    low_disagreement_dataset.append([row[-1], labels[i]])
            # Case: Single-label --> single-label
            # Just append the existing entry if is not being filtered out
            else:
                if row[0] in desired_reflections:
                    low_disagreement_dataset.append(row)
    # Case: Multi-label --> multi-label
    # Same in structure as single-label --> single-label
    else:
        low_disagreement_dataset.append(full_dataset[0])  # write header row
        for row in full_dataset[1:]:
            if row[-1] in desired_reflections or row[0] in desired_reflections:
                low_disagreement_dataset.append(row)

    filtered_label_sets = []
    # filter label_sets as well to run validation (below)
    for row in unfiltered_label_sets:
        if row[0] in desired_reflections:
            filtered_label_sets.append(row)  # filtered dataset is the same as low_disagreement_dataset,
            # just formatted the same way as label_sets and always multi-label.
            # this is necessary because we need access to every set of labels from every annotator.
            # filtered_dataset is multi-label regardless of whether single_label is true because
            # it wouldn't make sense to compare the krippendorff of a single-label dataset to a multi-label one
            # and the starting label_sets is always multi-label (and we are comparing against the starting dataset)

    alpha_prev, alpha_new = _validation(unfiltered_label_sets, filtered_label_sets)

    return low_disagreement_dataset, alpha_new


def filter_dataset(kalpha_agreement_threshold, single_label, full_dataset, label_sets):
    """
        Filter a dataset (full_dataset, an iterable) either by an annotator agreement threshold or to only contain single-label reflections.

        Individual reflection annotator agreement is calculated using the set distance metric MASI, averaged over all possible combinations of annotator label set pairs.
        Krippendorff's alpha for the entire dataset is calculated and returned in the console after dataset filtering.

        A reflection is considered "single-label" if the average length of the annotators' label sets for that reflection is 1 (one label is, on average, attached to the reflection).
        single_label=True filters out all reflections not considered single-label.

        :param kalpha_agreement_threshold: [0.0, 1.0], transform the dataset to only contain reflections above that value. this value refers to Krippendorff's alpha agreement
        :param single_label: boolean, whether to transform the dataset to be single-label
        :param full_dataset: iterable that represents the dataset to be transformed. Reflections contained in full_dataset must be a subset of param label_sets.
        :param label_sets iterable of reflections paired with annotation sets.
        :return:
    """
    singlelabel_dataset_filtered, dataset_kalpha = _filter_dataset(masi_agreement_threshold=0,
                                                                     single_label=single_label,
                                                                     full_dataset=full_dataset,
                                                                     label_sets=label_sets)

    # Zero in on a dataset with a high Krippendorff's alpha
    epsilon = 0.01
    while dataset_kalpha < kalpha_agreement_threshold:
        singlelabel_dataset_filtered, dataset_kalpha = _filter_dataset(masi_agreement_threshold=epsilon,
                                                                         single_label=single_label,
                                                                         full_dataset=singlelabel_dataset_filtered,
                                                                         label_sets=label_sets)
        epsilon += 0.01

    print(f"Krippendorff's alpha of new dataset: {dataset_kalpha} with dataset length: {len(singlelabel_dataset_filtered)}")

    return singlelabel_dataset_filtered
"""
@inproceedings{passonneau-2006-measuring,
    title = "Measuring Agreement on Set-valued Items ({MASI}) for Semantic and Pragmatic Annotation",
    author = "Passonneau, Rebecca",
    editor = "Calzolari, Nicoletta  and
      Choukri, Khalid  and
      Gangemi, Aldo  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Odijk, Jan  and
      Tapias, Daniel",
    booktitle = "Proceedings of the Fifth International Conference on Language Resources and Evaluation ({LREC}{'}06)",
    month = may,
    year = "2006",
    address = "Genoa, Italy",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2006/pdf/636_pdf.pdf",
    abstract = "Annotation projects dealing with complex semantic or pragmatic phenomena face the dilemma of creating annotation schemes that oversimplify the phenomena, or that capture distinctions conventional reliability metrics cannot measure adequately. The solution to the dilemma is to develop metrics that quantify the decisions that annotators are asked to make. This paper discusses MASI, distance metric for comparing sets, and illustrates its use in quantifying the reliability of a specific dataset. Annotations of Summary Content Units (SCUs) generate models referred to as pyramids which can be used to evaluate unseen human summaries or machine summaries. The paper presents reliability results for five pairs of pyramids created for document sets from the 2003 Document Understanding Conference (DUC). The annotators worked independently of each other. Differences between application of MASI to pyramid annotation and its previous application to co-reference annotation are discussed. In addition, it is argued that a paradigmatic reliability study should relate measures of inter-annotator agreement to independent assessments, such as significance tests of the annotated variables with respect to other phenomena. In effect, what counts as sufficiently reliable intera-annotator agreement depends on the use the annotated data will be put to.",
}
also cite nltk
"""