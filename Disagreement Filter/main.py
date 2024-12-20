import csv
from itertools import combinations

label_encodings = {
    "None": 0,
    "Python and Coding": 1,
    "Github": 2,
    "MySQL": 3,
    "Assignments ": 4,
    "Quizzes ": 5,
    "Understanding requirements and instructions ": 6,
    "Learning New Material": 7,
    "Course Structure and Materials": 8,
    "Time Management and Motivation": 9,
    "Group Work": 10,
    "IDE and Package Installation": 11,
    "API": 12,
    "FastAPI": 12,
    "Personal Issue": 13,
    "Other": 14
}


# Fixed version of NLTK's masi_distance()
# MASI = jaccard_distance * m, the original NLTK implementation
# had (1-jaccard) * m (jaccard = cardinality(intersection)/cardinality(union))
# also per the original MASI paper:
# "If two sets Q and P are identical, M is 1. If one
# set is a subset of the other, M is 2/3. If the intersection
# and the two set differences are all non-null, then M is 1/3.
# If the sets are disjoint, M is 0." (Passonneau, 2006)
# http://www.lrec-conf.org/proceedings/lrec2006/pdf/636_pdf.pdf
def masi_distance(label1, label2):
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


# Calculates the agreement for each reflection based on label_sets.csv (see below) and filter out reflections
# from a provided dataset with 
def main():
    # Instructions: place a file called "full_dataset.csv" containing all of your reflections and a file
    # called "label_sets.csv" containing all of your reflections with a list of label sets into the same
    # directory as main.py before running main
    # see my MLCompare journal for examples of full_dataset.csv and label_sets.csv or email me for help
    # if you can't run the code at amorga94@charlotte.edu
    # ***Use my dataset generation code under Dataset Construction to create a full_dataset.csv
    # and label_sets.csv for any labels you wish
    # Last, alter top_n_agreement_categories_to_include to change the agreement threshold for inclusion
    # in the final dataset. TODO - change this from top categories to a float agreement threshold

    top_n_agreement_level_categories_to_include = 1

    # TODO - See if I can just calculate MASI distance on sets of labels strings rather than encoded integers to simplify things
    dist_to_ref = {}
    with open("label_sets.csv", "r", encoding="utf-8") as ls:
        c_r = csv.reader(ls)
        for elem in list(c_r):  # elem is tuple of the reflection-label set pair
            # Encode string labels back to encoded integers
            labels_enc = []
            for label_list in eval(elem[1]):  # label_sets.csv stored the lists as strings, convert
                # to python lists using eval()
                labels_enc.append([])
                for label in label_list:
                    if label in list(label_encodings.keys()):
                        labels_enc[len(labels_enc)-1].append(label_encodings[label])
                    else:
                        labels_enc[len(labels_enc)-1].append(14)

            # calculating reflection agreement by taking the averaged masi distance across
            # all possible unique subsets of the labels for some reflection
            dist = 0
            all_combinations = list(combinations([i for i in range(0, len(labels_enc))], 2))
            for combin in all_combinations:
                dist += masi_distance(set(labels_enc[combin[0]]), set(labels_enc[combin[1]]))
            dist = dist / len(all_combinations)

            print(f"{dist} for label set: {labels_enc}")

            # closed addressing collision handling is just easier to work with
            if dist not in dist_to_ref.keys():
                dist_to_ref.update({dist: [elem[0]]})
            else:
                dist_to_ref[dist].append(elem[0])

    dists = list(dist_to_ref.keys())
    dists.sort()

    for item in dist_to_ref.items():
        print(item)
        
    top_n_agreement_level_categories_to_include *= -1
    top_ref_dists = [dist for dist in dists[:top_n_agreement_level_categories_to_include]]
    desired_reflections = []
    for d in top_ref_dists:
        desired_reflections.extend(dist_to_ref[d])
    print(top_ref_dists)

    with open("low_disagreement_dataset.csv", "w", encoding="utf-8", newline="") as low_d:
        c_w = csv.writer(low_d)
        with open("full_dataset.csv", "r", encoding="utf-8") as full_d:
            c_r = list(csv.reader(full_d))
            c_w.writerow(c_r[0])  # write header row
            for row in c_r[1:]:
                if row[4] in desired_reflections:
                    c_w.writerow(row)


if __name__ == "__main__":
    main()

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
