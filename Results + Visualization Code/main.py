import matplotlib.pyplot as plt
import csv
from pathlib import Path


def read_csv(file, averages):
    with open(file, "r", encoding="utf-8") as f:
        c_r = csv.reader(f)
        for row in c_r:
            if row:
                if row[0] not in averages.keys():
                    averages.update({row[0]: [row[1]]})
                else:
                    averages[row[0]].append(row[1])
    return averages


def resolve_list(averages):
    for label in averages.keys():
        if label != "accuracy":
            averages[label] = [int(num) for num in averages[label]]
            averages[label] = round(sum(averages[label]) / len(averages[label]))
        else:
            averages[label] = [float(num) for num in averages[label]]
            averages[label] = float(sum(averages[label]) / len(averages[label]))

    return averages


def main():
    gpt_data = Path("gpt_data").glob("*")
    setfit_data = Path("setfit-data").glob("*")

    gpt_averages = {}
    setfit_averages = {}
    for file in gpt_data:
        gpt_averages = read_csv(file, gpt_averages)
    for file in setfit_data:
        setfit_averages = read_csv(file, setfit_averages)

    gpt_averages = resolve_list(gpt_averages)
    setfit_averages = resolve_list(setfit_averages)
    names = ["tn", "fp", "fn", "tp"]
    curr = gpt_averages
    print(gpt_averages)

    plt.subplot(151)
    bars = plt.bar(names, list(curr.values())[:4])
    plt.ylim(0, 120)
    plt.gca().set_title("Python and Coding\n(34 total occurrences / 150)")
    plt.gca().bar_label(bars)
    plt.subplot(152)
    bars = plt.bar(names, list(curr.values())[4:8])
    plt.ylim(0, 120)
    plt.gca().set_title("GitHub\n(43 total occurrences / 150)")
    plt.gca().bar_label(bars)
    plt.subplot(153)
    bars = plt.bar(names, list(curr.values())[8:12])
    plt.ylim(0, 120)
    plt.gca().set_title("Assignments\n(36 total occurrences / 150)")
    plt.gca().bar_label(bars)
    plt.subplot(154)
    bars = plt.bar(names, list(curr.values())[12:16])
    plt.ylim(0, 120)
    plt.gca().set_title("Time Management and Motivation\n(43 total occurrences / 150)")
    plt.gca().bar_label(bars)
    plt.subplot(155)
    plt.pie([float(list(curr.values())[16]), 1.0-float(list(curr.values())[16])], labels=["Correct", "Incorrect"], colors=["g", "r"], autopct='%1.1f%%')
    plt.gca().set_title("Overall Accuracy")
    plt.suptitle("Optimal GPT-4o results on test dataset with 150 reflections (averages across five trials)")
    plt.show()


if __name__ == '__main__':
    main()

