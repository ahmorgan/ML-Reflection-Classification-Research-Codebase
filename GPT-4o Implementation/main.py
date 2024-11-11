from openai import OpenAI
import csv
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

# using the OpenAI API to prompt GPT-x models for multi-label classification of data

print("Setting up data / GPT-4...")
client = OpenAI()

# list of responses that will be given to the LLM
response_prompts = []

questions = [
    "How do you feel about the course so far?",
    "Explain why you selected the above choice(s).",
    "What was your biggest challenge(s) for these past modules?",
    "How did you overcome this challenge(s)? Or what steps did you start taking towards overcoming it?",
    "Do you have any current challenges in the course? If so, what are they?"
]

# all labels for reference:
"""
    "None",
    "Python and Coding",
    "Github",
    "MySQL",
    "Assignments",
    "Quizzes",
    "Understanding requirements and instructions",
    "Learning New Material",
    "Course Structure and Materials",
    "Time Management and Motivation",
    "Group Work",
    "API",
    "Project"
"""

labels = [
    "Python and Coding",
    "Github",
    "Assignments",
    "Time Management and Motivation",
]


def prompt_model(refs, num_preds, temperature=None):
    i = 0
    llm_classif = []
    t = temperature if temperature else 1
    for response in refs:
        completion = client.chat.completions.create(
            model="chatgpt-4o-latest",
            # proompt engineering
            messages=[
                {"role": "system", "content": "You are a software engineering professor who has just received "
                                              "feedback responses from your students regarding their issues "
                                              "and/or experiences with your class. You seek to help them with"
                                              "their issues and ensure their success in your class."},
                {"role": "user",
                 "content": f"Regarding the following student feedback response enclosed in quotations: "
                            f""
                            f"'{response}'"
                            f""
                            f"Choose or one more label(s) from the following list that best represents"
                            f"the issue(s) faced by the student. Respond only with your chosen labels enclosed"
                            f"in brackets."
                            f""
                            f"{labels}"}
            ],
            temperature=t
        )
        print(t)
        classif = completion.choices[0].message.content
        print(classif)
        llm_classif.append(classif)
        i += 1
        # Only generating the first ten LLM responses to save time (and a few pennies in API calls).
        if i == num_preds:
            break
    return llm_classif


def trial(llm_classifications):
    print("Encoding classifications...")
    # encode gpt responses to create a confusion matrix out of them
    gpt_preds_enc = []
    for classification in llm_classifications:
        response = [0 for _ in range(0, len(labels))]
        i = 0
        for label in labels:
            # gpt output is not always formatted correctly, but
            # always contains the issue classification as a substring.
            # therefore, search for substring of label in the output
            # to make classifications
            if label in classification:
                print("found")
                response[i] = 1
            i += 1
        print(response)
        gpt_preds_enc.append(response)

    with open("raw_gpt_preds.csv", 'w', encoding="utf-8", newline='') as rgp:
        c_w = csv.writer(rgp)
        c_w.writerows(gpt_preds_enc)

    # data_preprocessor.process_data(file_name="intermediate_preds.csv")
    true_preds_enc = []

    # encode the test dataset in the same way as the classifications
    with open("gpt_test.csv", "r") as gpt_test:
        csv_r = csv.reader(gpt_test)
        for row in csv_r:
            # int cast is important because the csv reader casts the int to a string by default
            true_preds_enc.append([int(pred) for pred in row])
    # remove now unneeded intermediate file
    # os.remove("intermediate_preds.csv")

    true = np.array(true_preds_enc)  # size num of reflections in dataset
    pred = np.array(gpt_preds_enc)  # size num_predictions

    # only first num_predictions predictions (true is every true prediction by default)
    true = true[:150]

    print(f"True values:\n{true}")
    print(f"Predictions:\n{pred}")

    print("Running metrics...")
    # generate confusion matrices and extract true positive, false negative,
    # true negative, and false negative counts for each label
    confusion_matrices = multilabel_confusion_matrix(true, pred)
    print(confusion_matrices)
    trial_result = {}
    x = 0

    for matrix in confusion_matrices:
        # flatten confusion matrix to list
        matrix = matrix.ravel()
        print(matrix)
        # populate results with information from the label's confusion matrix
        trial_result.update({f"{labels[x]}-tn": matrix[0].item()})
        trial_result.update({f"{labels[x]}-fp": matrix[1].item()})
        trial_result.update({f"{labels[x]}-fn": matrix[2].item()})
        trial_result.update({f"{labels[x]}-tp": matrix[3].item()})
        x += 1
        if x >= len(labels):
            break
    accuracy = 0.0
    for label in labels:
        # 100 is the number of reflections used in evaluation
        # acc = num_of_correct_classifications / 100
        accuracy += (trial_result[f"{label}-tp"] + trial_result[f"{label}-tn"]) / 150
    accuracy /= len(labels)
    trial_result.update({"accuracy": accuracy})

    print("Results for trial:")
    print(trial_result)
    return trial_result


def main():
    # minor data preprocessing
    # iterate through reflections, concatenate each question with each student sub-response into a string that represents the full reflection
    with open("gpt_reflections.csv", "r", encoding="utf-8") as gpt:
        c_r = csv.reader(gpt)
        for row in c_r:
            full_student_response = ""
            i = 0
            for value in row:
                full_student_response += f"{questions[i]}: {value} "
                i += 1
            response_prompts.append(full_student_response)

    print("GPT-4 making classifications...")

    # will be of shape {temperature: resulting_metrics)
    hp_search = {}
    # for j in range(1, 11):
    classifications = prompt_model(response_prompts, num_preds=150, temperature=0.5)
    hp_search.update({f"{0.5}": trial(classifications)})

    result = {}
    max_acc = 0.0
    optimal_temperature = -1
    for entry in hp_search.items():
        if entry[1]["accuracy"] > max_acc:
            max_acc = entry[1]["accuracy"]
            result = entry[1]
            optimal_temperature = entry[0]
    print(f"Optimal temperature is {optimal_temperature}")

    # write unprocessed classifications to a csv file (optional)
    """
    with open("unprocessed_predictions.csv", "w") as output:
        i = 0
        for c in llm_classifications:
            output.write(c)
            if i != len(llm_classifications)-1:
                output.write(",\n")
            i += 1
    """

    with open("metrics.csv", "w") as m:
        c_w = csv.writer(m)
        for entry in result.items():
            arr = [entry[0], entry[1]]
            c_w.writerow(arr)

    print("Results written to metrics.csv")


if __name__ == "__main__":
    main()
