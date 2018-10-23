import numpy as np
import pandas as pd
from copy import deepcopy
from queue import PriorityQueue
import os

def vector_w(file, special = False, names_sf = None, p = None):
    '''
    Read a text file line by line. Then, builts two dictionaries.
    The first is a dictionary with all the features. The second is a
    dictionary with features by line.

    Input:
        file = string
        special = boolean (True if using special features)
        names_sf = list of string (names of special features)
        p = dictionary (parameter for the model)
    Output:
        dic_words = features by line
        dic_gen = all features
    '''
    dic_words = {}
    dic_gen = {}

    if special:
        # Source positive and negative: https://github.com/williamgunn/SciSentiment
        positive = set(pd.read_csv("data/positive-words.txt", header = None)[0])
        negative = set(pd.read_csv("data/negative-words.txt", header = None)[0])
        neutral_ref = positive.union(negative)
        # Source contrast: https://writing.wisc.edu/Handbook/Transitions.html
        contrast = ("yet nevertheless nonetheless but however"
            "though otherwise notwithstanding").split(" ")

    with open(file, "r") as ins:
        for num_line, line in enumerate(ins):
            content = line.strip().split()
            if special:
                control_contrast = 3*[0]
                counts_sentiment = 3*[0]

            for j, word in enumerate(content[:-1]):
                dic_gen[(int(content[-1]), word)] = 0
                if j == 0:
                    dic_words[num_line] = {'y': int(content[-1]),
                        'words': [word]}
                else:
                    dic_words[num_line]['words'].append(word)
                # Special features
                if special:
                    control_contrast = update_contrast_feature(word,
                        control_contrast, negative, contrast, positive)
                    counts_sentiment = update_count_feature(word,
                        counts_sentiment, negative, positive, neutral_ref)

            if special:
                # Feature 1: contrast
                if names_sf[0] != None:
                    if ((sum(control_contrast) >= p[1]["nw"]) &
                            (int(content[-1]) == p[1]["fn"])):
                        dic_words[num_line]['words'].append(names_sf[0])
                        dic_gen[(int(content[-1]), names_sf[0])] = 0

                proportion = counts_sentiment[2] / (len(content) - 1)
                # Feature 2: Neutral words
                if names_sf[1] != None:
                    if (proportion > p[2]["prop"]) & (
                        int(content[-1]) == p[2]["fn"]):
                        dic_words[num_line]['words'].append(names_sf[1])
                        dic_gen[(int(content[-1]), names_sf[1])] = 0

                # Feature 3: Proportion of Positve words
                if names_sf[2] != None:
                    positiveness = counts_sentiment[1] / (len(content) - 1)
                    if (proportion > p[3]["prop"]) & (
                        int(content[-1]) == p[3]["fn"]):
                        dic_words[num_line]['words'].append(names_sf[2])
                        dic_gen[(int(content[-1]), names_sf[2])] = 0

    return dic_words, dic_gen



def update_contrast_feature(word, control_sf, negative, contrast, positive):
    '''
    Updates the control_sf list

    Inputs:
        word = string
        control_sf = list (len 3)
        negative = set of strings (negative words)
        contrast = set of string (contrasting words)
        positive = set of strings (positive words)
    Output:
        control_sf = list (len 3)
    '''
    if word in negative:
        control_sf[0] = 1
    elif word in contrast:
        control_sf[1] = 1
    elif word in positive:
        control_sf[2] = 1

    return control_sf

def update_count_feature(word, counts_sentiment,
        negative, positive, neutral_ref):
    '''
    Updates the counts by sentiment

    Inputs:
        word = string
        counts_sentiment = integer
        negative = set of strings (negative words)
        positive = set of strings (positive words)
        neutral_ref = set of strings (neutral words)
    Output:
        counts_sentiment = integer
    '''

    if word in negative:
        counts_sentiment[0] += 1

    if word in positive:
        counts_sentiment[1] += 1

    if word not in neutral_ref:
        counts_sentiment[2] += 1


    return counts_sentiment


def score(dic_words, dic_gen, y_assumed, num_line):
    '''
    Computes the score of one assumption of "y."

    Inputs:
        dic_words = dictionary
        dic_gen = dictionary
        y_assumed = integer
        num_line = integer
    Output:
        tot_sum = float
    '''
    # x, y are in dic_words
    # w is in dic_gen
    tot_sum = 0.0
    for word in dic_words[num_line]["words"]:
        try:
            tot_sum += dic_gen[(y_assumed, word)]
        except:
            None

    return tot_sum

def classify(dic_words, dic_gen, num_line, categories):
    '''
    Classifies a line of text according to the current weights

    Inputs:
        dic_words = dictionary
        dic_gen = dictionary
        num_line = integer
        categories = integer
    Output:
        classification = integer
    '''
    compiler = []
    for cat_num in range(categories):
        s = score(dic_words, dic_gen, cat_num, num_line)
        compiler.append(s)

    classification = compiler.index(max(compiler))

    return classification


def costClassify(dic_words, dic_gen, num_line, categories):
    '''
    Computes the costClassify fucntion for hinge loss.

    Inputs:
        dic_words = dictionary
        dic_gen = dictionary
        num_line = integer
        categories = integer
    Output:
        cost_classify = integer
    '''
    y_classified = classify(dic_words, dic_gen, num_line, categories)

    compiler = []
    for cat_num in range(categories):
        s = score(dic_words, dic_gen, cat_num, num_line)
        if y_classified != cat_num:
            s += 1
        compiler.append(s)

    cost_classify = compiler.index(max(compiler))

    return cost_classify


def f(dic_gen, line, word, y_feature, y_classified):
    '''
    Gives 1, 0 or -1 to update one weight. Note that this function is two
    fucntions in the write-up (f(x,y) anf f(x, classify(x,w)))

    Inputs:
        dic_gen = dictionary
        line = dictionary
        word = string
        y_feature = integer
        y_classified = integer
    Outputs:
        rv1 - rv2 = Integer E[-1, 1]
    '''

    if word in line["words"]:
        # Part 1: actual y
        if y_feature == line["y"]:
            rv1 = 1
        else:
            rv1 = 0

        # Part 2: classification
        if y_feature == y_classified:
            rv2 = 1
        else:
            rv2 = 0

    return rv1 - rv2


def actualize_weights(d_words, d_gen, num_line, categories, eta, hinge = False):
    '''
    Actualize all weights considering one sentence

    Inputs:
        d_words = dictionary
        d_gen = dictionary
        num_line = integer
        categories = integer
        eta = float
        hinge = boolean
    Output:
        d_gen = dictionary
    '''
    y_classified = classify(d_words, d_gen, num_line, categories)
    if hinge:
        y_classified = costClassify(d_words, d_gen, num_line, categories)

    for word in d_words[num_line]["words"]:
        for cat_num in range(categories):
            add = eta * f(d_gen, d_words[num_line],
                word, cat_num, y_classified)
            try:
                d_gen[(cat_num, word)] += add

            except:
                None

    return d_gen


def train_perceptron(file_train, file_dev, file_dev_test, num_categories,
        n_epochs, eta, hinge = False, special = False, names_sf = None,
        p = None):
    '''
    Trains the model, prints accuracy each 20000 lines, and returns a dictionary
    that contains 1) the accuracy in dev, 2) accuracy in dev_test,
    3) weights, 4) lines correctly classified, 5) lines uncorrectly classified

    Inputs:
        file_train = string (location of the training data)
        file_dev = string (location of the dev data)
        file_dev_test = string (location of the dev test data)
        num_categories = integer (number of sentiments)
        n_epochs = integer
        eta = float
        hinge = boolean
        special = boolean
        names_sf = list of integers
        p = dictionary (parameters)
    Output:
        results = dictionary (including weights)
    '''
    dic_words, weights = vector_w(file_train, special, names_sf, p)
    dic_words_d, dic_gen_d = vector_w(file_dev, special, names_sf, p)
    dic_words_dt, dic_gen_dt = vector_w(file_dev_test, special, names_sf, p)

    list_accuracy = []
    results = {}
    for epoch in range(n_epochs):
        print(" In EPOCH NUM: ", epoch)
        for i in range(len(dic_words)):
            weights = actualize_weights(dic_words, weights, i,
                num_categories, eta, hinge)
            if i % 50 == 0:
                accuracy_d = accuracy(dic_words_d, weights,
                    num_categories)
                list_accuracy.append(accuracy_d)
                print(" The accuracy in epoch: ", epoch, " in the line: ", i,
                    " is: ", accuracy_d)
                if accuracy_d == max(list_accuracy):
                    accuracy_dt, right, wrong = accuracy(dic_words_dt, weights,
                        num_categories, report_mistakes = True)
                    results["dev"] = accuracy_d
                    results["dev_test"] = accuracy_dt
                    results["weights"] = deepcopy(weights)
                    results["right"] = deepcopy(right)
                    results["wrong"] = deepcopy(wrong)

                    print(" The DEV TEST accuracy in epoch: ", epoch,
                        " in the line: ", i, " is: ", accuracy_dt)

            if i == len(dic_words):
                accuracy_test = accuracy(dic_words_test, weights,
                    num_categories)
                print(" The accuracy at the END of the epoch: ", epoch,
                    " in the line: ", i, " is: ", accuracy_test)

    return results


def accuracy(dic_words, weights, num_categories, report_mistakes = False):
    '''
    Computes the accuracy of the model

    Inputs:
        dic_words = dictionary (keys: num_line, val: {"words": [], "y": int})
        weights = dictionary (keys: (clasification, word); val: weight)
        num_categories = Integer
        report_mistakes = boolean (to output dictionaries with incorrect
                          and correct classifications)
    Output:
        acc = float (accuracy)
        right = dictionary (keys: num_line, val: {"gold": , "classifed": })
        wrong = dictionary (keys: num_line, val: {"gold": , "classifed": })
    '''

    right = {}
    wrong =  {}
    num_well_classified = 0
    for i, val in dic_words.items():
        classification = classify(dic_words, weights, i, num_categories)
        if classification == val['y']:
            num_well_classified += 1
            right[i] = {'gold': val['y'], 'classified': classification}
        else:
            wrong[i] = {'gold': val['y'], 'classified': classification}

    acc = num_well_classified / len(dic_words)

    if report_mistakes:
        return acc, right, wrong
    else:
        return acc


def remove_file_if_exists(name_file):
    '''
    Deletes a file if it exits.

    Input:
        name_file = string (location)
    '''
    try:
        os.remove(name_file)
    except OSError:
        pass


def weigths_analysis(weights, num_categories, name_file = None, total_ex = 10):
    '''
    Prints the most important words by category

    Inputs:
        weights = dictionary
        num_categories = integer
        name_file = string (loication to save)
    Outputs:
        None. It only prints
        Save one file (name_file)
    '''
    if name_file:
        remove_file_if_exists(name_file)

    dic_queue = {}
    for i in range(num_categories):
        dic_queue["q_{}".format(i)] = PriorityQueue()

    for k,v in weights.items():
        y, word = k
        dic_queue["q_{}".format(y)].put((-v, word))

    for cat in range(num_categories):
        print_ = "####### Examples of weights, y = {} ########".format(cat)
        if name_file:
            save_line(name_file, print_)
        print(print_)
        for i in range(total_ex):
            if dic_queue["q_{}".format(cat)].empty():
                None
            else:
                extracted = dic_queue["q_{}".format(cat)].get()
                if name_file:
                    save_line(name_file, str(extracted))
                print(extracted)

    return print(" Top words enlisted. ")

def save_line(name_file, string):
    '''
    Saves a line in file.

    Inputs:
        name_file = string (location of file)
        string = string (one line to save in the file)
    '''
    f = open(name_file, 'a')
    f.write(string + "\n")
    f.close()


def run_cases(file_train, file_dev, file_dev_test, num_categories,
        n_epochs, eta, hinge, special, names_sf, testing_cases = None,
        y_targets = [0, 1, 2]):
    '''
    Runs different combinations of parameters that will help to calibrate
    the model

    Inputs:
        file_train = string (location of the training data)
        file_dev = string (location of the dev data)
        file_dev_test = string (location of the dev test data)
        num_categories = integer (number of sentiments)
        n_epochs = integer
        eta = float
        hinge = boolean
        special = boolean
        names_sf = list of integers
        testing_cases = list of floats (proportions to test)
        y_targets = list of integers (integers for the features; for instance,
            y[0] is the "y" for the feature number 1) additionally to the
            conditions specified, like "have positive, negative,
            and neutral words"
    Output:
        compilation_results = dictionary (keys: name_model, values: results)
        lc_df = df (last compilation df (summary of models))
        key_winner = string (key of the dic compilation_results that give
            the best performance)
    '''
    if not testing_cases:
        testing_cases = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    number_params_test = len([x for x in names_sf if x != None])
    compilation_results = {}

    if number_params_test <= 0:
        results = train_perceptron(file_train, file_dev, file_dev_test,
            num_categories,n_epochs, eta, hinge, special, names_sf)
        compilation_results["unique"] = results

    elif (number_params_test == 1) & (names_sf[0] != None):
        p = {1: {"nw": 3, "fn": y_targets[0]}}
        results = train_perceptron(file_train, file_dev, file_dev_test,
            num_categories, n_epochs, eta, hinge, special, names_sf, p)
        compilation_results["unique"] = results

    elif (number_params_test == 1) & (names_sf[0] == None):
        for number in testing_cases:
            p = {1: {"nw": 3, "fn": y_targets[0]},
                2: {"prop": number, "fn": y_targets[1]},
                3: {"prop": number, "fn": y_targets[2]}}
            results = train_perceptron(file_train, file_dev, file_dev_test,
                num_categories, n_epochs, eta, hinge, special, names_sf, p)

            compilation_results[number] = results

    elif (number_params_test == 2) & (names_sf[0] != None):
        for number in testing_cases:
            p = {1: {"nw": 3, "fn": y_targets[0]},
                2: {"prop": number, "fn": y_targets[1]},
                3: {"prop": number, "fn": y_targets[2]}}
            results = train_perceptron(file_train, file_dev, file_dev_test,
                num_categories, n_epochs, eta, hinge, special, names_sf, p)

            compilation_results[number] = results

    elif number_params_test >= 2:
        for number in testing_cases:
            for number_2 in testing_cases:
                p = {1: {"nw": 3, "fn": y_targets[0]},
                    2: {"prop": number, "fn": y_targets[1]},
                    3: {"prop": number_2, "fn": y_targets[2]}}
                results = train_perceptron(file_train, file_dev, file_dev_test,
                    num_categories, n_epochs, eta, hinge, special, names_sf, p)

                compilation_results[(number, number_2)] = results

    last_compilation = {}
    for k, v in compilation_results.items():
        if (number_params_test <= 1) or ((number_params_test == 2)
                & (names_sf[1] != None)):
            last_compilation[k] = v["dev_test"]
        else:
            last_compilation[k] = (k[0], k[1], v["dev_test"])

    lc_df = pd.DataFrame.from_dict(last_compilation, orient = "index")
    num_cols_lc_df = len(lc_df.columns)
    lc_df.sort_values(num_cols_lc_df-1, ascending = False, inplace = True)

    key_winner = lc_df.index[0]

    return compilation_results, lc_df, key_winner


def error_analysis_fun(results, file_dev_test):
    '''
    Creates several objects usable to analyze the sentences that was well and
    bad classified.

    Inputs:
        results = dictionary (including weights)
        file_dev_test = string (location of the dev test data)
    Outputs:
        wrong_df = DataFrame (num_line, gold, classified)
        right_df = DataFrame (num_line, gold, classified)
        complete_sentences = DataFrame (sentence, gold)
        main_summary = DataFrame (summary of complete_sentences)
        error_analysis = DataFrame (sentence, gold, classified), only wrong
        error_analysis_counts = DataFrame (misclassified by gold standard)
        success_analysis = DataFrame (sentence, gold, classified), only right
        success_analysis_counts = DataFrame (correct classified by gold standard)
    '''
    wrong_df = pd.DataFrame.from_dict(results["wrong"], orient='index')
    right_df = pd.DataFrame.from_dict(results["right"], orient='index')
    complete_sentences = pd.read_csv(file_dev_test,
        sep="\t", names = ["sentence", "gold"])
    main_summary = complete_sentences.groupby("gold").count()

    error_analysis = pd.merge(complete_sentences, wrong_df,
        left_index=True, right_index=True).drop("gold_y", axis = 1).rename(
        columns = {"gold_x": "gold"})
    error_analysis_counts =  error_analysis.groupby("gold").count()

    success_analysis = pd.merge(complete_sentences, right_df,
        left_index=True, right_index=True).drop("gold_y", axis = 1).rename(
        columns = {"gold_x": "gold"})
    success_analysis_counts =  success_analysis.groupby("gold").count()

    return (wrong_df, right_df, complete_sentences, main_summary, error_analysis,
        error_analysis_counts, success_analysis, success_analysis_counts)


def print_accuracy(compilation, file_save = None, key = "unique",
        model = "Missing Val"):
    '''
    Prints and saves the accuracy of the models.

    Inputs:
        compilation = dictionary (compilation of models)
        file_save = string (location file to save accuracy)
        key = string (key for the best model in compilation)
        model = string (name of the model to save in the file)
    Output:
        compilation[key]["dev"] = float (accuracy in dev data)
        compilation[key]["dev_test"] = float (accuracy in dev_test data)
        * Saves in the file file_save
    '''
    dev_string = "The accuracy in the DEV set is: " + str(
        compilation[key]["dev"])
    dev_test_string = "The accuracy in the DEV_TEST is: " + str(
        compilation[key]["dev_test"])
    print(dev_string)
    print(dev_test_string)

    if file_save:
        save_line(file_save, " ############# " + model + " ############# ")
        save_line(file_save, dev_string)
        save_line(file_save, dev_test_string)

    return compilation[key]["dev"], compilation[key]["dev_test"]
