import json
import os
import random
import csv
import numpy as np
from collections import Counter
from sklearn import tree
from sklearn import metrics
import re
from gtts import gTTS
import pickle
import time
import pandas as pd

TO_TRAIN = False  # whether a new decision tree is fitted and saved
TO_LOAD = True   # whether an already fitted decision tree is loaded
LOG = True
LOG_FILE = "logfile.txt"
f = open(LOG_FILE, "a")

SLEEP_TIME = 1

SURVEY_QUESTIONS = ("I am satisfied with this system", "I would recommend this system", "This system is fun to use")
SURVEY_SCALE = 5

# properties
USER_MODE = True    # whether the program start displaying the dialog (if False it start showing some options)
ASK_CONFIRMATION = False    # whether confirmation is asked after every preference expressed
MIN_PREF = 2    # the minimum number of preferences to express before a restaurant is suggested
RANDOM_ORDER = True     # whether the preferences can be expressed in a random order
CHANGE_POSSIBLE = True  # whether it is possible to modify the preferences
RE_ASK_INEFFECTIVE_PREFERENCES = True  # whether the user is asked to repeat when an expressed preference has no effects
ASK_LEVENSHTEIN_CORRECTNESS = True  # whether confirmation is asked after lev. is used to predict a mistake
LEVENSHTEIN_LIMIT = 3     # the maximum levenshtein distance accepted
MAX_LEVENSHTEIN_RATIO = 0.3     # the maximum ratio distance/length accepted
LEVENSHTEIN_PREPROCESSING = True    # whether the system checks for typos as soon as the input is received
NORMALIZATION_PREPROCESSING = True  # whether '?', '!' and apostrophes are removed before processing the input
LIMIT_UTTERANCES = False    # whether the dialog terminates once the max number of utterances is reached
MAX_UTTERANCES_NUMBER = 10  # the maximum length of the dialog
BASELINE_CLASSIFIER = False     # whether the rule-based baseline is used instead of the machine learning classifier
RESTART_ALLOWED = True      # whether the user is given the possibility to restart the dialog
ALL_CAPS = False    # whether the system output is capitalized
RANDOM_SUGG = True  # whether the suggestions are given in a random order
SINGLE_PREF = False # whether a single preference per utterance is accepted
INPUT_TO_LOWER_CASE = True  # whether the input from the user is converted to lowercase
# whether the input is converted to lowercase during the identification phase
# Necessary to test the uppercase CSV lookup without affecting the classification performance
LOWER_CASE_ACT_IDENTIFICATION = False
TEXT_TO_SPEECH = False      # whether text to speech is active for system utterances
PRINT_RESPONSE = True       # whether system utterances are printed to console


# files and folders
DATA_FOLDER_1 = "dstc2_traindev"
DATA_FOLDER_2 = "dstc2_test"
DATA_SUBFOLDER = "data"
SYSTEM_JSON = 'log.json'
USER_JSON = 'label.json'
ONTOLOGY_JSON = 'ontology_dstc2.json'
UTT_ACTS_TEXT_FILE = "dialog_acts.txt"
TURNS_ATTRIBUTE = "turns"
RESTAURANT_INFO_FILE = "restaurantinfo.csv"

# dialog acts
ACTS = ("ack", "affirm", "bye", "confirm", "deny", "hello", "inform", "negate",
        "null", "repeat", "reqalts", "reqmore", "request", "restart", "thankyou")

# rule based baseline keywords
KEYWORDS = (("thatll", "good", "fine"), ("yes", "yeah", "yea", "right"), ("good", "bye", "goodbye"),
            (), "wrong", ("hi", "hello"), ("care", "find", "looking", "matter", "west", "east", "north", "south",
                                           "town", "food", "any", "cheap", "expensive"), ("no", "not"),
            ("unintelligible", "noise", "inaudible"),
            ("back", "again", "repeat",), ("what", "else", "how", "about", "other", "any", "give", "just", "more"),
            "more",
            ("could", "may", "can", "what", "address", "phone", "number", "post", "code", "whats", "type", "food"),
            ("start", "reset"), ("thank", "thanks", "thankyou"))

# difficult instances, used to test the system with critical utterances
DIFFICULT_INSTANCES_NEGATION = (("i dont want a spanish restaurant", "deny"), ("i am not looking for a chinese restaurant", "deny"),
                                ("italian food is not what i am looking for", "deny"), ("no in any area", "negation"),
                                ("i dont care", "inform"), ("i dont want chinese", "deny"), ("i dont want that one whats another one", "reqalts"),
                                ("that's not what i want", "deny"))
DIFFICULT_INSTANCES_NULL = (("i want to ride my bicycle", "null"), ("ffjndjsfn", "null"), ("", "null"), ("i would like djfvjd food", "inform"))
DIFFICULT_INSTANCES_APOSTROPHE = (("i'm looking for chinese", "inform"), ("i don't care", "inform"), ("i don't want chinese", "deny"))

# correct words used in the pre-processing function to identify typos
CORRECT_WORDS = ("phone", "number", "address", "postcode", "restaurant", "location", "more", "about", "repeat")

# words that do not have to be corrected if not in a regex
CRITIC_WORDS = ("would")

# system messages
LOADING_MESSAGE = "\nLoading...\n"
ENTER_OPTIONS = "\nWhat do you want to do?\n\t0)\tget classifiers performance\n\t1)\ttry the rule-based baseline\n" \
          "\t2)\ttry the random baseline\n\t3)\ttry the decision tree\n\t4)\tLet me help you find the perfect" \
          " restaurant!\n\t5)\tTest difficult instances\n\nPress 'q' to quit\n"
ENTER_AVAILABLE_OPTIONS = "\nPlease enter an available option:\n\t0)\tget classifiers performance\n\t" \
                          "1)\ttry the rule-based baseline\n\t2)\ttry the random baseline\n\t" \
                          "3)\ttry the decision tree\n\t4)\tLet me help you find the perfect restaurant!" \
                          "\n\t5)\tTest difficult instances\n\nPress 'q' to quit\n\n\n"

ENTER_NEXT_OPTION = "\nWhat do you want to do next?\n\t0)\tget classifiers performance\n\t1)\ttry the rule-based" \
                    " baseline\n\t2)\ttry the random baseline\n\t3)\ttry the decision tree\n\t4)\tLet me help you" \
                    " find the perfect restaurant!\n\t5)\tTest difficult instances\n\nPress 'q' to quit\n\n\n"

# system dialog utterances
WELCOME_UTT = "Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area," \
              " price range or food type. How may I help you?"
SINGLE_PREF_WARNING = "\nYou can only state a single preference for utterance"
ASK_PREF = ("What kind of food would you like?",
            "Would you like something in the cheap, moderate or expensive price range?",
            "What part of town do you have in mind?")
NO_MORE_SUGG = "There are no other restaurants meeting your criteria"
I_DONT_UNDERSTAND = "Could you please be more clear?"
REPEAT_PREFERENCES = "Can you express your preferences in a different way?"
RESTART_NOT_ALLOWED = "Sorry, you are not allowed to restart the dialog"
END_UTT = "Have a nice day"
ACK_LIST = ("Good", "Perfect", "Ok")
ASK_CONF_SENTENCE_OPENING_LIST = ("Ok...", "So...", "Let me see...")
ASK_CONF_SENTENCE_END_LIST = (", right?", ", am I correct?", ", is it what you're looking for?")

# dialog states
ASK_PREFERENCE_STATE = 1
SUGGEST_STATE = 2
ASK_CONFIRMATION_STATE = 3
END_STATE = 4

# training data/tools
ALL_WORDS = 'all_words.data'
SAVED_DT = 'finalized_model.sav'
SAVED_ACT_UTT_COUPLES = 'saved_act_utt_couples.data'
all_words = list()
train_list = list()
test_list = ()
dt = tree.DecisionTreeClassifier()
dt_ready = False

# training parameters
TOLERANCE = 0.1
SPLIT_RATIO = 15

INFORM_USER = True

START_SCREEN_INFORM = "\n\tThank you for volunteering to take part in this experiment.\n\n\tWe are interested in learning about the user’s experience of a restaurant recommendation system.\n" \
                "\tThis system will respond to you via text, or via text and speech.\n" \
                "\tIn this experiment we ask you to use the recommendation system to perform some simple tasks.\n\tThe paper in front of you contains the tasks you are required to perform in the listed order.\n\n" \
                "\tAfter each dialog, we would like to ask you about your experience by asking you three questions.\n" \
                "\tPlease keep in mind that there are no right or wrong answers.\n\n\tThis should take around 10 minutes to complete.\n\tYour responses will be confidential and we do not collect identifying information such as your name or email address.\n\n\n" \
                "\tPress ENTER to continue\n"

START_SCREEN = "\n\tThank you for volunteering to take part in this experiment.\n\n\tWe are interested in learning about the user’s experience of a restaurant recommendation system.\n" \
                "\tIn this experiment we ask you to use the recommendation system to perform some simple tasks.\n\tThe paper in front of you contains the tasks you are required to perform in the listed order.\n\n" \
                "\tAfter each dialog, we would like to ask you about your experience by asking you three questions.\n" \
                "\tPlease keep in mind that there are no right or wrong answers.\n\n\tThis should take around 10 minutes to complete.\n\tYour responses will be confidential and we do not collect identifying information such as your name or email address.\n\n\n" \
                "\tPress ENTER to continue\n"

END_SCREEN = "\n\tThank you for taking part in this experiment.\n\tThe results are very important to us as we examine if speech as output of a recommendation system will influence the user’s experience.\n\n\n\n\n" \
             "\tIf you are interested in learning about the results, please contact Sam Meyer via s.j.meyer2@students.uu.nl\n"


def main():

    if TO_TRAIN:
        # save acts and utterances in a file
        # save couples of act-utterance in a list
        f = open(UTT_ACTS_TEXT_FILE, "w+")
        act_utt_couples = get_and_print_utterances_acts(os.path.join(DATA_FOLDER_1, DATA_SUBFOLDER), f) + \
                          get_and_print_utterances_acts(os.path.join(DATA_FOLDER_2, DATA_SUBFOLDER), f)
        pickle.dump(act_utt_couples, open(SAVED_ACT_UTT_COUPLES, 'wb'))
    if TO_LOAD:
        act_utt_couples = pickle.load(open(SAVED_ACT_UTT_COUPLES, 'rb'))

    # split the list of act-utterance couples in training set and test set
    split_list(act_utt_couples, SPLIT_RATIO)

    # check the acts are proportionally split
    while not_proportional(act_utt_couples, TOLERANCE):
        split_list(act_utt_couples, SPLIT_RATIO)

    # save all distinct words in a list
    global all_words
    all_words = get_all_words([j[1] for j in train_list])

    if USER_MODE:
        fit_decision_tree()
        clear()
        dialog_manager()
        return

    print(LOADING_MESSAGE)

    # prompt the user offering some options
    option = input(ENTER_OPTIONS).lower()

    # handles the user input
    while option != 'q':

        while option != '0' and option != '1' and option != '2' and option != '3' and option != '4'\
                and option != '5' and option != 'q':
            option = input(ENTER_AVAILABLE_OPTIONS).lower()

        if option == '0':
            get_performance()
        if option == '1':
            rule_based_baseline()
        elif option == '2':
            random_baseline()
        elif option == '3':
            fit_decision_tree()
            decision_tree()
        elif option == '4':
            fit_decision_tree()
            dialog_manager()
        elif option == '5':
            fit_decision_tree()
            analyze_difficult_instances()

        option = input(ENTER_NEXT_OPTION).lower()


'''
functions creating the available classifiers
'''


# offers a prompt to enter a new utterance and classify this utterance, and repeats the prompt until the user exits
# utterances are classified by a rule-based baseline system based on keyword matching
def rule_based_baseline():
    print("This is a baseline capable of classify utterances on the base of the presence of specific keywords")
    utterance = input("Enter an utterance\n\n").lower()
    while utterance != 'q':
        print(rule_based_processing(utterance))
        utterance = input("Enter an utterance\n\n").lower()


# returns the predicted dialog act given an utterance, following some rules
def rule_based_processing(utterance):
    if ("is it" in utterance or "does it" in utterance or "is that" in utterance or "is there" in utterance) and \
            (
                    "food" in utterance or "west" in utterance or "east" in utterance or "south" in utterance or "north" in utterance):
        return "confirm"
    if utterance == 'no' or utterance == 'not':
        return "negate"
    if utterance == "more":
        return "reqmore"
    if "okay" in utterance and len(utterance.split()) < 4:
        return "ack"
    if "bye" in utterance and BASELINE_CLASSIFIER:
        return "bye"
    acts_occurrences = list()
    for word in utterance.split():
        if word == "wrong":
            return "deny"
        if word == "no":
            return "negate"
        if word == "repeat" or word == "back" or word == "again":
            return "repeat"
        if word == "thank":
            return "thankyou"
        if word == "hi" or word == "hello":
            return "hello"
        if word == "start" or word == "reset":
            return "restart"
        for act in ACTS:
            if word in get_keywords(act):
                acts_occurrences.append(act)
    if len(acts_occurrences) == 0:
        return "inform"
    data = Counter(acts_occurrences)
    return data.most_common(1)[0][0]


# offers a prompt to enter a new utterance and classify this utterance, and repeats the prompt until the user exits
# utterances are classified by a baseline system that randomly assigns labels according to the label distribution in
# the data set
def random_baseline():
    print("This is a random baseline capable of classify utterances on the base of"
          "the acts distribution in the training set")
    utterance = ''
    acts_distribution = list()
    for act in ACTS:
        acts_distribution.append([j[0] for j in train_list].count(act) / len(train_list))
    while utterance != 'q':
        utterance = input("Enter a utterance\n\n").lower()
        if utterance != 'q':
            print(random_processing(acts_distribution))
    return


# returns the predicted dialog act, looking at the label distribution in the data set
def random_processing(acts_distribution):
    return random.choices(ACTS, acts_distribution)[0]


# given a list of couple dialog_act-utterance, returns a list of array of features and the relative labels,
# hence the parameters necessary to fit the decision tree
def get_dt_training_samples(act_utt_couples):
    # list of samples
    x = list()
    # list of labels
    y = list()

    for utt in [j[1] for j in act_utt_couples]:
        # append the list of features
        x.append(get_features(utt))

    for label in [j[0] for j in act_utt_couples]:
        y.append(ACTS.index(label))

    return x, y


# fits the decision tree
def fit_decision_tree():
    if dt_ready:
        return
    global dt
    if TO_TRAIN:
        param = get_dt_training_samples(train_list)
        dt = dt.fit(param[0], param[1])
        pickle.dump(dt, open(SAVED_DT, 'wb'))
    if TO_LOAD:
        dt = pickle.load(open(SAVED_DT, 'rb'))

    return


# offers a prompt to enter a new utterance and classify this utterance, and repeats the prompt until the user exits
# utterances are classified by a decision tree
def decision_tree():
    utt = input("write an utterance\n\n").lower()
    while utt != 'q':
        if utt != '':
            print(dt_processing(utt))
        utt = input("\nwrite an utterance\n\n").lower()


# returns the predicted dialog act, exploiting a decision tree
def dt_processing(utt):
    return ACTS[dt.predict([get_features(utt)])[0]]


'''
functions to get performance
'''


# prints a report of the decision tree performance
def get_dt_performance():
    print("Computing performance...\n")
    y_true = [j[0] for j in test_list]
    y_pred = list()
    for utt in [j[1] for j in test_list]:
        y_pred.append(dt_processing(utt))
    print(metrics.classification_report(y_true, y_pred, target_names=list(ACTS), digits=3))


# prints a report of the selected baseline performance
def get_baseline_performance(baseline):
    print("Computing performances...")
    y_true = [j[0] for j in test_list]
    y_pred = list()
    if baseline == "rule based":
        for utt in [j[1] for j in test_list]:
            y_pred.append(rule_based_processing(utt))
    elif baseline == "random":
        acts_distribution = list()
        for act in ACTS:
            acts_distribution.append([j[0] for j in train_list].count(act) / len(train_list))
        for i in range(len(test_list)):
            y_pred.append(random_processing(acts_distribution))
    print(metrics.classification_report(y_true, y_pred, target_names=list(ACTS), digits=3))


# prints the performance of the three classifiers
def get_performance():
    print("\nRule based baseline performance:\n\n")
    get_baseline_performance("rule based")
    print("\nRandom baseline performance:\n\n")
    get_baseline_performance("random")
    print("\nDecision tree performance:\nFitting the decision tree...\n")
    fit_decision_tree()
    get_dt_performance()


# processes some difficult instances and prints the results
def analyze_difficult_instances():
    print("\n\nFirst case of difficult instances: negation\n")
    for couple in DIFFICULT_INSTANCES_NEGATION:
        print("utterance:\t\t", couple[0])
        print("baseline act:\t", rule_based_processing(couple[0]), "( exp = ", couple[1], ')')
        print("dt act:\t\t\t", dt_processing(couple[0]), "( exp = ", couple[1], ")\n")
    print("\n\nSecond case of difficult instances: null utterances")
    for couple in DIFFICULT_INSTANCES_NULL:
        print("utterance:\t\t", couple[0])
        print("baseline act:\t", rule_based_processing(couple[0]), "( exp = ", couple[1], ')')
        print("dt act:\t\t\t", dt_processing(couple[0]), "( exp = ", couple[1], ")\n")
    print("\n\nSecond case of difficult instances: apostrophe")
    for couple in DIFFICULT_INSTANCES_APOSTROPHE:
        print("utterance:\t\t", couple[0])
        print("baseline act:\t", rule_based_processing(couple[0]), "( exp = ", couple[1], ')')
        print("dt act:\t\t\t", dt_processing(couple[0]), "( exp = ", couple[1], ")\n")


'''
functions to extract or elaborate data
'''


# randomly splits a list in two lists, according to the split ratio
def split_list(act_user_messages, split_ratio):
    global test_list, train_list
    random.shuffle(act_user_messages)
    k = len(act_user_messages) * split_ratio // 100
    test_list = act_user_messages[:k]
    train_list = act_user_messages[k:]


# returns false if train_list and test_list contain more or less proportional amounts of dialog acts
# the tolerance defines the maximum error allowed for the distribution of dialog acts
def not_proportional(dialog_list, tolerance):
    for i in ACTS:
        frequency = [j[0] for j in dialog_list].count(i) / len(dialog_list)
        train_discrepancy = abs(frequency - [j[0] for j in train_list].count(i) / len(train_list)) / frequency
        test_discrepancy = abs(frequency - [j[0] for j in test_list].count(i) / len(test_list)) / frequency
        if train_discrepancy > tolerance and test_discrepancy > tolerance:
            return True
    return False


# extracts act-utterance couples from a directory and saves them in a file
# returns a list containing the couples
def get_and_print_utterances_acts(folder_path, text_file):
    to_return = list()

    # iterating on all folders
    for root, dirs, files in os.walk(folder_path):
        for directory in dirs:
            if directory.find("Mar") != -1:

                # get all files' and folders' names in the current directory
                file_names = os.listdir(os.path.join(folder_path, directory))

                # select a directory path and prints the dialog in it, for every directory
                for i in range(len(file_names)):
                    dir_path = os.path.join(root, directory, file_names[i])

                    with open(os.path.join(dir_path, USER_JSON)) as json_data_user:

                        data_dict_user = json.load(json_data_user)

                        # save the user turns in a list
                        act_user_messages = []

                        for turn in data_dict_user.get(TURNS_ATTRIBUTE):
                            # user_messages.append(turn.get("transcription"))
                            verbose_act = turn.get("semantics").get("cam")
                            act = verbose_act[0:verbose_act.find("(")]

                            capitalized_utterance = turn.get("transcription")
                            utterance = capitalized_utterance.lower()

                            act_user_messages.append((act, utterance))
                            to_return.append((act, utterance))

                        # prints the dialog in the correct order
                        for msg in range(len(act_user_messages)):
                            text_file.write(act_user_messages[msg][0] + ((10 - len(act_user_messages[msg][0])) * " "))
                            text_file.write(act_user_messages[msg][1])
                            text_file.write("\n")

    return to_return


# returns the keywords for the specified dialog act
def get_keywords(act):
    return KEYWORDS[ACTS.index(act)]


# given a list of utterances, returns a list containing all the words, each present a single time
def get_all_words(all_utt):
    word_list = list()
    if TO_TRAIN:
        for utt in all_utt:
            for word in utt.split():
                if not (word in word_list):
                    word_list.append(word)
        pickle.dump(word_list, open(ALL_WORDS, 'wb'))
    if TO_LOAD:
        word_list = pickle.load(open(ALL_WORDS, 'rb'))
    return word_list


# given an utterance, returns an array of zeros and ones
# the ones symbolize the words contained in the utterance
# the returned array contains then the features needed to train a decision tree
def get_features(utt):
    x_features = list()
    for keyword in all_words:
        found = False
        for word in utt.split():
            if word == keyword:
                found = True
        if found:
            x_features.append(1)
        else:
            x_features.append(0)
    return x_features


'''
    state transition function
'''


def show_prompt_next_dialog(first=False):
    if first:
        if INFORM_USER:
            input("\n\t" + START_SCREEN_INFORM)
        else:
            input("\n\t" + START_SCREEN)
    else:
        input("\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t Press ENTER to start next dialog\n")
    clear()
    return


# manages a dialog with the user, taking input from the keyboard and printing output to console
# exploits a state-transition function to follow the dialog police
#
# features that can be switched on or off:
#
# Levenshtein edit distance for preference extraction
# Convert to lower case for CSV look-up
# Ask user about correctness of match for Levenshtein results
# Allow preferences to be stated in random order or not
# Allow preferences to be stated in a single utterance only, or in multiple utterances with one preference per
# utterance only, or without restrictions (any number of utterances and any number of preferences per utterance)
# Use your baseline for dialog act recognition instead of the machine learning classifier
# Start offering suggestions after the first preference type is recognized vs. wait until all preference types
# are recognized
# Allow dialog restarts or not
# Ask confirmation for each preference or not
# Allow users to change their preferences or not
# OUTPUT IN ALL CAPS OR NOT
# Limit the dialog to a certain number of utterances and fail or restart if the dialog is not finished
def dialog_manager():
    user_utterances_count = 0
    welcome_string = WELCOME_UTT
    if SINGLE_PREF:
        welcome_string = welcome_string + SINGLE_PREF_WARNING
    if ALL_CAPS:
        welcome_string = welcome_string.upper()

    # prints or reproduce the string
    output_response(welcome_string)

    next_state = ASK_PREFERENCE_STATE
    expressed_pref = {'food': '', 'pricerange': '', 'area': ''}
    reset_pref = {'food': '', 'pricerange': '', 'area': ''}
    suggestions = list()
    old_system_utt = WELCOME_UTT

    while next_state != END_STATE:
        state = next_state
        utterance = input()
        if INPUT_TO_LOWER_CASE:
            utterance = utterance.lower()
        if LEVENSHTEIN_PREPROCESSING:
            utterance = levenshtein_preprocessing(utterance)
        if NORMALIZATION_PREPROCESSING:
            utterance = utterance.translate({ord(i): None for i in '!?\''})
        next_state, response, expressed_pref, reset_pref, suggestions = state_transition(state, old_system_utt,
                                                                                         utterance,
                                                                                         expressed_pref, reset_pref,
                                                                                         suggestions)
        user_utterances_count += 1
        if user_utterances_count > MAX_UTTERANCES_NUMBER and LIMIT_UTTERANCES:
            print("\nMaximum number of utterances reached.\n")
            next_state = END_STATE
        old_system_utt = response

        if ALL_CAPS:
            response = response.upper()

        output_response(response)

    user_input = input("\nDialog terminated.\n1)\tstart new dialog\n2)\texit\n\n")
    if user_input == '1':
        clear()
        dialog_manager()
    return


# returns the next state and the system answer, given the present state and the user utterance
# the state consists in a main value (state parameter) plus the current and the old list of preferences, the current
# list of suggestions and the last system utterance
def state_transition(state, old_system_utt, utterance, expressed_pref, reset_pref, suggestions):

    to_process = utterance
    if LOWER_CASE_ACT_IDENTIFICATION:
        to_process = ('' + utterance).lower()
    act = dt_processing(to_process)
    if BASELINE_CLASSIFIER:
        act = rule_based_processing(to_process)
    answer = ''
    next_state = ''
    updated_pref = expressed_pref
    updated_sugg = suggestions

    # the reaction to these acts does not depend on the system state

    if act == act == "bye":
        next_state = END_STATE
        answer = END_UTT
        return next_state, answer, updated_pref, reset_pref, updated_sugg
    if act == "repeat" and state != ASK_CONFIRMATION_STATE:
        next_state = state
        answer = old_system_utt
        return next_state, answer, updated_pref, reset_pref, updated_sugg
    if act == "null"and state != ASK_CONFIRMATION_STATE:
        next_state = state
        answer = I_DONT_UNDERSTAND
        return next_state, answer, updated_pref, reset_pref, updated_sugg
    if act == "restart":
        if not RESTART_ALLOWED:
            next_state = state
            answer = RESTART_NOT_ALLOWED
            return next_state, answer, updated_pref, reset_pref, updated_sugg
        else:
            next_state = ASK_PREFERENCE_STATE
            answer = WELCOME_UTT
            updated_pref["food"] = ''
            updated_pref["pricerange"] = ''
            updated_pref["area"] = ''
            return next_state, answer, updated_pref, reset_pref, updated_sugg

    # the reaction to these acts does depend on the system state

    # no suggestions yet
    if state == ASK_PREFERENCE_STATE:
        if act == "inform" or act == "reqalts":
            # adds new preferences
            alt = act == "reqalts"
            to_ask = next_pref(updated_pref)
            updated_pref, found, to_check, mistake, choice, stop_change = process_pref(utterance, expressed_pref, alt, old_system_utt, to_ask)
            # updates suggested restaurants, considering if enough preferences have been expressed and a match is found
            # also updates state and answer
            if stop_change:
                intro = "You are not allowed to modify your preferences. "
                updated_sugg, next_state, answer = manage_lookup(updated_pref, updated_sugg, ack=False)
                answer = intro + answer
            elif to_check and ASK_LEVENSHTEIN_CORRECTNESS:
                answer = "I did not recognize " + mistake + ". Did you mean " + choice + "?"
                next_state = ASK_CONFIRMATION_STATE
            elif not found and RE_ASK_INEFFECTIVE_PREFERENCES:
                next_state = ASK_PREFERENCE_STATE
                answer = REPEAT_PREFERENCES
                if not RANDOM_ORDER:
                    answer = answer + ask_next_pref(updated_pref)
            elif ASK_CONFIRMATION:
                answer = random.choice(ASK_CONF_SENTENCE_OPENING_LIST) + build_sentence(updated_pref) + random.choice(ASK_CONF_SENTENCE_END_LIST)
                next_state = ASK_CONFIRMATION_STATE
            else:
                updated_sugg, next_state, ans = manage_lookup(updated_pref, updated_sugg)
                answer = "Ok, " + uncapitalize(ans)
        else:
            if not enough_pref(updated_pref, 3):
                next_state = ASK_PREFERENCE_STATE
                if act == "deny" or act == "negate":
                    answer = "Please cooperate. " + ask_next_pref(updated_pref)
                elif act == "hello":
                    answer = "Hello to you too, " + uncapitalize(ask_next_pref(updated_pref))
                elif act == "thankyou":
                    answer = "You're welcome, " + uncapitalize(ask_next_pref(updated_pref))
                elif act == "reqmore":
                    answer = "Please tell me what you're looking for before asking for more. " + ask_next_pref(updated_pref)
                elif act == "request" or "confirm":
                    answer = "Sorry, I can only give you this information after I found a restaurant for you.\n" \
                             "Help me find that! " + ask_next_pref(updated_pref)

                else:
                    # if the user does not express preferences, he is forced to do so
                    answer = "Ok, " + uncapitalize(ask_next_pref(updated_pref))
            else:
                next_state = ASK_PREFERENCE_STATE
                answer = "Please, tell me what I can do to help you"

    # suggestions available
    elif state == SUGGEST_STATE:
        if act == "hello":
            next_state = SUGGEST_STATE
            answer = "Hello, I'm here to help you"
        if act == "request":
            next_state = SUGGEST_STATE
            answer = give_info(updated_sugg[0], utterance)
        if act == "reqmore":
            next_state = SUGGEST_STATE
            if len(updated_sugg) < 2:
                answer = NO_MORE_SUGG
            else:
                updated_sugg.pop(0)
                answer = suggest(updated_sugg[0], updated_pref)
        if act == "inform" or act == "reqalts":
            to_ask = next_pref(updated_pref)
            alt = act == "reqalts"
            updated_pref, found, to_check, mistake, choice, stop_change = process_pref(utterance, expressed_pref, alt, old_system_utt, to_ask)
            if stop_change:
                intro = "You are not allowed to modify your preferences. "
                updated_sugg, next_state, answer = manage_lookup(updated_pref, updated_sugg, ack=False)
                answer = intro + answer
            elif to_check and ASK_LEVENSHTEIN_CORRECTNESS:
                answer = "I did not recognize " + mistake + ". Did you mean " + choice + "?"
                next_state = ASK_CONFIRMATION_STATE
            elif not found and RE_ASK_INEFFECTIVE_PREFERENCES:
                next_state = SUGGEST_STATE
                answer = REPEAT_PREFERENCES
                if not RANDOM_ORDER:
                    answer = answer + ask_next_pref(updated_pref)
            elif ASK_CONFIRMATION:
                answer = "Mmm... " + build_sentence(updated_pref) + ", right?"
                next_state = ASK_CONFIRMATION_STATE
            else:
                updated_sugg, next_state, ans = manage_lookup(updated_pref, updated_sugg)
                answer = "Ok, " + uncapitalize(ans)
        if act == "confirm":
            next_state = SUGGEST_STATE
            answer = check(updated_sugg[0], utterance)
        if act == "ack" or act == "affirm":
            next_state = SUGGEST_STATE
            answer = random.choice(ACK_LIST) + ", do you need anything else?"
        if act == "negate" or act == "deny":
            next_state = SUGGEST_STATE
            answer = "Oh, how can I help you then?"
        if act == "thankyou":
            next_state = SUGGEST_STATE
            answer = "You're welcome, do you need anything else?"

    # confirmation needed
    elif state == ASK_CONFIRMATION_STATE:
        if act == "affirm":
            updated_sugg, next_state, ans = manage_lookup(updated_pref, updated_sugg)
            answer = "Ok, " + uncapitalize(ans)
            reset_pref = updated_pref.copy()
        elif act == "deny" or act == "negate":
            updated_pref = reset_pref.copy()
            updated_sugg, next_state, ans = manage_lookup(updated_pref, updated_sugg, ack=False)
            answer = "Ok, what can I do then?"
        else:
            next_state = state
            answer = old_system_utt

    return next_state, answer, updated_pref, reset_pref, updated_sugg


# returns the new preferences by processing the user utterance
def process_pref(utterance, old_pref, alt, old_system_utt, to_ask):

    # the new preferences
    new_pref = old_pref.copy()
    # whether the preferences were identified
    found = False
    # whether Levenshtein was used
    to_check = False
    # the input wrong word
    mistake = ''
    # the guessed correct word
    choice = ''
    found_words = list()

    # if a reqalts has been expressed, the old preferences must be removed
    if alt:
        new_pref['food'] = ''
        new_pref['pricerange'] = ''
        new_pref['area'] = ''

    reset_pref = old_pref.copy()

    # how to interpret 'any'
    if "any" in utterance or "whatever" in utterance or "whichever" in utterance or "whatsoever" in utterance or "dont care" in utterance:
        found = True
        if "any food" in utterance and (RANDOM_ORDER or to_ask == "food"):
            new_pref['food'] = 'any'
        elif "any price" in utterance and (RANDOM_ORDER or to_ask == "pricerange"):
            new_pref['pricerange'] = 'any'
        elif "any area" in utterance and (RANDOM_ORDER or to_ask == "area"):
            new_pref['area'] = 'any'
        elif "food" in old_system_utt and (RANDOM_ORDER or to_ask == "food"):
            new_pref['food'] = 'any'
        elif "range" in old_system_utt and (RANDOM_ORDER or to_ask == "pricerange"):
            new_pref['pricerange'] = 'any'
        elif "town" in old_system_utt and (RANDOM_ORDER or to_ask == "area"):
            new_pref['area'] = 'any'

    # look for exact matches
    with open(ONTOLOGY_JSON) as json_ontology:
        data_dict = json.load(json_ontology)

        # save the user turns in a list
        for pref in data_dict.get('informable'):
            if found and SINGLE_PREF:
                break
            if pref != "name" and (RANDOM_ORDER or pref == to_ask):
                for index, word in enumerate(utterance.split()):
                    two_words_name = False
                    for guess in data_dict.get('informable').get(pref):
                        word = map_terms(word)
                        before = word
                        if index > 0 :
                            before = utterance.split()[index - 1] + ' ' + word
                        after = word
                        if len (utterance.split()) > index + 1:
                            after = utterance.split()[index + 1] + ' ' + word
                        if before == guess or after == guess:
                            new_pref[pref] = guess
                            found = True
                            two_words_name = True
                            found_words.append(guess)
                            break
                    if not two_words_name:
                        for guess in data_dict.get('informable').get(pref):
                            if word == guess:
                                new_pref[pref] = guess
                                found = True
                                found_words.append(guess)
                                break

    # if not found:
    food_regex = re.findall(r"\w+(?=\s*food)", utterance)
    if food_regex and food_regex[0] != "cheap" and food_regex[0] != "moderate" and food_regex[0] != "expensive":
        found = True
        choice = food_regex[0]
        new_pref['food'] = choice

    # look for typos in the word found in the regex
    if found:
        new_pref, corrected_regex, to_check, mistake, choice = levenshtein_checker(choice, data_dict, new_pref, to_ask, True, found_words)

    # look for typos in the other words
    new_pref, found_with_lev, to_check, mistake, choice = levenshtein_checker(utterance, data_dict, new_pref, to_ask, False, found_words)

    if found_with_lev:
        found = True

    # if a reqalts caused the deletion of the preferences but no new preferences were identified, the
    # old preferences are restored
    if not found:
        new_pref = reset_pref.copy()

    # eventual changes of already expressed preferences are deleted if not allowed
    stop_change = False
    if not CHANGE_POSSIBLE:
        for pref in old_pref:
            if old_pref.get(pref) != '' and new_pref.get(pref) != old_pref.get(pref):
                stop_change = True
                new_pref = reset_pref.copy()

    return new_pref, found, to_check, mistake, choice, stop_change


# makes lowercase the first letter of a sentence and returns that sentence
def uncapitalize(s):
    return s[:1].lower() + s[1:]


# builds a sentence describing a restaurant given the expressed preferences
def build_sentence(pref):
    sentence = ''
    food = pref.get('food')
    if food != '' and food != 'any':
        sentence = sentence + " " + food
    sentence = sentence + " restaurant"
    price = pref.get('pricerange')
    if price != '' and price != 'any':
        sentence = sentence + " in the " + price + " range"
    area = pref.get('area')
    if area != '' and area != 'any':
        sentence = sentence + " in the " + area + " of the town"
    return sentence


# given a found restaurant and the expressed preferences, returns a string to display to the user
def suggest(rest, pref, ack=True):
    if ack:
        suggestion = rest + " is a nice" + build_sentence(pref)
    else:
        suggestion = rest + " is a nice" + build_sentence(pref)
    return suggestion


# given expressed preferences, returns a string to display to the user, notifying him that no restaurants were found
def notify_absence(pref):
    apology = "Sorry, I could not find any" + build_sentence(pref)
    return apology


# returns suggested restaurants, if enough preferences have been expressed and a match is found
# also returns answer and next_state
def manage_lookup(updated_pref, sugg, ack= True):
    updated_sugg = sugg
    if enough_pref(updated_pref):
        updated_sugg, match_found = lookup_restaurant(updated_pref)
        if RANDOM_SUGG:
            random.shuffle(updated_sugg)
        if match_found:
            next_state = SUGGEST_STATE
            answer = suggest(updated_sugg[0], updated_pref, ack)
        else:
            next_state = ASK_PREFERENCE_STATE
            answer = notify_absence(updated_pref)
    else:
        next_state = ASK_PREFERENCE_STATE
        answer = ask_next_pref(updated_pref)

    return updated_sugg, next_state, answer


# lookup for restaurants matching the preferences
def lookup_restaurant(pref):
    restaurant_found = False
    sugg_restaurants = list()  # test multiple suggestions: european, south, expensive

    with open(RESTAURANT_INFO_FILE) as file:
        restaurant_info = csv.reader(file)

        for row in restaurant_info:
            if ((pref["pricerange"] == row[1] or pref["pricerange"] == '' or pref["pricerange"] == "any") and
                    (pref["area"] == row[2] or pref["area"] == '' or pref["area"] == "any") and
                    (pref["food"] == row[3] or pref["food"] == '' or pref["food"] == "any")):
                sugg_restaurants.append(row[0])
                restaurant_found = True

    return sugg_restaurants, restaurant_found


# processes a question and answers checking the restaurant data
def give_info(restaurant, question):
    restaurant_found = False
    question_identified = False
    info = ''

    with open(RESTAURANT_INFO_FILE) as file:
        restaurant_info = csv.reader(file)

        for row in restaurant_info:
            if restaurant == row[0]:
                restaurant_found = True
                if "phone" in question or "number" in question:
                    question_identified = True
                    if row[4] != '':
                        ans = 'You can reach the restaurant by calling ' + row[4]
                        info = info + ans + '\n'
                    else:
                        ans = 'Phone number is unknown'
                        info = info + ans + '\n'

                if "address" in question or "location" in question:
                    question_identified = True
                    if row[5] != '':
                        ans = 'You can find the restaurant at ' + row[5]
                        info = info + ans + '\n'
                    else:
                        ans = 'Address is unknown'
                        info = info + ans + '\n'

                if "postcode" in question or "location" in question:
                    question_identified = True
                    if row[6] != '':
                        ans = 'The postcode of the restaurant is ' + row[6]
                        info = info + ans + '\n'
                    else:
                        ans = 'Postcode is unknown'
                        info = info + ans + '\n'

    if not restaurant_found:
        return "I'm sorry, this restaurant is unknown"
    if not question_identified:
        return "I'm sorry, I cannot answer that question"
    return info


# returns whether the restaurant matches the data in the utterance
def check(restaurant, utterance):

    right = False

    with open(RESTAURANT_INFO_FILE) as file:
        restaurant_info = csv.reader(file)

        for row in restaurant_info:
            if restaurant == row[0]:
                for i in range(1,6):
                    if row[i] in utterance:
                        right = True
                        break

    if right:
        return "yes"
    else:
        return "no"


# randomly asks for a so far unvoiced preference. Returns a string containing the question
def ask_next_pref(old_pref):
    if not RANDOM_ORDER:
        pref_to_ask = next_pref(old_pref)
        if pref_to_ask == 'food':
            return ASK_PREF[0]
        if pref_to_ask == 'pricerange':
            return ASK_PREF[1]
        if pref_to_ask == 'area':
            return ASK_PREF[2]
    else:
        options = list()
        if old_pref.get('food') == '':
            options.append(ASK_PREF[0])
        if old_pref.get('pricerange') == '':
            options.append(ASK_PREF[1])
        if old_pref.get('area') == '':
            options.append(ASK_PREF[2])
        return random.choice(options)


# return the next preference to express, following the order "food", "pricerange", "area"
def next_pref(pref):
    if pref.get('food') == '':
        return "food"
    if pref.get('pricerange') == '':
        return "pricerange"
    if pref.get('area') == '':
        return "area"
    return "ALL PREFERENCES EXPRESSED"


# returns whether enough preferences have been expressed so far
def enough_pref(pref, n=MIN_PREF):
    count = 0
    for field in pref:
        if pref.get(field) != '':
            count += 1
    return count >= n


# maps some common words to the keyword in the json file
def map_terms(word):
    if word == "cheaply" or word == "low-priced" or word == "low":
        return "cheap"
    if word == "expensively" or word == "high-priced" or word == "high":
        return "expensive"
    if word == "moderately" or word == "medium-priced" or word == "medium":
        return "moderate"
    if word == "center":
        return "centre"
    return word


# given an utterance, returns a string containing the missplelled word an a string containing the relative guesses
def levenshtein_checker(utterance, data_dict, new_pref, to_ask, pattern, found_words):
    found = False
    to_check = False
    mistakes = ""
    choices = ""
    for word in utterance.split():
        close_words = [[] for i in range(LEVENSHTEIN_LIMIT)]
        if len(word) > 2 and (word not in CRITIC_WORDS or pattern):
            if pattern:
                for guess in data_dict.get('informable').get("food"):
                    lev_distance = levenshtein(word, guess)
                    if lev_distance/len(guess) < MAX_LEVENSHTEIN_RATIO and guess not in found_words:
                        close_words[lev_distance].append((guess, word, "food"))
            else:
                for pref in data_dict.get('informable'):
                    if pref != "name" and (RANDOM_ORDER or pref == to_ask):
                        for guess in data_dict.get('informable').get(pref):
                            lev_distance = levenshtein(word, guess)
                            if lev_distance/len(guess) < MAX_LEVENSHTEIN_RATIO and guess not in found_words:
                                close_words[lev_distance].append((guess, word, pref))
            for i in range(0, LEVENSHTEIN_LIMIT):
                if len(close_words[i]) > 1:
                    word_info = random.choice(close_words[i])
                    choices = choices + (word_info[0]) + ', '
                    mistakes = mistakes + (word_info[1]) + ', '
                    new_pref[word_info[2]] = word_info[0]
                    found = True
                    to_check = True
                    break
                elif len(close_words[i]) > 0:
                    word_info = close_words[i][0]
                    choices = choices + (word_info[0]) + ', '
                    mistakes = mistakes + (word_info[1]) + ', '
                    new_pref[word_info[2]] = word_info[0]
                    found = True
                    to_check = True
                    break

    if choices.endswith(', '):
        choices = choices[:-2]
    if mistakes.endswith(', '):
        mistakes = mistakes[:-2]

    return new_pref, found, to_check, mistakes, choices


# checks if the utterance contain the misspelling of a word particularly common in this context
def levenshtein_preprocessing(utt):
    for word in utt.split():
        for correct_word in CORRECT_WORDS:
            if levenshtein(word, correct_word)/len(correct_word) < MAX_LEVENSHTEIN_RATIO:
                utt = utt.replace(word, correct_word)
    return utt


# returns the levenshtein distance between two strings
def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return int(matrix[size_x - 1, size_y - 1])


# prints and/or reproduces an utterance
def output_response(response):

    global f
    f.write("S: " + response + "\n")
    if PRINT_RESPONSE:
        print("S: ", response)
    if TEXT_TO_SPEECH:
        vocal = gTTS(text=response, lang='en', slow=False)
        vocal.save("utterance.mp3")
        os.system("utterance.mp3")


# clears the console
def clear():
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')

        # for mac and linux
    else:
        _ = os.system('clear')


# conducts a survey and stores the collected data in a csv file
def survey(group_number):
    data_path = "text_data.csv"

    if TEXT_TO_SPEECH:
        data_path = "speech_data.csv"

    if os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0)

    answers = np.zeros(len(SURVEY_QUESTIONS))
    print("\n\tPlease rate the degree to which the following phrases correspond with your experience on a scale of 1 to " + str(SURVEY_SCALE) + ", where:\n\n" \
          "\t1 = Strongly disagree\n\t2 = Disagree\n\t3 = Neither agree nor disagree\n\t4 = Agree\n\t5 = Strongly agree\n\n\n")
    for index, question in enumerate(SURVEY_QUESTIONS):
        ans = input("\n" + question + "\n")
        while ans != '1' and ans != '2' and ans != '3' and ans != '4' and ans != '5':
            ans = input("Please enter a number between 1 and " + str(SURVEY_SCALE) + "\n")
        answer = int(ans)
        answers[index] = answer

    if not os.path.exists(data_path):
        headers = np.array(["question1", "question2", "question3"])
        data = pd.DataFrame(columns=headers)
        data.loc[0] = answers
    else:
        data.loc[int(data.shape[0])] = answers

    data.to_csv(data_path)

    # sleep 1 sec not to scare the user?
    time.sleep(SLEEP_TIME)
    clear()


if __name__ == "__main__":
    main()
