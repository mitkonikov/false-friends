import torch
import numpy as np
import argparse
import collections
import utils
import features
import classify
import llm

from sklearn import naive_bayes, neighbors, svm, tree, neural_network

if __name__ == "__main__":
    CLF_OPTIONS = {
        "DT": tree.DecisionTreeClassifier(),
        "GNB": naive_bayes.GaussianNB(),
        "kNN": neighbors.KNeighborsClassifier(),
        "SVM": svm.SVC(),
        "NN": neural_network.MLPClassifier([2000, 2000])
    }

    def command_classify(args_):
        print(f'Random seed is set to {args_.seed}.')

        torch.manual_seed(args_.seed)
        np.random.seed(args_.seed)

        false_friends = utils.read_dataset(args_.false_friends, True)
        true_friends = utils.read_dataset(args_.true_friends, False)

        train, test = utils.split_train_test(false_friends + true_friends, train_size = 0.8, seed = args_.seed)
        
        bert = llm.BERTExtractor()

        (X_train, y_train) = features.features_and_labels(train, bert)
        (X_test, y_test) = features.features_and_labels(test, bert)

        clf = classify.build_classifier(CLF_OPTIONS[args_.classifier])
        measures = classify.classify(X_train, X_test, y_train, y_test, clf=clf) # (X, y, clf=clf)
        utils.print_measures(measures, False)

    COMMANDS = collections.OrderedDict(
        [
           (
                "classify",
                {
                    "function": command_classify,
                    "help": "Train a classifier given a false/true friends dataset",
                    "parameters": [
                        {
                            "name": "--false_friends",
                            "args": {},
                        },
                        {
                            "name": "--true_friends",
                            "args": {},
                        },
                        {
                            "name": "--classifier",
                            "args": {
                                "choices": sorted(list(CLF_OPTIONS.keys())),
                                "default": "NN",
                            },
                        },
                        {
                            "name": "--seed",
                            "args": {
                                "default": 0,
                            },
                        },
                    ],
                },
            ),
        ]
    )

    def args():
        arg_parser_ = argparse.ArgumentParser()
        subparsers = arg_parser_.add_subparsers(dest="command", title="command")

        for command, command_values in COMMANDS.items():
            sub_parser = subparsers.add_parser(command, help=command_values["help"])

            for parameter in command_values["parameters"]:
                sub_parser.add_argument(parameter["name"], **parameter["args"])

        return arg_parser_, arg_parser_.parse_args()

    arg_parser, args = args()

    if args.command:
        # noinspection PyCallingNonCallable
        COMMANDS[args.command]["function"](args)
    else:
        arg_parser.print_help()
