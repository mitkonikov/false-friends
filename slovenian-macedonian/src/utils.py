import random

def read_dataset(path, false_friends = True):
    dataset = []
    
    # Open the input file for reading
    with open(path, 'r', encoding='utf8') as infile:
        # Read each line from the input file
        for line in infile:
            # Split the line into words
            words = line.split()
            # Check if the line has at least two words
            if len(words) >= 2:
                # Extract the first and last words
                first_word = words[0]
                last_word = words[-1]
                # Write the first and last words to the output file
                dataset.append((first_word, last_word, 1 if false_friends else 0))

    return dataset

def split_train_test(dataset, train_size = 0.8, seed = 42):
    # Set the random seed for reproducibility
    random.seed(seed)

    # Shuffle the lines randomly
    random.shuffle(dataset)

    # Calculate the split index
    split_index = int(train_size * len(dataset))

    # Split the lines into training and testing datasets
    train = dataset[:split_index]
    test = dataset[split_index:]

    return train, test

def _print_metrics_matrix(measures):
    np = measures["Neg. Precision"]
    nr = measures["Neg. Recall"]
    nf = measures["Neg. F1-score"]
    pp = measures["Precision"]
    pr = measures["Recall"]
    pf = measures["F1-score"]

    print("               precision      recall         f1-score")
    print("")
    print(
        "     False     {np:0.4f}         {nr:0.4f}         {nf:0.4f}".format(
            np=np, nr=nr, nf=nf
        )
    )
    print(
        "     True      {pp:0.4f}         {pr:0.4f}         {pf:0.4f}".format(
            pp=pp, pr=pr, pf=pf
        )
    )
    print("")
    print(
        "avg / total    {ap:0.4f}         {ar:0.4f}         {af:0.4f}".format(
            ap=(np + pp) / 2, ar=(nr + pr) / 2, af=(nf + pf) / 2
        )
    )
    print("")

def _print_confusion_matrix(measures):
    tn = measures["tn"]
    fp = measures["fp"]
    fn = measures["fn"]
    tp = measures["tp"]
    print("Confusion matrix")
    print("")
    print("\t\t(classified as)")
    print("\t\tTrue\tFalse")
    print("(are)\tTrue\t{tp:0.4f}\t{fn:0.4f}".format(tp=tp, fn=fn))
    print("(are)\tFalse\t{fp:0.4f}\t{tn:0.4f}".format(fp=fp, tn=tn))
    print("")

def print_measures(measures, print_verbose = True):
    if print_verbose:
        print("")
        print("Cross-validation measures with 95% of confidence:")

        for measure_name, (mean, delta) in measures.items():
            print(
                "{measure_name}: {mean:0.4f} ± {delta:0.4f} --- [{inf:0.4f}, {sup:0.4f}]".format(
                    measure_name=measure_name,
                    mean=mean,
                    delta=delta,
                    inf=mean - delta,
                    sup=mean + delta,
                )
            )

    print("")

    mean_measures = {
        measure_name: mean for measure_name, mean in measures.items()
    }
    _print_metrics_matrix(mean_measures)
    _print_confusion_matrix(mean_measures)
