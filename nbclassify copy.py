import re
import numpy as np
import os
import sys
import string
import math
from nblearn import STOP_WORDS
from nblearn import NaiveBayes, clean_up
from collections import defaultdict
from collections import Counter
import json
import glob

polarity_dict = {'0' : 'negative',
                    '1' : 'positive'}
truth_dict = {'0' : 'deceptive',
                '1' : 'truthful'}

def load_test_data(input_path):
    X = []
    X_paths = glob.glob(os.path.join(input_path, "*", "*", "*", "*.txt"))
    for file_path in X_paths:
        with open(file_path, 'r') as f:
            file_content = f.read()
            X.append(clean_up(file_content))
    return X, X_paths

if __name__ == '__main__':
    test_dir = sys.argv[1]
    param_file = './nbmodel.txt'
    output_path = './nboutput.txt'
    with open(param_file) as f:
        params = json.load(f)

    NB_Polarity = NaiveBayes()
    NB_Polarity.load_params(params['PolarityClassifierParams'])

    NB_truth = NaiveBayes()
    NB_truth.load_params(params['TruthClassifierParams'])

    test_X, test_paths = load_test_data(test_dir)
    polarity_prediction = NB_Polarity.predict_multi(test_X)
    truth_prediction = NB_truth.predict_multi(test_X)
    
    polarities = [polarity_dict[x] for x in polarity_prediction]
    truths = [truth_dict[x] for x in truth_prediction]

    with open(output_path, 'w') as f:
        for label_a, label_b, path in zip(truths, polarities, test_paths):
            f.write(str(label_a) + ' ' + str(label_b) + ' ' + str(path) + '\n')
            