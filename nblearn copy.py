import re
from textwrap import indent
import numpy as np
import os
import sys
import string
import math
from collections import defaultdict
from collections import Counter
import unicodedata
import json

STOP_WORDS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
            'about','above','after','all','am','an','and','any','as','at','be','been','before','being','below','between',
            'both','but','by','doing','down','during','each','for','from','further','had','has','have','having','he','hed',
            'll','hes','her','here','here','hers','herself','him','himself','his','how','hows','id','im','ive','if','in',
            'into','is','it','its','lets','me','more','my','myself','of','on','only','or','other','ought','our','ours','ourselves',
            'over','own','she','so','some','such','than','that','thats','the','their','theirs','them','themselves','then','there',
            'theyd','theyll','theyre','theyve','ve','this','those','through','to','too','under','up','was','we','wed','well',
            'were','weve','what','whats','when','where','which','while','who','whom','why','with','would','you',
            'theres','these','they','youd','youll','youre','youve','your','yours','yourselves', 'hotel', 'hotels']

def load_data(path_to_data):
    X = []
    X_paths = []
    y1 = []
    y2 = []

    for root, subdirs, files in os.walk(path_to_data+'/negative_polarity/deceptive_from_MTurk'):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as f:
                    file_content = f.read()
                    X.append(clean_up(file_content))
                    y1.append('0')
                    y2.append('0')
                    X_paths.append(str(file_path))

    for root, subdirs, files in os.walk(path_to_data+'/negative_polarity/truthful_from_Web'):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as f:
                    file_content = f.read()
                    X.append(clean_up(file_content))
                    y1.append('0')
                    y2.append('1')
                    X_paths.append(str(file_path))

    for root, subdirs, files in os.walk(path_to_data+'/positive_polarity/deceptive_from_MTurk'):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as f:
                    file_content = f.read()
                    X.append(clean_up(file_content))
                    y1.append('1')
                    y2.append('0')
                    X_paths.append(str(file_path))

    for root, subdirs, files in os.walk(path_to_data+'/positive_polarity/truthful_from_TripAdvisor'):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as f:
                    file_content = f.read()
                    X.append(clean_up(file_content))
                    y1.append('1')
                    y2.append('1')
                    X_paths.append(str(file_path))

    return X, X_paths, y1, y2

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def remove_stopwords(words):
    words = words.split(' ')
    filtered_words = [word for word in words if word.lower() not in STOP_WORDS]
    return ' '.join(filtered_words)


def clean_up(stringerino):
    processed_str = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", stringerino)
    processed_str = ' '.join(processed_str.split())
    processed_str = strip_accents(processed_str)
    processed_str = remove_stopwords(processed_str)
    return processed_str

def save_params(polarity_params, truth_params, filepath):
    parameters = {'PolarityClassifierParams': polarity_params, 
                  'TruthClassifierParams': truth_params}

    with open(filepath, 'w') as f:
        f.write(json.dumps(parameters, indent = 4))

class NaiveBayes:

    def __init__(self):
        self.log_likelihoods = defaultdict(lambda : defaultdict(float))
        self.num_samples = defaultdict(int)
        self.log_class_priors = defaultdict(float)
        self.word_freq = defaultdict(lambda : defaultdict(float))
        self.word_list = defaultdict(int)
        self.log_likelihood_denoms = defaultdict(int)
        self.total_freq = defaultdict(int)        

    def tokenize_string(self, string):
        return re.split("\W+", clean_up(string))

    def get_word_freq(self, words):
        word_freq = defaultdict(float)
        for word in words:
            word_freq[word] = word_freq[word] + 1.0
        return word_freq
    
    #following algorithmic flow : 
    '''Compute log of class priors. Class priors can be counted by class freq divided by total num of training samples'''
    def train(self, X, y):
        num = len(X)
        self.num_samples['0'] = sum(1 for label in y if label == '0')
        self.num_samples['1'] = sum(1 for label in y if label == '1')
        
        self.log_class_priors['0'] = math.log(self.num_samples['0'] / num)
        self.log_class_priors['1'] = math.log(self.num_samples['1'] / num)

        for sample, label in zip(X, y):
            counts = self.get_word_freq(self.tokenize_string(sample))
            for word, count in counts.items():
                self.word_list[word] += count
                self.word_freq[label][word] += count
        
        for word in self.word_list.keys():
            self.total_freq['0'] += self.word_freq['0'][word]
            self.total_freq['1'] += self.word_freq['1'][word]
        
        self.log_likelihood_denoms['0'] = self.total_freq['0'] + len(self.word_list)
        self.log_likelihood_denoms['1'] = self.total_freq['1'] + len(self.word_list)

        for word in self.word_list.keys():
            self.log_likelihoods['0'][word] = math.log((self.word_freq['0'][word] + 1) / self.log_likelihood_denoms['0'])
            self.log_likelihoods['1'][word] = math.log((self.word_freq['1'][word] + 1) / self.log_likelihood_denoms['1'])

    def get_params(self):
        parameters = {
                    'Word_List':self.word_list,
                    'Log_Class_Priors':self.log_class_priors,
                    'Word_Freq_Per_Class':self.word_freq,
                    'Log_Likelihoods':self.log_likelihoods
                    }
        return parameters

    def load_params(self, parameters):
        self.word_list = parameters['Word_List']
        self.log_class_priors = parameters['Log_Class_Priors']
        self.word_freq = parameters['Word_Freq_Per_Class']
        self.log_likelihoods = parameters['Log_Likelihoods']

    def predict_single(self, sample):
            counts = self.get_word_freq(self.tokenize_string(sample))
            log_prob = defaultdict(float)
            for word, _ in counts.items():
                #ignoring uknown tokens for the time being
                if word not in self.word_list.keys():
                    continue

                log_prob['0'] += self.log_likelihoods['0'][word]
                log_prob['1'] += self.log_likelihoods['1'][word]

            log_prob['0'] += self.log_class_priors['0']
            log_prob['1'] += self.log_class_priors['1']

            prediction = max(log_prob,key=log_prob.get)
            return prediction

    def predict_multi(self, X):
        predictions = []
        for sample in X:
            prediction = self.predict_single(sample)
            predictions.append(prediction)
        return predictions

if __name__ == '__main__':
    train_dir = sys.argv[1]
    X, X_paths, y1, y2 = load_data(train_dir)

    NB_Polarity = NaiveBayes()
    NB_Polarity.train(X, y1)
    polarity_params = NB_Polarity.get_params()

    NB_truth = NaiveBayes()
    NB_truth.train(X, y2)
    truth_params = NB_truth.get_params()

    save_params(polarity_params, truth_params, './nbmodel.txt')