import argparse
import io

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_embedding(embedding_path):
    print(f'Loading embedding from {embedding_path}')
    data = dict()
    fin = io.open(embedding_path, 'r', encoding='utf-8', newline='\n')
    n, d = map(int, fin.readline().split())
    for line in fin:
        t = line.rstrip().split()
        data[t[0]] = list(map(float, t[1:]))
    return data

class Dataset():
    def __init__(self, dataset, file_path, DT_embedding_data, fasttext_embedding_data, relations1, relations2, is_en, random_state=42):
        self.dataset = dataset
        self.file_path = file_path
        self.random_state = random_state
        self.is_en = is_en
        self.df = pd.read_csv(self.file_path, header=None, sep='\t')
        self.df = self.df.sample(n=len(self.df), random_state=self.random_state)

        if(self.dataset == 'bless-gems'):
            self.df.columns = ['word1', 'class', 'relation', 'word2']
            self.df.word1 = self.df.word1.apply(lambda x: x.rsplit('-', 1)[0])
            self.df.word2 = self.df.word2.apply(lambda x: x.rsplit('-', 1)[0])
            self.df = self.df[self.df.relation.isin(['random-n', 'random-v', 'random-j', 'coord', 'hyper'])]
        elif(self.dataset == 'ROOT9' or self.dataset == 'Baroni'):
            self.df.columns = ['word1', 'relation', 'word2']
            self.df.word1 = self.df.word1.apply(lambda x: x.rsplit('-', 1)[0])
            self.df.word2 = self.df.word2.apply(lambda x: x.rsplit('-', 1)[0])
        elif(self.dataset == 'EVALution'):
            self.df.columns = ['word1', 'relation', 'word2']

        self.DT_embedding_data = DT_embedding_data
        self.fasttext_embedding_data = fasttext_embedding_data
        self.DT_embed_tokens = list()
        self.fasttext_embed_tokens = list()
        if(self.is_en):
            for t in self.DT_embedding_data:
                token, tag = t.rsplit('/', 1)
                if(tag in ['NN', 'NNP', 'NNS']):
                    self.DT_embed_tokens.append(token)
        else:
            self.DT_embed_tokens = list(self.DT_embedding_data.keys())
        self.fasttext_embed_tokens = list(self.fasttext_embedding_data.keys())

        self.df_DT = self.df[self.df['word1'].isin(self.DT_embed_tokens) & self.df['word2'].isin(self.DT_embed_tokens)]
        self.df_fasttext = self.df[~self.df.apply(tuple, 1).isin(self.df_DT.apply(tuple, 1))]

        assert(self.df.shape[0] == self.df_DT.shape[0] + self.df_fasttext.shape[0])

        self.relations1 = relations1
        if(relations2 is None or relations2 == ['random']): # relations2 contains everything apart from relations1
            self.relations2 = list(set(self.df.relation.unique()) - set(relations1))
        else:
            self.relations2 = relations2
        
        comparison_relations = []
        for r1 in self.relations1:
            comparison_relations.append(r1.lower())
            comparison_relations.append(r1.upper())
        for r2 in self.relations2:
            comparison_relations.append(r2.lower())
            comparison_relations.append(r2.upper())

        self.df_DT = self.df_DT[self.df_DT.relation.isin(comparison_relations)]
        self.df_fasttext = self.df_fasttext[self.df_fasttext.relation.isin(comparison_relations)]

        # print(f'self.df[self.df.relation.isin(comparison_relations)].shape: {self.df[self.df.relation.isin(comparison_relations)].shape}')
        # print(f'self.df_DT.shape: {self.df_DT.shape}')
        # print(f'self.df_fasttext.shape: {self.df_fasttext.shape}')

        def mapper(x):
            if(x in self.relations1):
                return 0
            elif(x in self.relations2):
                return 1
        print('relations1', self.relations1)
        print('relations2', self.relations2)
        self.df_DT.relation = self.df_DT.relation.apply(lambda x: mapper(x))
        self.df_fasttext.relation = self.df_fasttext.relation.apply(lambda x: mapper(x))

    def replace_token_with_DT_embed(self, token):
        if(self.is_en):
            embed_tokens_with_tags = self.DT_embedding_data.keys()
            token_with_tags = [token + '/NN', token + '/NNP', token + '/NNS']
            for token_with_tag in token_with_tags:
                if(token_with_tag in embed_tokens_with_tags):
                    return self.DT_embedding_data[token_with_tag]
        else:
            return self.DT_embedding_data[token]

    def replace_token_with_fasttext_embed(self, token):
        if(token in self.fasttext_embed_tokens):
            return self.fasttext_embedding_data[token]
        else:
            return 300*[0.0]

    def perform_operation(self, operation):
        DT_word_1_embeddings = np.array(self.df_DT.word1.apply(lambda x: self.replace_token_with_DT_embed(x)).tolist())
        DT_word_2_embeddings = np.array(self.df_DT.word2.apply(lambda x: self.replace_token_with_DT_embed(x)).tolist())
        assert(DT_word_1_embeddings.shape == DT_word_2_embeddings.shape)
        DT_encoded_relation = np.array(self.df_DT.relation.tolist())
        fasttext_word_1_embeddings = np.array(self.df_fasttext.word1.apply(lambda x: self.replace_token_with_fasttext_embed(x)).tolist())
        fasttext_word_2_embeddings = np.array(self.df_fasttext.word2.apply(lambda x: self.replace_token_with_fasttext_embed(x)).tolist())
        assert(fasttext_word_1_embeddings.shape == fasttext_word_2_embeddings.shape)
        fasttext_encoded_relation = np.array(self.df_fasttext.relation.tolist())

        if(operation == 'DIFF'):
            DT_feature = DT_word_1_embeddings - DT_word_2_embeddings
            fasttext_feature = fasttext_word_1_embeddings - fasttext_word_2_embeddings
        elif(operation == 'MULT'):
            DT_feature = np.multiply(DT_word_1_embeddings, DT_word_2_embeddings)
            fasttext_feature = np.multiply(fasttext_word_1_embeddings, fasttext_word_2_embeddings)
        elif(operation == 'ADD'):
            DT_feature = DT_word_1_embeddings + DT_word_2_embeddings
            fasttext_feature = fasttext_word_1_embeddings + fasttext_word_2_embeddings
        elif(operation == 'CAT'):
            DT_feature = np.concatenate((DT_word_1_embeddings, DT_word_2_embeddings), axis=1)
            fasttext_feature = np.concatenate((fasttext_word_1_embeddings, fasttext_word_2_embeddings), axis=1)

        return (DT_feature, DT_encoded_relation), (fasttext_feature, fasttext_encoded_relation)

    def perform_sampling(self, operation, sampler):
        (DT_feature, DT_encoded_relation), (fasttext_feature, fasttext_encoded_relation) = self.perform_operation(operation=operation)
        if(sampler == 'under'):
            rus = RandomUnderSampler(random_state=random_state)
            DT_feature, DT_encoded_relation = rus.fit_sample(DT_feature, DT_encoded_relation)
            fasttext_feature, fasttext_encoded_relation = rus.fit_sample(fasttext_feature, fasttext_encoded_relation)
        DT_encoded_relation = pd.Series(DT_encoded_relation)
        print('Sampled DT distribution')
        print(DT_encoded_relation.value_counts())
        fasttext_encoded_relation = pd.Series(fasttext_encoded_relation)
        print('Sampled fasttext distribution')
        print(fasttext_encoded_relation.value_counts())
        return (DT_feature, DT_encoded_relation), (fasttext_feature, fasttext_encoded_relation)

def train_on_custom_split(model_name, train_dataset, test_dataset, operation, sampler, scoring, random_state):
    print('Training')
    (train_DT_feature, train_DT_encoded_relation), (train_fasttext_feature, train_fasttext_encoded_relation) = train_dataset.perform_sampling(operation, sampler)
    print('Testing')
    (test_DT_feature, test_DT_encoded_relation), (test_fasttext_feature, test_fasttext_encoded_relation) = test_dataset.perform_sampling(operation, sampler)
    print(model_name, operation)
    if(model_name == 'SVM'):
        DT_clf = LinearSVC(random_state=random_state, max_iter=100000)
        fasttext_clf = LinearSVC(random_state=random_state, max_iter=100000)
    elif(model_name == 'RBF-SVM'):
        DT_clf = SVC(kernel='rbf')
        fasttext_clf = SVC(kernel='rbf')
    elif(model_name == 'RF'):
        DT_clf = SVC(kernel='rbf')
        fasttext_clf = SVC(kernel='rbf')
    DT_clf.fit(train_DT_feature, train_DT_encoded_relation)
    fasttext_clf.fit(train_fasttext_feature, train_fasttext_encoded_relation)
    DT_clf_pred = DT_clf.predict(test_DT_feature)
    fasttext_clf_pred = fasttext_clf.predict(test_fasttext_feature)
    pred = np.concatenate((DT_clf_pred, fasttext_clf_pred), axis=0)
    target = np.concatenate((test_DT_encoded_relation.to_numpy(), test_fasttext_encoded_relation.to_numpy()), axis=0)
    print(f'Accuracy: {accuracy_score(target, pred)}')

if __name__ == '__main__':
    random_state = 42
    np.random.seed(random_state)
    models = ['SVM', 'RF', 'RBF-SVM']
    operations = ['DIFF', 'MULT', 'ADD', 'CAT']

    parser = argparse.ArgumentParser(description='train back-off classifier for semantic relation classification task')
    parser.add_argument('--input_token_emd_path',
                        required=True,
                        help='input path to token.emd')
    parser.add_argument('--fasttext_path',
                        help='optional path to fasttext embedding')
    parser.add_argument('--input_dataset',
                        required=True,
                        choices=['bless-gems', 'ROOT9', 'EVALution', 'Baroni'],
                        help='input dataset type')
    parser.add_argument('--input_dataset_path',
                        required=True,
                        help='input path to dataset')
    parser.add_argument('--relations1', 
                        nargs='*',
                        help='specify relations in category 1')
    parser.add_argument('--relations2', 
                        nargs='*',
                        help='specify relations in category 1')
    parser.add_argument('--operation',
                        choices=operations + ['ALL'], 
                        required=True, 
                        help='specify vector operation to perform on the word embeddings')
    parser.add_argument('--sampler', 
                        choices=['under', 'none'],
                        help='mention sampling technique')
    parser.add_argument('--model_name',
                        choices=models + ['ALL'],
                        required=True,
                        help='mention the model to be used')
    parser.add_argument('--scoring',
                        default='accuracy',
                        choices=['accuracy', 'f1'])
    parser.add_argument('--is_en',
                        action='store_true',
                        default=False,
                        help='enable if analysis is for lang=en')
    parser.add_argument('--test_split')
    args = parser.parse_args()

    if(args.model_name == 'RBF-SVM'):
        assert(args.operation == 'DIFF')

    print('Loading DT embedding')
    DT_embedding_data = load_embedding(args.input_token_emd_path)
    print('Loading fasttext embedding')
    fasttext_embedding_data = load_embedding(args.fasttext_path)

    print('Loading training dataset')
    train_dataset = Dataset(dataset=args.input_dataset, file_path=args.input_dataset_path, DT_embedding_data=DT_embedding_data, fasttext_embedding_data=fasttext_embedding_data, relations1=args.relations1, relations2=args.relations2, is_en=args.is_en, random_state=random_state)
    print('Loaded training dataset')

    print('Loading testing dataset')
    test_dataset = Dataset(dataset=args.input_dataset, file_path=args.test_split, DT_embedding_data=DT_embedding_data, fasttext_embedding_data=fasttext_embedding_data, relations1=args.relations1, relations2=args.relations2, is_en=args.is_en, random_state=random_state)
    print('Loaded testing dataset')
    if(args.model_name == 'ALL' and args.operation == 'ALL'):
  	    for model_name in models:
  	        for operation in operations:
  	            train_on_custom_split(model_name=model_name, train_dataset=train_dataset, test_dataset=test_dataset, operation=operation, sampler=args.sampler, scoring=args.scoring, random_state=random_state)
    else:
        train_on_custom_split(model_name=args.model_name, train_dataset=train_dataset, test_dataset=test_dataset, operation=args.operation, sampler=args.sampler, scoring=args.scoring, random_state=random_state)
