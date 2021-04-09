import argparse 
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

class Dataset():
    def __init__(self, 
                dataset, 
                relation, 
                file_path, 
                embedding_path, 
                relations1, 
                relations2, 
                is_en,
                is_full_test,
                hybrid_model,
                random_state,
                fasttext_path=None):
        self.file_path = file_path
        self.dataset = dataset
        self.random_state = random_state
        self.df = pd.read_csv(self.file_path, header=None, sep='\t')
        self.df = self.df.sample(n=len(self.df), random_state=self.random_state)

        self.hybrid_model = hybrid_model

        self.fasttext_path = None
        if(fasttext_path is not None):
            self.fasttext_path = fasttext_path
            self.fasttext_embedding_data = dict()
            print('Loading fasttext embeddings')
            fin = io.open(self.fasttext_path, 'r', encoding='utf-8', newline='\n')
            n, d = map(int, fin.readline().split())
            for line in fin:
                tokens = line.rstrip().split()
                self.fasttext_embedding_data[tokens[0]] = list(map(float, tokens[1:]))

        self.embedding_path = embedding_path
        self.embedding_data = dict()
        fin = io.open(self.embedding_path, 'r', encoding='utf-8', newline='\n')
        n, d = map(int, fin.readline().split())
        for line in fin:
            tokens = line.rstrip().split()
            self.embedding_data[tokens[0]] = list(map(float, tokens[1:]))

        if(self.dataset == 'bless-gems'):
            self.df.columns = ['word1', 'class', 'relation', 'word2']
            self.df.word1 = self.df.word1.apply(lambda x: x.rsplit('-', 1)[0])
            self.df.word2 = self.df.word2.apply(lambda x: x.rsplit('-', 1)[0])
            self.df = self.df[self.df.relation.isin(['random-n', 'random-v', 'random-j', 'coord', 'hyper'])]
        elif(self.dataset == 'learninghypernyms'):
            self.relation = relation
            val_to_relation = {0: 'random', 1: self.relation}
            self.df.columns = ['word1', 'word2', 'relation']
            self.df.relation = self.df.relation.apply(lambda x: val_to_relation[x])
        elif(self.dataset == 'ROOT9' or self.dataset == 'Baroni'):
            self.df.columns = ['word1', 'relation', 'word2']
            self.df.word1 = self.df.word1.apply(lambda x: x.rsplit('-', 1)[0])
            self.df.word2 = self.df.word2.apply(lambda x: x.rsplit('-', 1)[0])
        elif(self.dataset == 'EVALution'):
            self.df.columns = ['word1', 'relation', 'word2']
        self.embed_tokens = list()
        self.is_en = is_en
        self.is_full_test = is_full_test
        if(self.is_en):
            for t in self.embedding_data:
                token, tag = t.rsplit('/', 1)
                if(tag in ['NN', 'NNP', 'NNS']):
                    self.embed_tokens.append(token)
        else:        
            self.embed_tokens = list(self.embedding_data.keys())
        if(not self.is_full_test):
            print('Working on sub-set')
            self.df_prep = self.df[self.df['word1'].isin(self.embed_tokens) & self.df['word2'].isin(self.embed_tokens)]
        else:
            print('Working on full-set')
            self.df_prep = self.df

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
        self.df_prep = self.df_prep[self.df_prep.relation.isin(comparison_relations)]

        def mapper(x):
            if(x in self.relations1):
                return 0
            elif(x in self.relations2):
                return 1
        print('relations1', self.relations1)
        print('relations2', self.relations2)
        self.df_prep.relation = self.df_prep.relation.apply(lambda x: mapper(x))
    
    def get_prep_dataset(self):
        return self.df_prep

    def replace_token_with_embed(self):
        if(self.is_en):
            def get_noun_embedding(token):
                if(self.fasttext_path is not None):
                    if(token in self.fasttext_embedding_data.keys()):
                        return self.fasttext_embedding_data[token]
                    return 300*[0.0] # standard dimension of fasttext
                else:
                    embed_tokens_with_tags = self.embedding_data.keys()
                    token_with_tags = [token + '/NN', token + '/NNP', token + '/NNS']
                    for token_with_tag in token_with_tags:
                        if(token_with_tag in embed_tokens_with_tags):
                            return self.embedding_data[token_with_tag] 
                    return 300*[0.0] # standard dimension of fasttext
            word1_embeddings = np.array(self.df_prep.word1.apply(lambda x: get_noun_embedding(x)).tolist())
            word2_embeddings = np.array(self.df_prep.word2.apply(lambda x: get_noun_embedding(x)).tolist())
        else:
            if(self.fasttext_path is not None):
                token_to_embedding = self.fasttext_embedding_data
            else:
                token_to_embedding = self.embedding_data
            word1_embeddings = np.array(self.df_prep.word1.apply(lambda x: token_to_embedding[x] if x in self.embed_tokens else 300*[0.0]).tolist())
            word2_embeddings = np.array(self.df_prep.word2.apply(lambda x: token_to_embedding[x] if x in self.embed_tokens else 300*[0.0]).tolist())
        encoded_relation = np.array(self.df_prep.relation.tolist())
        return (word1_embeddings, word2_embeddings, encoded_relation)

    def hybrid_replace_token_with_embed(self):
        assert(self.hybrid_model is not None)
        assert(self.fasttext_path is not None)
        if(self.is_en):
            def get_noun_embedding(token):
                embed_tokens_with_tags = self.embedding_data.keys()
                token_with_tags = [token + '/NN', token + '/NNP', token + '/NNS']
                for token_with_tag in token_with_tags:
                    if(token_with_tag in embed_tokens_with_tags):
                        return self.embedding_data[token_with_tag]
                if(token in self.fasttext_embedding_data.keys()):
                    return self.fasttext_embedding_data[token]
                return 300*[0.0]
            word1_embeddings = np.array(self.df_prep.word1.apply(lambda x: get_noun_embedding(x)).tolist())
            word2_embeddings = np.array(self.df_prep.word2.apply(lambda x: get_noun_embedding(x)).tolist())
        else:
            def get_embedding(token):
                if(token in self.embedding_data.keys()):
                    return self.embedding_data[token]
                if(token in self.fasttext_embedding_data.keys()):
                    return self.fasttext_embedding_data[token]
                return 300*[0.0]
            word1_embeddings = np.array(self.df_prep.word1.apply(lambda x: get_embedding(x)).tolist())
            word2_embeddings = np.array(self.df_prep.word2.apply(lambda x: get_embedding(x)).tolist())
        encoded_relation = np.array(self.df_prep.relation.tolist())
        return (word1_embeddings, word2_embeddings, encoded_relation)
    
    def perform_operation(self, operation):
        if(self.hybrid_model is None):
            print(f'Using standard model')
            word1_embeddings, word2_embeddings, encoded_relation = self.replace_token_with_embed()
        else:
            print(f'Using hybrid model: {self.hybrid_model}')
            word1_embeddings, word2_embeddings, encoded_relation = self.hybrid_replace_token_with_embed()
        assert(word1_embeddings.shape == word2_embeddings.shape)
        if(operation == 'DIFF'):
            feature = word1_embeddings - word2_embeddings
        elif(operation == 'MULT'):
            feature = np.multiply(word1_embeddings, word2_embeddings)
        elif(operation == 'ADD'):
            feature = word1_embeddings + word2_embeddings
        elif(operation == 'CAT'):
            feature = np.concatenate((word1_embeddings, word2_embeddings), axis=1)
        return (feature, encoded_relation)
    
    def perform_sampling(self, operation, sampler):
        feature, target = self.perform_operation(operation=operation)
        if(sampler == 'under'):
            rus = RandomUnderSampler(random_state=random_state)
            feature, target = rus.fit_sample(feature, target)
        target = pd.Series(target)
        print('Sampled distribution')
        print(target.value_counts())
        return (feature, target)
        

def train_model(model_name, dataset, operation, sampler, scoring, random_state):
    feature, target = dataset.perform_sampling(operation, sampler)
    print(model_name, operation)
    if(model_name == 'SVM'):
        clf = LinearSVC(random_state=random_state, max_iter=100000)
    elif(model_name == 'RBF-SVM'):
        clf = SVC(kernel='rbf')
    elif(model_name == 'RF'):
        clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    scores = cross_val_score(estimator=clf, X=feature, y=target, scoring=scoring, n_jobs=-1, cv=10)
    print(f'{scoring}: {scores.mean()} +/- {scores.std() ** 2}')

def train_on_custom_split(model_name, train_dataset, test_dataset, operation, sampler, scoring, random_state):
    print('Training distribution')
    train_feature, train_target = train_dataset.perform_sampling(operation, sampler)
    print('Testing distribution')
    test_feature, test_target = test_dataset.perform_sampling(operation, sampler)
    print(model_name, operation)
    if(model_name == 'SVM'):
        clf = LinearSVC(random_state=random_state, max_iter=100000)
    elif(model_name == 'RBF-SVM'):
        clf = SVC(kernel='rbf')
    elif(model_name == 'RF'):
        clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    clf.fit(train_feature, train_target)
    print(f'acuracy: {clf.score(test_feature, test_target)}')
    


if __name__ == '__main__':
    random_state = 42
    np.random.seed(random_state)
    models = ['SVM', 'RF', 'RBF-SVM']
    operations = ['DIFF', 'MULT', 'ADD', 'CAT']

    parser = argparse.ArgumentParser(description='train the classifier for semantic relation classification task')
    parser.add_argument('--input_token_emd_path',
                        required=True,
                        help='input path to token.emd')
    parser.add_argument('--fasttext_path',
                        help='optional path to fasttext embedding')
    parser.add_argument('--input_dataset',
                        required=True,
                        choices=['bless-gems', 'ROOT9', 'learninghypernyms', 'EVALution', 'Baroni'],
                        help='input dataset type')
    parser.add_argument('--relation',
                        choices=['coord', 'hyper'],
                        help = 'to be used only with learninghypernyms dataset')
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
    parser.add_argument('--is_full_test',
                        action='store_true',
                        default=False,
                        help='enable if full dataset is to considered')
    parser.add_argument('--exp_on_non_overlap_split', action='store_true')
    parser.add_argument('--test_split')
    parser.add_argument('--hybrid_model')
    args = parser.parse_args()

    if(args.hybrid_model):
        assert(args.fasttext_path is not None)

    if(args.input_dataset == 'learninghypernyms'):
        assert(args.relation is not None)

    if(args.model_name == 'RBF-SVM'):
        assert(args.operation == 'DIFF')

    if(not args.exp_on_non_overlap_split):
  	    dataset = Dataset(dataset=args.input_dataset, 
  	                        relation=args.relation,
  	                        file_path=args.input_dataset_path, 
  	                        embedding_path=args.input_token_emd_path,
  	                        relations1=args.relations1,
  	                        relations2=args.relations2,
  	                        is_en=args.is_en,
  	                        is_full_test=args.is_full_test,
                            hybrid_model=args.hybrid_model,
  	                        random_state=random_state,
  	                        fasttext_path=args.fasttext_path)
  	    print('Number of datapoints in the (imbalanced) dataset:', dataset.get_prep_dataset().shape[0])
  	    if(args.model_name == 'ALL' and args.operation == 'ALL'):
  	        for model_name in models:
  	            for operation in operations:
  	                train_model(model_name=model_name, dataset=dataset, operation=operation, sampler=args.sampler, scoring=args.scoring, random_state=random_state)
  	    else:
  	        train_model(model_name=args.model_name, dataset=dataset, operation=args.operation, sampler=args.sampler, scoring=args.scoring, random_state=random_state)
    else:
        assert(args.test_split is not None)
        print('Loading training dataset')
        train_dataset = Dataset(dataset=args.input_dataset,
  	                        relation=args.relation,
  	                        file_path=args.input_dataset_path, 
  	                        embedding_path=args.input_token_emd_path,
  	                        relations1=args.relations1,
  	                        relations2=args.relations2,
  	                        is_en=args.is_en,
  	                        is_full_test=args.is_full_test,
                            hybrid_model=args.hybrid_model,
  	                        random_state=random_state,
  	                        fasttext_path=args.fasttext_path)
        print('Loaded training dataset')
        print('Number of datapoint in the (imbalanced) training dataset:', train_dataset.get_prep_dataset().shape[0])
        print('Loading testing dataset')
        test_dataset = Dataset(dataset=args.input_dataset,
  	                        relation=args.relation,
  	                        file_path=args.test_split, 
  	                        embedding_path=args.input_token_emd_path,
  	                        relations1=args.relations1,
  	                        relations2=args.relations2,
  	                        is_en=args.is_en,
  	                        is_full_test=args.is_full_test,
                            hybrid_model=args.hybrid_model,
  	                        random_state=random_state,
  	                        fasttext_path=args.fasttext_path)
        print('Loaded testing dataset')
        print('Number of datapoint in the (imbalanced) testing dataset:', test_dataset.get_prep_dataset().shape[0])
        if(args.model_name == 'ALL' and args.operation == 'ALL'):
  	        for model_name in models:
  	            for operation in operations:
  	                train_on_custom_split(model_name=model_name, train_dataset=train_dataset, test_dataset=test_dataset, operation=operation, sampler=args.sampler, scoring=args.scoring, random_state=random_state)
        else:
            train_on_custom_split(model_name=args.model_name, train_dataset=train_dataset, test_dataset=test_dataset, operation=args.operation, sampler=args.sampler, scoring=args.scoring, random_state=random_state)
