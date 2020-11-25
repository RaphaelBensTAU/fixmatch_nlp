from eda_nlp.code import eda
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch 
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from bert import BertBinaryClassifier
from torch.optim import Adam
from torch import nn
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output

#HYPERPARAMETERS
#AUGMENTATION
WEAK_SR = 0.1
WEAK_RI = 0.1
WEAK_SR = 0.0
ERAK_RD = 0.0

STRONG_SR = 0.2
STRONG_RI = 0.2
STRONG_SR = 0.0
STRONG_RD = 0.1

#EXPERIMENT
NUM_OF_OVERALL_TRAIN = 2000
NUM_OF_TEST = 500
NUM_OF_LABELED_TRAIN = 1000

NUM_OF_EPOCHS = 1
LABELED_BATCH_SIZE = 2
UNLABELED_BATCH_SIZE = 2
UNLABELED_LOSS_WEIGHT = 10

ONE_HOT_THRESHOLD = 0.6

weak_augment = lambda x : augment_data(x, WEAK_SR, WEAK_SI, WEAK_RS, WEAK_RD, 1)
strong_augment = lambda x : augment_data(x, STRONG_SR, STRONG_SI, STRONG_RS, STRONG_RD, 1)

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def augment_data(lst_of_sentences, alpha_sr, alpha_ri, alpha_rs, p_rd, num_aug):
    return np.array([eda.eda(sentence, alpha_sr, alpha_ri, alpha_rs, p_rd, num_aug) for sentence in lst_of_sentences])

#base line, take data from beginning
def prepare_data(train_data, test_data, num_of_train, num_of_test, num_of_labeled_train):
    out = {}
    train = train_data[:num_of_train].to_dict(orient='records')
    test = test_data[:num_of_test].to_dict(orient='records')
    train_texts, train_labels = list(zip(*map(lambda d: (d['text'], d['sentiment']), train)))
    out['test_texts'], out['test_labels'] = list(zip(*map(lambda d: (d['text'], d['sentiment']), test)))
    out['test_labels'] = np.array(out['test_labels']) == "pos"
    out['train_labeled'] = np.array(train_texts[:num_of_labeled_train])
    out['train_unlabeled'] = np.array(train_texts[num_of_labeled_train:])
    out['train_labels'] = np.array(train_labels[:num_of_labeled_train]) == "pos"
    return out

def get_ids_and_masks(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], data))
    tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, tokens)), maxlen=512, truncating="post", padding="post", dtype="int")
    masks = [[float(i > 0) for i in ii] for ii in tokens_ids]
    return torch.tensor(tokens_ids).cuda(), torch.tensor(masks).cuda()

def create_binary_one_hot_dist(logits, threshold):
    indexes = []
    pass_threshold1_ind = (logits > threshold) == True
    pass_threshold0_ind = (logits < 1 - threshold) == True
    for i in range(len(pass_threshold1_ind)):
        if pass_threshold0_ind[i]:
            indexes.append(0)
        elif pass_threshold1_ind[i]:
            indexes.append(1)
        else:
            indexes.append(-1)
    return np.array(indexes)

out = prepare_data(train_data, test_data, NUM_OF_OVERALL_TRAIN, NUM_OF_TEST, NUM_OF_LABELED_TRAIN)

train_l = np.concatenate((out['train_labeled'][:,None], out['train_labels'][:,None]),axis = 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_clf = BertBinaryClassifier()
bert_clf = bert_clf.cuda()     # running BERT on CUDA_GPU
param_optimizer = list(bert_clf.sigmoid.named_parameters()) 
optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(bert_clf.parameters(), lr=3e-6)

torch.cuda.empty_cache()   # Clearing Cache space for a fresh Model run

for ep in range(NUM_OF_EPOCHS):
    np.random.shuffle(train_l)
    np.random.shuffle(out['train_unlabeled'])

    for batch_num in range(len(train_l)//LABELED_BATCH_SIZE):
        labeled_train_batch = train_l[batch_num * LABELED_BATCH_SIZE: batch_num * LABELED_BATCH_SIZE + LABELED_BATCH_SIZE, 0]
        labeled_train_labels = torch.from_numpy((train_l[batch_num * LABELED_BATCH_SIZE: batch_num * LABELED_BATCH_SIZE + LABELED_BATCH_SIZE, 1] == 'True').reshape(-1,1)).float().cuda()

        unlabeled_train_batch = out['train_unlabeled'][batch_num * UNLABELED_BATCH_SIZE: batch_num * UNLABELED_BATCH_SIZE + UNLABELED_BATCH_SIZE]

        weakly_augmented_labeled = weak_augment(labeled_train_batch)
        weakly_augmented_labeled_tokens_ids, weakly_augmented_labeled_masks = get_ids_and_masks(weakly_augmented_labeled.reshape((weakly_augmented_labeled.shape[0])))

        logits_weakly_augmented_labeled = bert_clf(weakly_augmented_labeled_tokens_ids, weakly_augmented_labeled_masks)
        loss_func = nn.BCELoss()
        loss = loss_func(logits_weakly_augmented_labeled, labeled_train_labels).mean()
        
        weakly_augmented_unlabeled = weak_augment(unlabeled_train_batch)
        weakly_augmented_unlabeled_tokens_ids, weakly_augmented_unlabeled_masks = get_ids_and_masks(weakly_augmented_unlabeled.reshape((weakly_augmented_unlabeled.shape[0])))

        strongly_augmented_unlabeled = strong_augment(unlabeled_train_batch)
        strongly_augmented_unlabeled_tokens_ids, strongly_augmented_unlabeled_masks = get_ids_and_masks(strongly_augmented_unlabeled.reshape((strongly_augmented_unlabeled.shape[0])))

        logits_weakly_augmented_unlabeled = bert_clf(weakly_augmented_unlabeled_tokens_ids, weakly_augmented_unlabeled_masks)
        logits_strongly_augmented_unlabeled = bert_clf(strongly_augmented_unlabeled_tokens_ids, strongly_augmented_unlabeled_masks)
        
        one_hot_indexes = create_binary_one_hot_dist(logits_weakly_augmented_unlabeled, ONE_HOT_THRESHOLD)
        pass_indexes = one_hot_indexes != -1
        one_hot_indexes = one_hot_indexes[pass_indexes]
        if one_hot_indexes.shape[0] > 0:
            one_hot_indexes = torch.tensor(one_hot_indexes.reshape(one_hot_indexes.shape[0], 1)).float().cuda()
            thresholded_logits_strongly_augmented_unlabeled = logits_strongly_augmented_unlabeled[pass_indexes]
            loss += UNLABELED_LOSS_WEIGHT * loss_func(thresholded_logits_strongly_augmented_unlabeled, one_hot_indexes).sum()/logits_weakly_augmented_unlabeled.shape[0]

        bert_clf.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters=bert_clf.parameters(), max_norm=1.0)
        optimizer.step()
        
        clear_output(wait=True)
        print('Epoch: ', ep + 1)
        print("\r" + "{0}/{1} loss: {2} ".format(batch_num, len(train_l) / LABELED_BATCH_SIZE, loss / (batch_num + 1)))
        
