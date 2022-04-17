import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from transformers import BertModel, BertTokenizer
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier

import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"

class HfBertClassifierModel(nn.Module):
    def __init__(self, n_classes, weights_name='bert-base-cased'):
        super().__init__()
        self.n_classes = n_classes
        self.weights_name = weights_name
        self.bert = BertModel.from_pretrained(self.weights_name)
        self.bert.train()
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim
        # adding in a new and redundant layer just to test torch interventions
        self.extra_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        # The only new parameters -- the classifier:
        self.classifier_layer = nn.Linear(
            self.hidden_dim, self.n_classes)
        self.layers = list(self.bert.children()) + [self.extra_layer, self.classifier_layer]

    # def forward(self, indices, mask):
    #     reps = self.bert(
    #         indices, attention_mask=mask)
    #     return self.classifier_layer(reps.pooler_output)

    def forward(self, X):
        # for the purpose of IIT, concatenating together the indices and the mask values
        split_index = X.shape[1] // 2
        X = X.to(torch.int32)
        indices, mask = X[:, :split_index], X[:, split_index:]
        reps = self.bert(
            indices, attention_mask=mask)
        return self.classifier_layer(self.extra_layer(reps.pooler_output))

class HfBertClassifier(TorchShallowNeuralClassifier):
    def __init__(self, weights_name='bert-base-cased', *args, **kwargs):
        self.weights_name = weights_name
        self.tokenizer = BertTokenizer.from_pretrained(self.weights_name)
        super().__init__(*args, **kwargs)
        self.params += ['weights_name']

    def build_graph(self):
        model = HfBertClassifierModel(self.n_classes_, self.weights_name)
        self.layers = model.layers
        return model

    def build_dataset(self, X, y=None):
        data = self.tokenizer.batch_encode_plus(
            X,
            max_length=None,
            add_special_tokens=True,
            padding='longest',
            return_attention_mask=True)
        indices = torch.tensor(data['input_ids'])
        mask = torch.tensor(data['attention_mask'])
        if y is None:
            dataset = torch.utils.data.TensorDataset(indices, mask)
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(indices, mask, y)
        return dataset
  