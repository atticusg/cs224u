import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch_bert_classifier import HfBertClassifier
import utils
from iit import IITModel

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


class TorchBertClassifierIIT(HfBertClassifier):
    def __init__(self,id_to_coords = None, **base_kwargs):
        super().__init__(**base_kwargs)
        loss_function= nn.CrossEntropyLoss(reduction="mean")
        self.loss = lambda preds, labels: loss_function(preds[0],labels[:,0]) + loss_function(preds[1],labels[:,1])
        self.id_to_coords = id_to_coords
        self.shuffle_train = False

    def build_graph(self):
        model = super().build_graph()
        IITmodel = IITModel(model, self.layers, self.id_to_coords, self.device)
        return IITmodel

    def batched_indices(self, max_len):
        batch_indices = [ x for x in range((max_len // self.batch_size))]
        output = []
        while len(batch_indices) != 0:
            batch_index = random.sample(batch_indices, 1)[0]
            batch_indices.remove(batch_index)
            output.append([batch_index*self.batch_size + x for x in range(self.batch_size)])
        return output
    
    # prepare data Bert tokenizer, inherited from BertClassifier
    # we should tokenize the base and the sources all at once, so that they
    # can have the same padding length!!
    def prepare_data(self, X):
        data = self.tokenizer.batch_encode_plus(
            X,
            max_length=None,
            add_special_tokens=True,
            padding='longest',
            return_attention_mask=True)
        indices = torch.tensor(data['input_ids'])
        mask = torch.tensor(data['attention_mask'])
        result = torch.cat([indices, mask], dim=1) # need to concatenate base and indices together to form single X
        return result

    def build_dataset(self, base, sources, base_y, IIT_y, coord_ids):
        """
        Define datasets for the model.

        Parameters
        ----------
        X : iterable of length `n_examples`
           Each element must have the same length.

        y: None or iterable of length `n_examples`

        Attributes
        ----------
        input_dim : int
            Set based on `X.shape[1]` after `X` has been converted to
            `np.array`.

        Returns
        -------
        torch.utils.data.TensorDataset` Where `y=None`, the dataset will
        yield single tensors `X`. Where `y` is specified, it will yield
        `(X, y)` pairs.

        """
        # combine base and source sentences to be tokenized together
        X = self.prepare_data(base + [sentence for source in sources for sentence in source])
        base_length = len(base)
        base = X[:base_length]
        # put sources back to lists of inputs, each of length equal to the base
        sources = [X[i:i + base_length] for i in range(base_length, len(X), base_length)]
        self.input_dim = base.shape[1] # should be shape of len(longest token sequence)
        coord_ids = torch.FloatTensor(np.array(coord_ids))

        # base = torch.FloatTensor(np.array(base))
        # sources = [torch.FloatTensor(np.array(source)) for source in sources]
        # self.input_dim = base.shape[1]
        # coord_ids = torch.FloatTensor(np.array(coord_ids))

        IIT_y = np.array(IIT_y)
        self.classes_ = sorted(set(IIT_y))
        self.n_classes_ = len(self.classes_)
        class2index = dict(zip(self.classes_, range(self.n_classes_)))
        IIT_y = [class2index[int(label)] for label in IIT_y]
        IIT_y = torch.tensor(IIT_y)

        base_y = np.array(base_y)
        self.classes_ = sorted(set(base_y))
        self.n_classes_ = len(self.classes_)
        class2index = dict(zip(self.classes_, range(self.n_classes_)))
        base_y = [class2index[label] for label in base_y]
        base_y = torch.tensor(base_y)

        bigX = torch.stack([base, coord_ids.unsqueeze(1).expand(-1, base.shape[1])] + sources, dim=1)
        bigy = torch.stack((IIT_y, base_y), dim=1)
        dataset = torch.utils.data.TensorDataset(bigX,bigy)
        return dataset

    def prep_input(self, base, sources, coord_ids):
        # convert coord_ids to tensor, in the case that it is passed in as a list
        if not isinstance(coord_ids, torch.FloatTensor):
            coord_ids = torch.FloatTensor(np.array(coord_ids))
        # tokenize base and sources together
        X = self.prepare_data(base + [sentence for source in sources for sentence in source])
        base_length = len(base)
        base = X[:base_length]
        # put sources back to lists of inputs, each of length equal to the base
        sources = [X[i:i + base_length] for i in range(base_length, len(X), base_length)]
        bigX = torch.stack([base, coord_ids.unsqueeze(1).expand(-1, base.shape[1])] + sources, dim=1)
        return bigX