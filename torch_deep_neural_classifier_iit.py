import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch_deep_neural_classifier import TorchDeepNeuralClassifier
import utils
from iit import IITModel

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


class TorchDeepNeuralClassifierIIT(TorchDeepNeuralClassifier):
    def __init__(self,id_to_coords = None, **base_kwargs):
        """
        A trainer for the Interchange Intervention Training (IIT) model, which 
        implements IIT training.
        """
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

    def build_dataset(self, base, sources, base_y, IIT_y, coord_ids):
        """
        Defines the dataset for the IITModel. The inputs are built from
        stacking the base input, location of intervention, and the input from
        the given sources. 

        Parameters
        ----------
        base : list of length n_example
            The base inputs for the model.

        sources : list of lists, of length n_example
            A collection of source inputs used to create counterfactuals 
            during IIT training.
        
        base_y : list of length n_example
            A list of the outputs for the high-level causal graph
            when computed on the base inputs.

        IIT_y : list of length n_example
            A list of the outputs for the high-level causal graph
            when computed on the base input, with an intervention by the source 
            inputs at the location of `coord_ids`.
        
        coord_ids : list of length n_example
            Location of intervention, which maps from a node in the high-level causal structure
            to parameters in the low-level neural network. 
            Note: all elements of `coord_ids` must be identical, because we are intervening on a single
            set of nodes.

        Returns
        -------
        torch.utils.data.TensorDataset` Where `y=None`, the dataset will
        yield single tensors `X`. Where `y` is specified, it will yield
        `(X, y)` pairs.
        X and y consist of stacking together (base, coord_ids, sources) and (base_y, IIT_y)
        respectively.
        """
        base = torch.FloatTensor(np.array(base))
        sources = [torch.FloatTensor(np.array(source)) for source in sources]
        self.input_dim = base.shape[1]
        coord_ids = torch.FloatTensor(np.array(coord_ids))

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
        bigX = torch.stack([base, coord_ids.unsqueeze(1).expand(-1, base.shape[1])] + sources, dim=1)
        return bigX



if __name__ == '__main__':
    simple_example()
