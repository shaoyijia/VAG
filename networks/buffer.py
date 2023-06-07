import pdb
from typing import Tuple

import numpy as np
import torch
from torchvision import transforms


class FixedSizeBuffer:
    """
    The memory buffer of rehearsal method. We save a fixed percentage of the training set.

    We keep the buffer in CPU memory.
    """

    def __init__(self, buffer_size):
        # For fixed size buffer, the buffer_size should be in (0,1), indicating the percentage of stored data.
        self.buffer_size = buffer_size
        self.attributes = ['examples', 'labels', 'logits', 'attention_mask', 'decoder_input_ids', 'masked_vocabulary']
        self.examples = None
        self.labels = None
        self.logits = None
        self.attention_mask = None
        self.decoder_input_ids = None
        self.masked_vocabulary = None
        self.num_seen_examples = 0

    def get_size(self):
        return self.num_seen_examples

    def add_data(self, examples, labels=None, logits=None, attention_mask=None, decoder_input_ids=None,
                 masked_vocabulary=None):

        selection = torch.bernoulli(torch.ones(examples.shape[0]) * self.buffer_size)
        indices = torch.nonzero(selection, as_tuple=False).squeeze(dim=-1)
        try:
            if len(indices) == 0:
                return
        except Exception as e:
            pdb.set_trace()
        examples = examples.cpu()[indices, :]
        if labels is not None:
            if len(labels.shape) == 1:
                labels = labels.cpu()[indices]
            else:
                labels = labels.cpu()[indices, :]
        attention_mask = attention_mask.cpu()[indices, :] if attention_mask is not None else None
        decoder_input_ids = decoder_input_ids[indices, :] if decoder_input_ids is not None else None
        masked_vocabulary = masked_vocabulary.cpu()[indices, :] if masked_vocabulary is not None else None

        if examples.shape[0] > 0:
            if self.examples is None:
                self.examples = examples
                self.labels = labels
                self.logits = logits
                self.attention_mask = attention_mask
                self.decoder_input_ids = decoder_input_ids
                self.masked_vocabulary = masked_vocabulary
            else:
                self.examples = torch.cat([self.examples, examples], dim=0)
                self.labels = torch.cat([self.labels, labels], dim=0) if self.labels is not None else None
                self.logits = torch.cat([self.logits, logits], dim=0) if self.logits is not None else None
                self.attention_mask = \
                    torch.cat([self.attention_mask, attention_mask], dim=0) if self.attention_mask is not None else None
                self.decoder_input_ids = torch.cat([self.decoder_input_ids, decoder_input_ids],
                                                   dim=0) if self.decoder_input_ids is not None else None
                self.masked_vocabulary = torch.cat([self.masked_vocabulary, masked_vocabulary],
                                                   dim=0) if self.masked_vocabulary is not None else None

            self.num_seen_examples += examples.shape[0]

    def get_data(self, size: int, transform: transforms = None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                if attr is not None:
                    ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples]),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
