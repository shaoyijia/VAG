"""LDBR implementation adapted from
https://github.com/SALT-NLP/IDBR/blob/main/src/train.py"""
import json
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class Predictor(torch.nn.Module):
    def __init__(self, num_class, hidden_size):
        super(Predictor, self).__init__()

        self.num_class = num_class

        self.dis = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, self.num_class)
        )

    def forward(self, z):
        return self.dis(z)


def process_array(lst):
    return [x.tolist() for x in lst]


def process_int(lst):
    return [int(x) for x in lst]


def process_lst(lst):
    return [np.array(x) for x in lst]


class Memory(object):
    def __init__(self):
        self.examples = []
        self.masks = []
        self.labels = []
        self.tasks = []
        self.features = []

    def append(self, example, mask, label, task):
        self.examples.append(example)
        self.masks.append(mask)
        self.labels.append(label)
        self.tasks.append(task)

    def save(self, path):
        with open(path, 'w') as f:
            obj = [self.examples, self.masks, self.labels, self.tasks, self.features]
            obj = [process_array(x) if i not in [2, 3] else process_int(x) for i, x in enumerate(obj)]
            json.dump(obj, f)

    def load(self, path):
        with open(path, 'r') as f:
            self.examples, self.masks, self.labels, self.tasks, self.features = json.load(f)
            self.examples = process_lst(self.examples)
            self.masks = process_lst(self.masks)
            # self.labels = process_lst(self.labels)
            # self.tasks = process_lst(self.tasks)
            self.features = process_lst(self.features)

    def store_features(self, model):
        """
        Args:
            model: The model trained just after previous task
        Returns: None
        store previous features before trained on new class
        """
        self.features = []
        length = len(self.labels)
        model.eval()
        with torch.no_grad():
            for i in range(length):
                x = torch.tensor(self.examples[i]).view(1, -1).cuda()
                mask = torch.tensor(self.masks[i]).view(1, -1).cuda()
                outputs = model(input_ids=x, attention_mask=mask)
                g_fea = outputs.total_g_fea
                s_fea = outputs.total_s_fea
                fea = torch.cat([g_fea, s_fea], dim=1).view(-1).data.cpu().numpy()
                self.features.append(fea)
        print(len(self.features))
        print(len(self.labels))

    def get_random_batch(self, batch_size, task_id=None):
        if task_id is None:
            permutations = np.random.permutation(len(self.labels))
            index = permutations[:batch_size]
            index_length = len(index)
            if index_length < batch_size:  # important for batch index if not enough examples saved !!!
                aug_times = math.ceil(batch_size / index_length)
                index *= aug_times
            mini_examples = [self.examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
            mini_features = [self.features[i] for i in index]
        else:
            index = [i for i in range(len(self.labels)) if self.tasks[i] == task_id]
            index_length = len(index)
            if index_length < batch_size:  # important for batch index if not enough examples saved !!!
                aug_times = math.ceil(batch_size / index_length)
                index *= aug_times
            np.random.shuffle(index)
            index = index[:batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
            mini_features = [self.features[i] for i in index]
        return torch.tensor(mini_examples), torch.tensor(mini_masks), torch.tensor(mini_labels), \
               torch.tensor(mini_tasks), torch.tensor(mini_features)

    def get_minibatch(self, batch_size):
        length = len(self.labels)
        permutations = np.random.permutation(length)
        for s in range(0, length, batch_size):
            if s + batch_size >= length:
                break
            index = permutations[s:s + batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
            mini_features = [self.features[i] for i in index]
            yield torch.tensor(mini_examples), torch.tensor(mini_masks), torch.tensor(mini_labels), \
                  torch.tensor(mini_tasks), torch.tensor(mini_features)

    def __len__(self):
        return len(self.labels)


def random_seq(src):
    # adding [SEP] to unify the format of samples for NSP
    batch_size = src.size(0)
    length = src.size(1)
    dst = []
    for i in range(batch_size):
        cur = src[i]
        first_pad = (cur.tolist() + [1]).index(1)
        cur = cur[1:first_pad].tolist()
        cur = random_string(cur)
        if length - len(cur) - 1 < 0:
            # import pdb
            # pdb.set_trace()
            cur = cur[len(cur) + 1 - length:]
            padding = []
        else:
            padding = [1] * (length - len(cur) - 1)  # For BART, pad_token_id = 1
        dst.append(torch.tensor([0] + cur + padding))
    return torch.stack(dst).cuda()


def random_string(str):
    # randomly split positive samples into two halves and add [SEP] between them
    str.remove(2)
    if 2 in str:
        str.remove(2)

    len1 = len(str)
    try:
        if len1 == 1:
            cut = 1
        else:
            cut = np.random.randint(1, len1)
    except Exception as e:
        import pdb
        pdb.set_trace()
    str = str[:cut] + [2] + str[cut:] + [2]
    return str


def change_string(str):
    # creating negative samples for NSP by randomly splitting positive samples
    # and swapping two halves
    str.remove(2)
    if 2 in str:
        str.remove(2)

    len1 = len(str)
    if len1 == 1:
        cut = 1
    else:
        cut = np.random.randint(1, len1)
    str = str[cut:] + [2] + str[:cut] + [2]
    return str


def get_permutation_batch(src, src_mask):
    # create negative samples for Next Sentence Prediction
    batch_size = src.size(0)
    length = src.size(1)
    dst = []
    dst_mask = []
    lbl = []
    for i in range(batch_size):
        cur = src[i]
        mask = src_mask[i].tolist()
        first_pad = (cur.tolist() + [1]).index(1)
        cur = cur[1:first_pad].tolist()
        cur = change_string(cur)
        lbl.append(1)

        if length - len(cur) - 1 < 0:
            # import pdb
            # pdb.set_trace()
            cur = cur[len(cur) + 1 - length:]
            padding = []
        else:
            padding = [1] * (length - len(cur) - 1)  # For BART, pad_token_id = 1
        dst.append(torch.tensor([0] + cur + padding))
        dst_mask.append(torch.tensor(mask))
    return torch.stack(dst).cuda(), torch.stack(dst_mask).cuda(), torch.tensor(lbl).cuda()


def train_step(model, x, mask, y, t, task_id, replay, x_feature, predictor):
    cls_CR = nn.CrossEntropyLoss()
    nsp_CR = nn.CrossEntropyLoss()
    batch_size = x.size(0)
    x = random_seq(x)
    pre_lbl = None

    p_x, p_mask, p_lbl = get_permutation_batch(x, mask)
    x = torch.cat([x, p_x], dim=0)
    mask = torch.cat([mask, p_mask], dim=0)
    if mask.shape[1] < x.shape[1]:
        mask = torch.concat([mask, torch.ones(mask.shape[0], x.shape[1] - mask.shape[1]).to(mask.device)], dim=1)
    r_lbl = torch.zeros_like(p_lbl)
    nsp_lbl = torch.cat([r_lbl, p_lbl], dim=0)

    y = torch.cat([y, y], dim=0)
    t = torch.cat([t, t], dim=0)

    outputs = model(input_ids=x, attention_mask=mask)
    total_g_fea, total_s_fea, cls_pred, task_pred = outputs.total_g_fea, \
                                                    outputs.total_s_fea, outputs.logits, outputs.task_pred

    g_fea = total_g_fea[:batch_size, :]
    s_fea = total_s_fea[:batch_size, :]

    # Calculate classification loss
    _, pred_cls = cls_pred.max(1)
    correct_cls = pred_cls.eq(y.view_as(pred_cls)).sum().item()
    cls_loss = cls_CR(cls_pred, y)

    task_loss = torch.tensor(0.0).cuda()
    reg_loss = torch.tensor(0.0).cuda()
    nsp_loss = torch.tensor(0.0).cuda()

    # Calculate regularization loss
    if x_feature is not None:
        fea_len = g_fea.size(1)
        g_fea = g_fea[:batch_size, :]
        s_fea = s_fea[:batch_size, :]
        old_g_fea = x_feature[:, :fea_len]
        old_s_fea = x_feature[:, fea_len:]

        # use hyperparameters from ldbr code
        reg_loss += 0.5 * torch.nn.functional.mse_loss(s_fea, old_s_fea) + \
                    0.5 * torch.nn.functional.mse_loss(g_fea, old_g_fea)

        if replay and task_id > 0:
            reg_loss *= 5.0
        elif not replay and task_id > 0:
            reg_loss *= 0.5
        elif task_id == 0:
            reg_loss *= 0.0  # no reg loss on the 1st task

    # Calculate task loss only when in replay batch
    task_pred = task_pred[:, :task_id + 1]
    _, pred_task = task_pred.max(1)
    correct_task = pred_task.eq(t.view_as(pred_task)).sum().item()
    if task_id > 0 and replay:
        task_loss += 1.0 * cls_CR(task_pred, t)

    # Calculate Next Sentence Prediction loss
    nsp_acc = 0.0
    nsp_output = predictor(total_g_fea)
    nsp_loss += 1.0 * nsp_CR(nsp_output, nsp_lbl)

    _, nsp_pred = nsp_output.max(1)
    nsp_correct = nsp_pred.eq(nsp_lbl.view_as(nsp_pred)).sum().item()
    nsp_acc = nsp_correct * 1.0 / (batch_size * 2.0)

    loss = cls_loss + task_loss + reg_loss + nsp_loss

    return loss


def select_samples_to_store(model, buffer, data_loader, task_id, store_ratio):
    """add examples to memory"""
    x_list = []
    mask_list = []
    y_list = []
    fea_list = []

    model.eval()
    with torch.no_grad():
        for inputs in data_loader:
            x = inputs['input_ids']
            mask = inputs['attention_mask']
            y = inputs['labels']
            x = x.cuda()
            mask = mask.cuda()
            y = y.cuda()
            outputs = model(**inputs)
            sentence_embedding = outputs.sentence_embedding
            x_list.append(x.to("cpu"))
            mask_list.append(mask.to("cpu"))
            y_list.append(y.to("cpu"))
            # Kmeans on bert embedding
            fea_list.append(sentence_embedding.to("cpu"))
    x_list = torch.cat(x_list, dim=0).data.cpu().numpy()
    mask_list = torch.cat(mask_list, dim=0).data.cpu().numpy()
    y_list = torch.cat(y_list, dim=0).data.cpu().numpy()
    fea_list = torch.cat(fea_list, dim=0).data.cpu().numpy()

    n_clu = int(store_ratio * len(x_list))
    estimator = KMeans(n_clusters=n_clu, random_state=2021)
    estimator.fit(fea_list)
    label_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    for clu_id in range(n_clu):
        index = [i for i in range(len(label_pred)) if label_pred[i] == clu_id]
        closest = float("inf")
        closest_x = None
        closest_mask = None
        closest_y = None
        for j in index:
            dis = np.sqrt(np.sum(np.square(centroids[clu_id] - fea_list[j])))
            if dis < closest:
                closest_x = x_list[j]
                closest_mask = mask_list[j]
                closest_y = y_list[j]
                closest = dis

        if closest_x is not None:
            buffer.append(closest_x, closest_mask, closest_y, task_id)

    print("Buffer size:{}".format(len(buffer)))
    print(buffer.labels)
    b_lbl = np.unique(buffer.labels)
    for i in b_lbl:
        print("Label {} in Buffer: {}".format(i, buffer.labels.count(i)))
