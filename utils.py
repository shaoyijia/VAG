import json
from typing import Any, NewType, Optional

import numpy as np
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from transformers import BartModel, BartConfig

from networks.bart_retrieve import BartWithLabelRetriever
from networks.l2p_model import L2PBartForSequenceClassification
from networks.my_bart_model import MyBartForSequenceClassification, MyBart


def load_json(file_name, encoding="utf-8"):
    with open(file_name, 'r', encoding=encoding) as f:
        content = json.load(f)
    return content


def dump_json(obj, file_name, encoding="utf-8", default=None):
    if default is None:
        with open(file_name, 'w', encoding=encoding) as fw:
            json.dump(obj, fw)
    else:
        with open(file_name, 'w', encoding=encoding) as fw:
            json.dump(obj, fw, default=default)


def num_params(model: Module):
    total_params, trainable_params = [], []
    for param in model.parameters():
        total_params.append(param.nelement())
        if param.requires_grad:
            trainable_params.append(param.nelement())

    return {
        'total': sum(total_params),
        'trainable': sum(trainable_params)
    }


def print_model_report(model):
    print('-' * 100)
    print(model)
    print('Dimensions =', end=' ')
    count = 0
    for p in model.parameters():
        print(p.size(), end=' ')
        count += np.prod(p.size())
    print()
    print('Num parameters = %s' % (human_format(count)))
    print('-' * 100)

    with open('para', 'a') as clocker_file:
        clocker_file.writelines((human_format(count)).replace('M', '') + '\n')

    return count


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim, '=', end=' ')
        opt = optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n + ':', opt[n], end=', ')
        print()
    return


# default `log_dir` is "runs" - we'll be more specific here
def setup_writer(name):
    writer = SummaryWriter(name)
    return writer


def log_loss(writer, loss_name='training loss', scalar_value=None, global_step=None):
    # ...log the running loss
    writer.add_scalar(loss_name, scalar_value, global_step=global_step)


########################################################################################################################


InputDataClass = NewType("InputDataClass", Any)


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result


def prepare_sequence_finetune(args):
    """Prepare a sequence of tasks for class-incremental learning."""
    with open(args.sequence_file.replace('_reduce', ''), 'r') as f:
        datas = f.readlines()[args.idrandom]
        data = datas.split()

    args.task_name = data

    if 'banking77' in args.sequence_file:
        args.ntasks = 7
        args.dataset_name = args.sequence_file.split('/')[-1]
    elif 'clinc150' in args.sequence_file:
        args.ntasks = 15
        args.dataset_name = args.sequence_file.split('/')[-1]
    elif '20news' in args.sequence_file:
        args.ntasks = 10
        args.dataset_name = args.sequence_file.split('/')[-1]
    elif 'fewrel' in args.sequence_file:
        args.ntasks = 8
        args.dataset_name = args.sequence_file.split('/')[-1]
    elif 'tacred' in args.sequence_file:
        args.ntasks = 8
        args.dataset_name = args.sequence_file.split('/')[-1]
    else:
        raise NotImplementedError('The current dataset is not supported!')

    if args.classifier_lr is None:
        args.classifier_lr = args.learning_rate

    if 'ewc' in args.baseline:
        args.lamb = 5000  # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000 for ewc

    output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[args.ft_task]) + "_model/"
    ckpt = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[args.ft_task - 1]) + "_model/"

    if args.ft_task > 0 and 'mtl' not in args.baseline:
        args.prev_output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
            args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[args.ft_task - 1]) + "_model/"
    else:
        args.prev_output = ''
    args.task = args.ft_task

    args.output_dir = output

    args.saved_output_dir = [args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[t]) + "_model/" for t in range(args.ft_task + 1)]

    if args.task == 0:  # Load the pre-trained model.
        if 'bart-base' in args.baseline:
            args.model_name_or_path = 'facebook/bart-base'  # Use the local backup.
        elif 'bart-large' in args.baseline:
            args.model_name_or_path = 'facebook/bart-large'
        else:
            raise NotImplementedError('Currently, we only support BART as the backbone model.')

    else:
        args.model_name_or_path = ckpt

    print('saved_output_dir: ', args.saved_output_dir)
    print('output_dir: ', args.output_dir)
    print('prev_output: ', args.prev_output)
    print('args.dataset_name: ', args.dataset_name)
    print('args.model_name_or_path: ', args.model_name_or_path)

    return args


def lookfor_model_finetune(args, taskcla):
    """Prepare the model for class-incremental learning."""
    if 'bart' in args.baseline:
        if 'retrieve' in args.baseline:
            model = BartWithLabelRetriever(args)

        elif 'distill' in args.baseline or 'ewc' in args.baseline:
            model = MyBartForSequenceClassification.from_pretrained(
                args.model_name_or_path, num_labels=taskcla, args=args)
            teacher = BartModel.from_pretrained(args.model_name_or_path)
            # Teacher is not trainable.
            for param in teacher.parameters():
                param.requires_grad = False
            model = MyBart(model, teacher=teacher, args=args)

        elif 'l2p' in args.baseline:
            config = BartConfig.from_pretrained(args.model_name_or_path)
            config.num_labels = taskcla
            model = L2PBartForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
            # Only the prompt pool is tunable.
            for n, param in model.model.named_parameters():
                param.requires_grad = False
            for n, param in model.model.encoder.prompt_pool.named_parameters():
                param.requires_grad = True
            for n, param in model.model.encoder.keys.named_parameters():
                param.requires_grad = True

        elif 'classification' in args.baseline:
            model = MyBartForSequenceClassification.from_pretrained(
                args.model_name_or_path, num_labels=taskcla, args=args)
            model = MyBart(model, args=args)

        if 'fix_classifier' in args.baseline:
            for n, param in model.model.classification_head.named_parameters():
                param.requires_grad = False
        if 'fix_encoder' in args.baseline:
            for n, param in model.model.model.encoder.named_parameters():
                param.requires_grad = False
        if 'fix_decoder' in args.baseline:
            for n, param in model.model.model.decoder.named_parameters():
                param.requires_grad = False
        if 'fix_last_layer' in args.baseline:
            for n, param in model.model.model.decoder.layers[-1].named_parameters():
                param.requires_grad = False
        if 'fix_embedding' in args.baseline:
            for n, param in model.model.model.named_parameters():
                if 'embed' in n:
                    param.requires_grad = False
        if 'fix_bias' in args.baseline:
            for n, param in model.model.model.named_parameters():
                if 'bias' in n:
                    param.requires_grad = False
    else:
        raise NotImplementedError('Currently, we only support BART as the backbone model.')

    print(f'Baseline name: {args.baseline}')

    return model
