"""Fine-tune BART in the generation framework."""
import logging
import os
import random

import nlpaug.augmenter.word as naw
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import concatenate_datasets, Dataset
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from transformers import (
    set_seed,
)
from transformers.data.data_collator import DataCollatorForSeq2Seq, DataCollatorWithPadding

import utils
from approaches.finetune import Appr
from config import parsing_finetune
from data import TACRED_LABEL_MAP, FEWREL_LABEL_MAP
from data import get_dataset
from utils import load_json, dump_json

# Set up logger

logger = logging.getLogger(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main():
    args = parsing_finetune()
    args = utils.prepare_sequence_finetune(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    appr = Appr(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(fp16=args.fp16, kwargs_handlers=[ddp_kwargs])
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if args.log_dir is not None:
        handler = logging.FileHandler(args.log_dir)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Declare the model and set the training parameters.
    logger.info('==> Building model..')

    taskcla = 200  # Won't be used, just a placeholder.
    model = utils.lookfor_model_finetune(args, taskcla)

    if args.print_model:
        utils.print_model_report(model)
        exit()

    # Get the datasets and process the data.
    if 'bart' in args.model_name_or_path:
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError('We only support T5 and Bart at present.')

    max_length = args.max_seq_length

    logger.info('==> Preparing data..')

    datasets, label_set = get_dataset(args.dataset_name, tokenizer=tokenizer, args=args, return_label_set=True)

    # Map the task id to task name so that we can support different sequences.
    try:
        train_dataset = datasets[args.task_name[args.task]]['train']
    except KeyError:
        train_dataset = datasets[int(args.task_name[args.task])]['train']

    def get_vocabulary_mask_for_each_task(tmp_label_set):
        # Restrict the vocabulary to tokens in the current task labels.
        if 'bart' in args.baseline:
            v_mask = torch.zeros(model.model.config.vocab_size)
        else:
            raise NotImplementedError('Currently, we only support BART as the backbone model!')
        for l in tmp_label_set.values():
            tokenized_l = tokenizer(l).input_ids
            for i in tokenized_l:
                v_mask[i] = 1
        return v_mask

    if 'label_replay' in args.baseline and args.task != 0:
        # Label augmentation.
        aug_ratio = args.aug_ratio
        # Cache the augmented labels.
        aug_data_dir = f'./tmp/{args.dataset_name}_ContextualWordEmbsAug_{aug_ratio}.json'
        if os.path.exists(aug_data_dir):
            aug_data = load_json(aug_data_dir)
        else:
            current_task_data_per_class_cnt = len(train_dataset) / len(
                label_set[args.task])  # We simply calculate the avg.
            aug = naw.ContextualWordEmbsAug(model_path='roberta-large', action='insert')
            aug_per_class_cnt = int(current_task_data_per_class_cnt * aug_ratio)
            aug_data = {}
            for t, task_label_set in label_set.items():
                for idx, label_name in label_set[t].items():
                    if 'tacred' in args.dataset_name:
                        aug_labels = aug.augment(TACRED_LABEL_MAP[label_name], n=aug_per_class_cnt)
                    elif 'fewrel' in args.dataset_name:
                        aug_labels = aug.augment(FEWREL_LABEL_MAP[label_name], n=aug_per_class_cnt)
                    else:
                        aug_labels = aug.augment(label_name, n=aug_per_class_cnt)
                    aug_data[label_name] = aug_labels
            os.makedirs('./tmp', exist_ok=True)
            dump_json(aug_data, aug_data_dir)

        # Each pseudo replay data is also associated with its vocabulary set.
        previously_seen_labels = []
        previously_seen_labels_targets = []
        previously_seen_labels_vocabulary_mask = []
        for t in range(args.task):
            try:
                one_task_label_set = label_set[args.task_name[t]]
            except KeyError:
                one_task_label_set = label_set[int(args.task_name[t])]

            vocabulary_mask = get_vocabulary_mask_for_each_task(one_task_label_set)

            for v in one_task_label_set.values():
                previously_seen_labels.extend(aug_data[v])
                previously_seen_labels_targets.extend([v] * len(aug_data[v]))
                previously_seen_labels_vocabulary_mask.append(
                    vocabulary_mask.reshape(1, -1).repeat(len(aug_data[v]), 1))

        tokenized_seen_labels = tokenizer(previously_seen_labels, return_tensors='pt', padding=True)
        model.previously_seen_labels_tokens = tokenized_seen_labels.input_ids
        label_targets = tokenizer(previously_seen_labels_targets, return_tensors='pt', padding=True).input_ids
        label_targets = torch.where(label_targets != tokenizer.pad_token_id, label_targets, -100)
        model.previously_seen_labels_targets = label_targets
        model.previously_seen_labels_attention_mask = tokenized_seen_labels.attention_mask
        if 'restrict_vocabulary' in args.baseline:
            model.previously_seen_labels_vocabulary_mask = torch.cat(previously_seen_labels_vocabulary_mask, dim=0)

        aug_train_data = {'text': [], 'semantic_labels': [], 'labels': []}
        try:
            one_task_label_set = label_set[args.task_name[args.task]]
        except KeyError:
            one_task_label_set = label_set[int(args.task_name[args.task])]
        for v in one_task_label_set.values():
            for aug_v in aug_data[v]:
                aug_train_data['text'].append(aug_v)
                aug_train_data['semantic_labels'].append(v)
                aug_train_data['labels'].append(-1)

        aug_train_data = Dataset.from_dict(aug_train_data)
        train_dataset = concatenate_datasets([train_dataset, aug_train_data])

    print('dataset_name: ', args.dataset_name)
    print('train_loader: ', len(train_dataset))
    print('test_loader: ', len(datasets[args.task]['test']))

    train_data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model.model,
        label_pad_token_id=-100
    )

    test_data_collator = DataCollatorWithPadding(tokenizer)

    input_column = 'text'
    label_column = 'semantic_labels'

    def preprocess_function(examples):
        inputs, targets = [], []
        for i in range(len(examples[input_column])):
            inputs.append(examples[input_column][i])
            targets.append(examples[label_column][i].replace('_', ' '))  # Make the label more like a NL phrase.

        inputs = [inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=max_length, padding='max_length', truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_length, padding='max_length', truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        # Labels mapped into index.
        model_inputs["idx_labels"] = examples["labels"]
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function,
                                      batched=True,
                                      remove_columns=[input_column, label_column])

    train_loader = DataLoader(train_dataset, collate_fn=train_data_collator, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, num_workers=8)

    test_loaders = []
    for eval_t in range(args.task + 1):
        # Map the task id to task name so that we can support different sequences.
        try:
            test_dataset = datasets[args.task_name[eval_t]]['test']
        except KeyError:
            test_dataset = datasets[int(args.task_name[eval_t])]['test']

        test_dataset = test_dataset.map(
            lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length),
            batched=True)
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_loader = DataLoader(test_dataset, collate_fn=test_data_collator, batch_size=args.batch_size, shuffle=False,
                                 drop_last=False,
                                 num_workers=8)
        test_loaders.append(test_loader)

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}. Decode to: {tokenizer.decode(train_dataset[index]['input_ids'])}")

    dev_loader = None
    if args.use_dev:
        try:
            dev_dataset = datasets[args.task_name[args.task]]['dev']
        except KeyError:
            dev_dataset = datasets[int(args.task_name[args.task])]['dev']
        dev_dataset = dev_dataset.map(
            lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length),
            batched=True)
        dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        dev_loader = DataLoader(dev_dataset, collate_fn=test_data_collator, batch_size=args.batch_size, shuffle=False,
                                drop_last=False, num_workers=8)

    train_loader_replay = None

    # Initialize the seen labels.
    seen_label_set = {}
    for eval_t in range(args.task + 1):
        try:
            one_task_label_set = label_set[args.task_name[eval_t]]
        except KeyError:
            one_task_label_set = label_set[int(args.task_name[eval_t])]
        for k, v in one_task_label_set.items():
            seen_label_set[k] = v
    model.initialize_label_pool(seen_label_set)

    task_mask = {}  # For calculating TIL performance (analysis purpose).
    task_cla = len(seen_label_set)
    cnt = 0
    for eval_t in range(args.task + 1):
        try:
            one_task_label_set = label_set[args.task_name[eval_t]]
        except KeyError:
            one_task_label_set = label_set[int(args.task_name[eval_t])]
        task_mask[eval_t] = torch.zeros(task_cla)
        for _ in range(len(one_task_label_set)):
            task_mask[eval_t][cnt] = 1
            cnt += 1

    if 'restrict_vocabulary' in args.baseline:
        # Restrict the vocabulary to tokens in the current task labels.
        try:
            current_task_label_set = label_set[args.task_name[args.task]]
        except KeyError:
            current_task_label_set = label_set[int(args.task_name[args.task])]
        vocabulary_mask = get_vocabulary_mask_for_each_task(current_task_label_set)
        print(f'======> vocabulary_mask.sum() = {vocabulary_mask.sum()}')
        model.set_masked_vocabulary(vocabulary_mask)

    appr.train(model, accelerator, tokenizer, train_loader, train_dataset, test_dataset, test_loaders, dev_loader,
               train_loader_replay, task_mask)


if __name__ == '__main__':
    main()
