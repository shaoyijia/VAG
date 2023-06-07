"""Fine-tune BART in the classifier framework."""
import logging
import os
import random

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, BartTokenizer
from transformers import (
    set_seed,
)

import utils
from approaches.finetune_baseline import Appr
from config import parsing_finetune
from data import get_dataset

# Set up logger
logger = logging.getLogger(__name__)


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

    # Get the datasets and process the data.
    if 'bart' in args.tokenizer_name:
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError(f'Tokenizer name {args.tokenizer_name} is not supported.')

    max_length = args.max_seq_length
    taskcla = 300

    logger.info('==> Preparing data..')

    datasets, label_set = get_dataset(args.dataset_name, tokenizer=tokenizer, args=args, return_label_set=True)

    # Map the task id to task name so that we can support different sequences.
    try:
        train_dataset = datasets[args.task_name[args.task]]['train']
    except KeyError:
        train_dataset = datasets[int(args.task_name[args.task])]['train']

    print('dataset_name: ', args.dataset_name)
    print('train_loader: ', len(train_dataset))
    print('test_loader: ', len(datasets[args.task]['test']))

    data_collator = DataCollatorWithPadding(tokenizer)

    with open(args.sequence_file, 'r') as f:
        datas = f.readlines()[args.idrandom]
        data = datas.split()

    train_dataset = train_dataset.map(
        lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_loader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, num_workers=8)

    test_loaders = []
    task_mask = {}  # For calculating TIL performance (analysis purpose).
    for eval_t in range(args.task + 1):
        # Map the task id to task name so that we can support different sequences.
        try:
            test_dataset = datasets[args.task_name[eval_t]]['test']
            task_mask[eval_t] = torch.zeros(taskcla)
            for idx in label_set[args.task_name[eval_t]]:
                task_mask[eval_t][idx] = 1
        except KeyError:
            test_dataset = datasets[int(args.task_name[eval_t])]['test']
            task_mask[eval_t] = torch.zeros(taskcla)
            for idx in label_set[int(args.task_name[eval_t])]:
                task_mask[eval_t][idx] = 1

        test_dataset = test_dataset.map(
            lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length),
            batched=True)
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_loader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.batch_size, shuffle=False,
                                 drop_last=False,
                                 num_workers=8)
        test_loaders.append(test_loader)

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
        dev_loader = DataLoader(dev_dataset, collate_fn=data_collator, batch_size=args.batch_size, shuffle=False,
                                drop_last=False, num_workers=8)

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}. "
            f"Decode to: {tokenizer.decode(train_dataset[index]['input_ids'])}")

    # Declare the model and set the training parameters.
    logger.info('==> Building model..')

    model = utils.lookfor_model_finetune(args, taskcla)

    if args.print_model:
        utils.print_model_report(model)
        exit()

    train_loader_replay = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=50,
        num_workers=0
    )

    appr.train(model, accelerator, tokenizer, train_loader, train_dataset, test_dataset, test_loaders, dev_loader,
               train_loader_replay, task_mask)


if __name__ == '__main__':
    main()
