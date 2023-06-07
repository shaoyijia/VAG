import argparse
import logging

from transformers import (
    MODEL_MAPPING,
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parsing_finetune():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--params', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--saved_output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_scheduler_type", default="linear", help="The scheduler type to use.",
                        choices=["none", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides epoch.")
    parser.add_argument("--idrandom", type=int, help="which sequence to use", default=0)
    parser.add_argument("--ft_task", type=int, help="task id")
    parser.add_argument("--base_dir", default='./outputs', type=str, help="task id")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--lamb', type=float, required=False, help='A hyper-parameter for loss combination.')
    parser.add_argument("--sequence_file", type=str, help="smax")
    parser.add_argument('--epoch', type=int)
    parser.add_argument("--ntasks", type=int, help="total number of tasks")
    parser.add_argument('--classifier_lr', type=float)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--task_name', type=str)
    parser.add_argument("--print_model", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument('--tokenizer_name', type=str, default='facebook/bart-base')
    parser.add_argument("--store_ratio", type=float, default=0.01, help='how many samples to store for replaying')
    parser.add_argument('--aug_ratio', type=float, help='Ratio of the augmented data.', default=0.1)
    parser.add_argument('--use_dev', action='store_true', help='Use the dev set for early stopping.')
    parser.add_argument("--eval_every_epoch", action="store_true", help="Evaluate in each epoch.")

    args = parser.parse_args()

    return args
