"""Generation framework for class-incremental learning."""
import logging
import math
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
)

from networks import fisher_model
from networks.buffer import FixedSizeBuffer
from utils import dump_json

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class Appr(object):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def train(self, model, accelerator, tokenizer, train_loader, train_dataset, test_dataset, test_loaders, dev_loader,
              train_loader_replay, task_mask):

        # before training *********************************************************************************************
        if 'ewc' in self.args.baseline:
            if os.path.exists(os.path.join(self.args.prev_output, 'fisher')):
                print('load fisher matrix **************')
                self_fisher = torch.load(os.path.join(self.args.prev_output, 'fisher'))
                for k, v in self_fisher.items():
                    self_fisher[k] = self_fisher[k].cuda()
            else:
                self_fisher = None

        if 'experience_replay' in self.args.baseline:
            # Load buffer.
            if self.args.task == 0:
                buffer = FixedSizeBuffer(buffer_size=self.args.store_ratio)
            else:
                buffer = torch.load(os.path.join(self.args.model_name_or_path, 'buffer.pth'))

        # Set the optimizer
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters)

        # Prepare everything with the accelerator
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        if dev_loader is not None:
            dev_loader = accelerator.prepare(dev_loader)

        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.epoch * num_update_steps_per_epoch
        else:
            self.args.epoch = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        # Use the dev set for early stopping.
        best_dev_result = -1

        # Train!
        if accelerator.is_main_process:
            logger.info("***** Running training *****")
            logger.info(
                f"Pretrained Model = {self.args.model_name_or_path},  Dataset name = {self.args.dataset_name}, "
                f"seed = {self.args.seed}, test size = {len(test_dataset)}, training size = {len(train_dataset)}")
            logger.info(
                f"  Learning Rate = {self.args.learning_rate}, Warmup Num = {self.args.num_warmup_steps}, "
                f"Pre-trained Model = {self.args.model_name_or_path}")
            logger.info(
                f"  Seq ID = {self.args.idrandom}, Task id = {self.args.task}, dataset name = {self.args.dataset_name},"
                f" Num task = {self.args.ntasks}")
            logger.info(
                f"  Baseline = {self.args.baseline}, Batch Size = {self.args.batch_size}, Epoch= {self.args.epoch}")

        if accelerator.is_main_process:
            tensorboard_file = os.path.join(self.args.output_dir, str(self.args.dataset_name) + '_log')
            print('tensorboard_file: ', tensorboard_file)
            if os.path.isdir(tensorboard_file):
                shutil.rmtree(tensorboard_file)

            # Delete previous models if we do not want to save all checkpoints.
            if 'save_all_ckpt' not in self.args.baseline:
                for saved_output_dir in self.args.saved_output_dir[:-2]:  # we need -2 so that we can load model
                    if os.path.isdir(saved_output_dir):
                        shutil.rmtree(saved_output_dir)

        try:
            for epoch in range(self.args.epoch):
                total_loss = 0
                total_num = 0
                progress_bar = tqdm(range(len(train_loader)), disable=not accelerator.is_local_main_process)
                for step, inputs in enumerate(train_loader):
                    model.train()

                    if 'experience_replay' in self.args.baseline:
                        outputs = model(**inputs, buffer=buffer)
                    elif 'ewc' in self.args.baseline:
                        outputs = model(**inputs, self_fisher=self_fisher)
                    else:
                        outputs = model(**inputs)

                    loss = outputs.loss

                    if 'distill' in self.args.baseline:
                        distill_loss = outputs.distill_loss
                        loss = loss + self.args.lamb * distill_loss

                    if 'label_replay' in self.args.baseline and self.args.ft_task != 0:
                        ori_loss = outputs.loss
                        label_replay_loss = outputs.label_replay_loss
                        loss = loss + self.args.lamb * label_replay_loss

                    accelerator.backward(loss)
                    total_loss += loss.data.cpu().numpy().item() * inputs['input_ids'].size(0)
                    total_num += inputs['input_ids'].size(0)

                    if accelerator.is_main_process and epoch < 1 and step < 1:
                        for n, p in model.named_parameters():
                            if p.grad is not None:
                                print('n,p： ', n, p.size())

                    optimizer.step()

                    lr_scheduler.step()
                    optimizer.zero_grad()

                    progress_bar.update(1)
                    if 'label_replay' in self.args.baseline and self.args.ft_task != 0:
                        progress_bar.set_description(
                            'Train Iter (Epoch=%3d,ori_loss=%5.3f, label_replay_loss=%5.3f)' %
                            (epoch, ori_loss.item(), label_replay_loss.item()))
                    else:
                        progress_bar.set_description(
                            'Train Iter (Epoch=%3d,loss=%5.3f)' % (epoch, loss.item()))

                if self.args.eval_every_epoch:
                    # We track the current task performance in every epoch.
                    test_loader = test_loaders[self.args.ft_task]
                    test_loader = accelerator.prepare(test_loader)
                    micro_f1, macro_f1, acc, _, _, _, _, _, _, _, _, _ = self.eval(model, test_loader, accelerator)
                    logger.info(
                        "Epoch {} macro_f1 = {:.4f}, acc = {:.4f}, average loss = {:.4f} (seed={})".format(
                            epoch, macro_f1, acc, total_loss / total_num, self.args.seed))

                if dev_loader is not None:
                    micro_f1, macro_f1, acc, _, _, _, _, _, _, _, _, _ = self.eval(model, dev_loader, accelerator)
                    logger.info(
                        "**Dev set performance** Epoch {} macro_f1 = {:.4f}, acc = {:.4f} (seed={})".format(
                            epoch, macro_f1, acc, self.args.seed))
                    if acc <= best_dev_result:
                        # We use the dev set for early stopping. Load the best model on dev set and stop training.
                        self.load_model(model)
                        break
                    else:
                        best_dev_result = acc
                        self.save_model(accelerator, model)
                    if epoch == (self.args.epoch - 1):
                        self.save_model(accelerator, model)

        except KeyboardInterrupt:  # Even if control-C, I still want to save the model.
            return

        # after training ***********************************************************************************************
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if dev_loader is None:
                # If we don't use dev set for early stopping, we save the model after the training is finished.
                self.save_model(accelerator, model)
            tokenizer.save_pretrained(self.args.output_dir)

        if 'ewc' in self.args.baseline:
            fisher_model.fisher_compute(train_loader, model, self_fisher, accelerator, self.args)

        if 'experience_replay' in self.args.baseline:
            np.random.seed(self.args.seed * train_loader.dataset['idx_labels'][0])
            # Add new data to the buffer and save the new buffer.
            for _, inputs in enumerate(train_loader):
                buffer.add_data(inputs['input_ids'],
                                labels=inputs['labels'],
                                attention_mask=inputs['attention_mask'],
                                decoder_input_ids=inputs['decoder_input_ids'])
            print(f'The buffer now contains {buffer.num_seen_examples} examples!')
            torch.save(buffer, os.path.join(self.args.output_dir, 'buffer.pth'))

        total_correct_cnt = 0
        total_sample_cnt = 0
        total_til_correct_cnt = 0  # within-task prediction
        total_tid_correct_cnt = 0  # task-id prediction
        predictions = []
        labels = []

        for eval_t in range(self.args.ft_task + 1):  # Test on all seen classes.
            self.args.task = eval_t

            test_loader = test_loaders[eval_t]
            test_loader = accelerator.prepare(test_loader)
            micro_f1, macro_f1, acc, _, correct_cnt, sample_cnt, pred_list, label_list, til_acc, til_correct_cnt, tid_acc, tid_correct_cnt = \
                self.eval(model, test_loader, accelerator, task_label_mask=task_mask[eval_t])
            total_sample_cnt += sample_cnt
            total_correct_cnt += correct_cnt
            total_til_correct_cnt += til_correct_cnt
            total_tid_correct_cnt += tid_correct_cnt
            predictions += pred_list
            labels += label_list

            if accelerator.is_main_process:

                logger.info(
                    "{} On {}, last epoch macro_f1 = {:.4f}, acc = {:.4f} (seed={})".format(
                        self.args.model_name_or_path,
                        self.args.dataset_name, macro_f1,
                        acc, self.args.seed))

                progressive_f1_path = os.path.join(self.args.output_dir + '/../',
                                                   'progressive_f1_' + str(self.args.seed))
                progressive_acc_path = os.path.join(self.args.output_dir + '/../',
                                                    'progressive_acc_' + str(self.args.seed))
                progressive_accumulated_acc_path = os.path.join(self.args.output_dir + '/../',
                                                                'accumulated_acc_' + str(self.args.seed))
                print('progressive_f1_path: ', progressive_f1_path)
                print('progressive_acc_path: ', progressive_acc_path)
                print('progressive_accumulated_acc_path: ', progressive_accumulated_acc_path)

                # Calculate the TIL results and task-id prediction results for analysis.
                progressive_til_acc_path = os.path.join(self.args.output_dir + '/../',
                                                        'til_progressive_acc_' + str(self.args.seed))
                til_accumulated_acc_path = os.path.join(self.args.output_dir + '/../',
                                                        'til_accumulated_acc_' + str(self.args.seed))
                progressive_tid_acc_path = os.path.join(self.args.output_dir + '/../',
                                                        'tid_progressive_acc_' + str(self.args.seed))
                tid_accumulated_acc_path = os.path.join(self.args.output_dir + '/../',
                                                        'tid_accumulated_acc_' + str(self.args.seed))

                if os.path.exists(progressive_f1_path) and os.path.exists(progressive_acc_path):
                    f1s = np.loadtxt(progressive_f1_path)
                    accs = np.loadtxt(progressive_acc_path)
                else:
                    f1s = np.zeros((self.args.ntasks, self.args.ntasks), dtype=np.float32)
                    accs = np.zeros((self.args.ntasks, self.args.ntasks), dtype=np.float32)

                if os.path.exists(progressive_accumulated_acc_path):
                    accumulated_accs = np.loadtxt(progressive_accumulated_acc_path)
                else:
                    accumulated_accs = np.zeros(self.args.ntasks, dtype=np.float32)

                if os.path.exists(progressive_til_acc_path) and os.path.exists(progressive_tid_acc_path):
                    til_accs = np.loadtxt(progressive_til_acc_path)
                    tid_accs = np.loadtxt(progressive_tid_acc_path)
                else:
                    til_accs = np.zeros((self.args.ntasks, self.args.ntasks), dtype=np.float32)
                    tid_accs = np.zeros((self.args.ntasks, self.args.ntasks), dtype=np.float32)

                if os.path.exists(til_accumulated_acc_path) and os.path.exists(tid_accumulated_acc_path):
                    til_accumulated_accs = np.loadtxt(til_accumulated_acc_path)
                    tid_accumulated_accs = np.loadtxt(tid_accumulated_acc_path)
                else:
                    til_accumulated_accs = np.zeros(self.args.ntasks, dtype=np.float32)
                    tid_accumulated_accs = np.zeros(self.args.ntasks, dtype=np.float32)

                f1s[self.args.ft_task][eval_t] = macro_f1
                np.savetxt(progressive_f1_path, f1s, '%.4f', delimiter='\t')

                accs[self.args.ft_task][eval_t] = acc
                np.savetxt(progressive_acc_path, accs, '%.4f', delimiter='\t')

                til_accs[self.args.ft_task][eval_t] = til_acc
                np.savetxt(progressive_til_acc_path, til_accs, '%.4f', delimiter='\t')

                tid_accs[self.args.ft_task][eval_t] = tid_acc
                np.savetxt(progressive_tid_acc_path, tid_accs, '%.4f', delimiter='\t')

                if eval_t == self.args.ft_task:  # Test results on all available test data.
                    accumulated_accs[eval_t] = total_correct_cnt * 1.0 / total_sample_cnt
                    np.savetxt(progressive_accumulated_acc_path, accumulated_accs, '%.4f', delimiter='\t')
                    til_accumulated_accs[eval_t] = total_til_correct_cnt * 1.0 / total_sample_cnt
                    np.savetxt(til_accumulated_acc_path, til_accumulated_accs, '%.4f', delimiter='\t')
                    tid_accumulated_accs[eval_t] = total_tid_correct_cnt * 1.0 / total_sample_cnt
                    np.savetxt(tid_accumulated_acc_path, tid_accumulated_accs, '%.4f', delimiter='\t')

                if self.args.ft_task == self.args.ntasks - 1:  # last ft task, we need a final one
                    final_f1 = os.path.join(self.args.output_dir + '/../', 'f1_' + str(self.args.seed))
                    final_acc = os.path.join(self.args.output_dir + '/../', 'acc_' + str(self.args.seed))

                    forward_f1 = os.path.join(self.args.output_dir + '/../', 'forward_f1_' + str(self.args.seed))
                    forward_acc = os.path.join(self.args.output_dir + '/../', 'forward_acc_' + str(self.args.seed))

                    print('final_f1: ', final_f1)
                    print('final_acc: ', final_acc)

                    # Save the confusion matrix.
                    cm = confusion_matrix(y_true=labels, y_pred=predictions, normalize='true')
                    np.savetxt(self.args.output_dir + '/../confusion_matrix', cm, '%.4f', delimiter='\t')

                    if self.args.baseline == 'one':
                        with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[j][j]) + '\n')
                                f1_file.writelines(str(f1s[j][j]) + '\n')

                    else:
                        with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[-1][j]) + '\n')
                                f1_file.writelines(str(f1s[-1][j]) + '\n')

                        with open(forward_acc, 'w') as file, open(forward_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[j][j]) + '\n')
                                f1_file.writelines(str(f1s[j][j]) + '\n')
        # Save the training arguments.
        training_args = {k: v for k, v in self.args.__dict__.items() if k != 'device'}
        dump_json(training_args, self.args.output_dir + '/../training_args.json')

    def eval(self, model, dataloader, accelerator, task_label_mask=None):
        model.eval()
        label_list = []
        prediction_list = []
        til_prediction_list = []
        tid_pred_correct_num = 0
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        with torch.no_grad():
            for batch, inputs in enumerate(dataloader):
                input_ids = inputs['input_ids']

                pred, til_pred, tid_pred_correct = model.predict(input_ids, task_label_mask=task_label_mask)

                predictions = accelerator.gather(pred)
                references = accelerator.gather(inputs['labels'])

                label_list += references.cpu().numpy().tolist()
                prediction_list += predictions.cpu().numpy().tolist()

                if task_label_mask is not None:
                    til_predictions = accelerator.gather(til_pred)
                    til_prediction_list += til_predictions.cpu().numpy().tolist()
                    tid_pred_correct_num += tid_pred_correct

                progress_bar.update(1)

        micro_f1 = f1_score(label_list, prediction_list, average='micro')
        macro_f1 = f1_score(label_list, prediction_list, average='macro')
        correct_num = sum([float(label_list[i] == prediction_list[i]) for i in range(len(label_list))])
        accuracy = correct_num * 1.0 / len(prediction_list)
        if task_label_mask is not None:
            til_correct_num = sum([float(label_list[i] == til_prediction_list[i]) for i in range(len(label_list))])
            til_accuracy = til_correct_num * 1.0 / len(til_prediction_list)
            tid_pred_accuracy = tid_pred_correct_num * 1.0 / len(til_prediction_list)
        else:
            til_correct_num = -1
            til_accuracy = -1
            tid_pred_correct_num = -1
            tid_pred_accuracy = -1  # Not applicable.

        return micro_f1, macro_f1, accuracy, 0, correct_num, len(prediction_list), prediction_list, label_list, \
               til_accuracy, til_correct_num, tid_pred_accuracy, tid_pred_correct_num

    def save_model(self, accelerator, model):
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.model.save_pretrained(self.args.output_dir)

    def load_model(self, model):
        model_dict_path = os.path.join(self.args.output_dir, 'pytorch_model.bin')
        model.model.load_state_dict(torch.load(model_dict_path, map_location='cpu'))
