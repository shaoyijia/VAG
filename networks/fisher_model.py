"""Calculate the Fisher matrix for EWC."""
import os

import torch
import torch.distributed as dist
from tqdm.auto import tqdm


def gather_importance(head_importance):
    head_importance_list = [torch.zeros_like(head_importance) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=head_importance_list, tensor=head_importance.contiguous())  # everyone need to do this
    head_importance_list = torch.stack(head_importance_list)
    head_importance = torch.mean(head_importance_list, dim=0)
    return head_importance


def fisher_compute(train_dataloader_prune, model, self_fisher, accelerator, args):
    torch.cuda.empty_cache()
    fisher_path = os.path.join(args.output_dir, 'fisher')

    if args.task > 0:
        fisher_old = {}
        for n, _ in model.named_parameters():
            fisher_old[n] = self_fisher[n].clone().cpu()  # Move fisher_old to cpu to save gpu memory.

    # Init
    progress_bar = tqdm(range(len(train_dataloader_prune)), disable=not accelerator.is_local_main_process)

    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = 0 * p.data
    # Compute
    model.train()

    for step, inputs in enumerate(train_dataloader_prune):
        model.zero_grad()
        input_ids = inputs['input_ids']
        sbatch = input_ids.size(0)

        if 'bart' in model.args.baseline or 't5' in model.args.baseline:
            outputs = model(**inputs, self_fisher=self_fisher)
        else:
            outputs = model(inputs, self_fisher=self_fisher)

        loss = outputs.loss  # loss 1

        loss = loss / args.gradient_accumulation_steps

        accelerator.backward(loss)  # sync
        progress_bar.update(1)
        progress_bar.set_description('EWC Fisher Compute Iter (loss=%5.3f)' % loss.item())
        # Get model
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += sbatch * p.grad.data.pow(2)

    # Mean
    for n, _ in model.named_parameters():
        fisher[n] = fisher[n] / len(train_dataloader_prune)
        fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)

    self_fisher = fisher

    if args.task > 0:
        for n, _ in model.named_parameters():
            self_fisher[n] = (self_fisher[n] + fisher_old[n].cuda() * args.task) / (
                    args.task + 1)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        torch.save(self_fisher, fisher_path)

    return fisher
