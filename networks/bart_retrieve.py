import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import BartTokenizer

from networks.l2p_model import L2PBartForConditionalGeneration
from networks.my_bart_model import MyBartForConditionalGeneration
from networks.utils import sim_matrix, DistillKL


class BartWithLabelRetriever(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if 'l2p' in self.args.baseline:
            self.model = L2PBartForConditionalGeneration.from_pretrained(args.model_name_or_path)
            # Only the prompt pool is tunable.
            for n, param in self.model.model.named_parameters():
                param.requires_grad = False
            for n, param in self.model.model.encoder.prompt_pool.named_parameters():
                param.requires_grad = True
            for n, param in self.model.model.encoder.keys.named_parameters():
                param.requires_grad = True
        else:
            self.model = MyBartForConditionalGeneration.from_pretrained(args.model_name_or_path, args=args)
        if 'ewc' in self.args.baseline or 'distill' in self.args.baseline:
            self.teacher = MyBartForConditionalGeneration.from_pretrained(args.model_name_or_path, args=args)
            for param in self.teacher.parameters():
                param.requires_grad = False
        if 'label_replay' in self.args.baseline:
            self.previously_seen_labels_tokens = None
            self.previously_seen_labels_targets = None
            self.previously_seen_labels_attention_mask = None
            self.previously_seen_labels_vocabulary_mask = None
        self.kd_loss = DistillKL(1)
        self.tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.label_pools = None
        self.label2idx = None

    def set_masked_vocabulary(self, masked_vocabulary):
        """For VAG loss."""
        self.model.masked_vocabulary = masked_vocabulary

    def set_predict_masked_vocabulary(self, predict_masked_vocabulary):
        """We also use VAG loss on label augmented data."""
        self.model.predict_masked_vocabulary = predict_masked_vocabulary

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                adapter_names=None,
                buffer=None,
                **kwargs):
        if 'distill' in self.args.baseline:
            output_hidden_states = True  # We need the hidden states to calculate distill_loss.

        if 'l2p' in self.args.baseline:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                adapter_names=adapter_names,
                add_prompt=True,
                **kwargs
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                adapter_names=adapter_names,
                restrict_vocabulary=True,
                **kwargs
            )

        # For experience replay, interleaving old samples with current data in training batches.
        if 'experience_replay' in self.args.baseline and buffer is not None and buffer.num_seen_examples > 0:
            replay_input_ids, replay_labels, replay_attention_mask, replay_decoder_input_ids = \
                buffer.get_data(input_ids.size(0))
            replay_input_ids = replay_input_ids.to(self.model.device)
            replay_labels = replay_labels.to(self.model.device)
            replay_attention_mask = replay_attention_mask.to(self.model.device)
            replay_decoder_input_ids = replay_decoder_input_ids.to(self.model.device)
            replay_outputs = self.model(input_ids=replay_input_ids,
                                        decoder_input_ids=replay_decoder_input_ids,
                                        labels=replay_labels,
                                        attention_mask=replay_attention_mask,
                                        restrict_vocabulary=False)
            outputs.loss += replay_outputs.loss

        if 'ewc' in self.args.baseline:
            loss_reg = 0
            if self.args.task > 0:
                for (name, param), (_, param_old) in zip(self.model.named_parameters(),
                                                         self.teacher.named_parameters()):
                    loss_reg += torch.sum(
                        kwargs['self_fisher']['model.' + name] * (param_old.cuda() - param.cuda()).pow(2)) / 2
                outputs.loss += self.args.lamb * loss_reg

        if 'distill' in self.args.baseline:
            student_ori = outputs
            teacher_ori = self.teacher(input_ids,
                                       attention_mask=attention_mask,
                                       decoder_input_ids=decoder_input_ids,
                                       decoder_attention_mask=decoder_attention_mask,
                                       head_mask=head_mask,
                                       decoder_head_mask=decoder_head_mask,
                                       cross_attn_head_mask=cross_attn_head_mask,
                                       encoder_outputs=encoder_outputs,
                                       inputs_embeds=inputs_embeds,
                                       decoder_inputs_embeds=decoder_inputs_embeds,
                                       labels=labels,
                                       use_cache=use_cache,
                                       output_attentions=output_attentions,
                                       output_hidden_states=True)
            distill_loss = self.kd_loss(teacher_ori.decoder_hidden_states[-1], student_ori.decoder_hidden_states[-1])
            outputs['distill_loss'] = distill_loss

        if 'label_replay' in self.args.baseline and self.previously_seen_labels_tokens is not None:
            label_replay_vocabulary_mask = None
            if self.previously_seen_labels_tokens.shape[0] <= input_ids.shape[0]:
                selected_labels_tokens = self.previously_seen_labels_tokens
                selected_labels_attention_mask = self.previously_seen_labels_attention_mask
                selected_labels_targets = self.previously_seen_labels_targets
                if self.previously_seen_labels_vocabulary_mask is not None:
                    label_replay_vocabulary_mask = self.previously_seen_labels_vocabulary_mask
            else:
                selected_idx = torch.randint(high=self.previously_seen_labels_tokens.shape[0],
                                             size=(input_ids.shape[0],)).to(self.previously_seen_labels_tokens.device)
                selected_labels_tokens = self.previously_seen_labels_tokens[selected_idx]
                selected_labels_attention_mask = self.previously_seen_labels_attention_mask[selected_idx]
                selected_labels_targets = self.previously_seen_labels_targets[selected_idx]
                if self.previously_seen_labels_vocabulary_mask is not None:
                    label_replay_vocabulary_mask = self.previously_seen_labels_vocabulary_mask[selected_idx]
            selected_labels_tokens = selected_labels_tokens.to(self.model.device)
            selected_labels_targets = selected_labels_targets.to(self.model.device)
            selected_labels_attention_mask = selected_labels_attention_mask.to(self.model.device)
            if label_replay_vocabulary_mask is not None:
                label_replay_vocabulary_mask = label_replay_vocabulary_mask.to(self.model.device)

            label_replay_output = self.model(
                input_ids=selected_labels_tokens,
                attention_mask=selected_labels_attention_mask,
                labels=selected_labels_targets,
                use_cache=use_cache,
                return_dict=return_dict,
                restrict_vocabulary=False,
                label_replay_vocabulary_mask=label_replay_vocabulary_mask,
                **kwargs
            )
            outputs['label_replay_loss'] = label_replay_output.loss

        return outputs

    def initialize_label_pool(self, label_set):
        """The generation framework needs to maintain a label pool which keeps growing in the class-incremental
        learning process."""
        label_names = []
        label2idx = []
        for k, v in label_set.items():
            label2idx.append(k)
            label_names.append(v)
        self.label_pools = torch.tensor(self.embedder.encode(label_names))
        self.label2idx = torch.tensor(label2idx)

    def predict(self, input_ids, task_label_mask=None, **kwargs):
        """At inference, the model will generate a sequence and the final label is retrieved from the label pool
        based on the text similarity."""
        if 'l2p' in self.args.baseline:
            outputs = self.model.generate(input_ids, add_prompt=True)
        else:
            outputs = self.model.generate(input_ids)
        decoded_labels = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        embeds_for_retrieving = torch.tensor(self.embedder.encode(decoded_labels))
        cosine_sim = sim_matrix(embeds_for_retrieving, self.label_pools)
        predictions = cosine_sim.argmax(dim=1)
        # Map the predictions to the label index to make calculating accuracy easier.
        pred_label = self.label2idx.to(predictions.device)[predictions]

        if task_label_mask is not None:
            # Check the within-task prediction and task-id prediction for analysis.
            tid_pred_correct_num = 0
            for i in predictions:
                y = i.item()
                if task_label_mask[y] == 1:
                    tid_pred_correct_num += 1
            pred = cosine_sim.softmax(dim=1)
            pred = pred * task_label_mask
            pred = pred.argmax(dim=1)
            til_pred_label = self.label2idx.to(pred.device)[pred]

            return pred_label, til_pred_label, tid_pred_correct_num
        else:
            return pred_label, None, None
