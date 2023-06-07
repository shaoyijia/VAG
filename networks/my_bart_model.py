import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import ModelWithHeadsAdaptersMixin, BartPretrainedModel, BartConfig, BartModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqSequenceClassifierOutput
from transformers.models.bart.modeling_bart import shift_tokens_right, BartClassificationHead

from networks.utils import DistillKL


class MyBartForConditionalGeneration(ModelWithHeadsAdaptersMixin, BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig, args):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        # For analysis.
        self.masked_vocabulary = None  # Tokens with mask as 1 will be kept in lm head softmax calculation.
        self.predict_masked_vocabulary = None  # Tokens with mask 1 will be kept in lm head softmax calculation.

        self.args = args

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
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
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            adapter_names=adapter_names,
        )
        lm_logits = self.model.encoder.invertible_adapters_forward(outputs[0], rev=True)
        lm_logits = self.lm_head(lm_logits) + self.final_logits_bias

        unrestricted_masked_lm_loss = None
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if 'restrict_vocabulary' in self.args.baseline and kwargs['restrict_vocabulary'] is True:
                if 'experience_replay' in self.args.baseline:
                    # We also use the loss w/ vocabulary restriction because replay data is available.
                    unrestricted_masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

                # We restrict the softmax probability to a reduced vocabulary.
                # print('Apply vocabulary mask successfully!')
                for tmp_i in labels:
                    for tmp_j in tmp_i:
                        if tmp_j != -100 and self.masked_vocabulary[tmp_j] == 0:  # Sanity check.
                            print('Masked vocabulary sanity check fail!')
                            import pdb
                            pdb.set_trace()
                lm_logits = lm_logits * self.masked_vocabulary.to(lm_logits.device)
                lm_logits = torch.where(lm_logits != 0, lm_logits, torch.tensor(-np.inf).to(lm_logits.device))
            if 'label_replay_vocabulary_mask' in kwargs and kwargs['label_replay_vocabulary_mask'] is not None:
                for tmp_i in range(labels.shape[0]):
                    for tmp_j in labels[tmp_i]:
                        if tmp_j != -100 and kwargs['label_replay_vocabulary_mask'][tmp_i][tmp_j] == 0:  # Sanity check.
                            print('Masked vocabulary sanity check fail!')
                            import pdb
                            pdb.set_trace()
                lm_logits = \
                    lm_logits * kwargs['label_replay_vocabulary_mask'].unsqueeze(1).repeat(1, lm_logits.shape[1], 1)
                lm_logits = torch.where(lm_logits != 0, lm_logits, torch.tensor(-np.inf).to(lm_logits.device))
                # print('Restrict replay label vocabulary successfully!')
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            if unrestricted_masked_lm_loss is not None:
                # We also use the loss w/ vocabulary restriction because replay data is available.
                masked_lm_loss += unrestricted_masked_lm_loss

        if self.predict_masked_vocabulary is not None:
            lm_logits = lm_logits * self.predict_masked_vocabulary.to(lm_logits.device)
            lm_logits = torch.where(lm_logits != 0, lm_logits, torch.tensor(-np.inf).to(lm_logits.device))
            # print('Restrict vocabulary in prediction successfully!')

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class MyBartForSequenceClassification(ModelWithHeadsAdaptersMixin, BartPretrainedModel):
    def __init__(self, config: BartConfig, args, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.args = args
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

        # For analysis.
        self.masked_label = None  # Tokens with mask as 1 will be kept in lm head softmax calculation.

    def set_masked_label(self, masked_label):
        self.masked_label = masked_label

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            adapter_names=None,
            restrict_label=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            adapter_names=adapter_names,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)
        eos_mask = self.model.encoder.adjust_attention_mask_for_parallel(hidden_states, eos_mask)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                                  :, -1, :
                                  ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                if self.masked_label is not None and restrict_label is True:
                    # We restrict the softmax probability to a reduced label set.
                    for tmp_i in labels:
                        if self.masked_label[tmp_i] == 0:  # Sanity check.
                            print('Masked label sanity check fail!')
                            import pdb
                            pdb.set_trace()
                    logits = logits * self.masked_label.to(logits.device)
                    logits = torch.where(logits != 0, logits, torch.tensor(-np.inf).to(logits.device))
                    # print('Restrict label successfully!')
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class MyBart(nn.Module):
    """Wrapper on top of MyBartForSequenceClassification."""

    def __init__(self, model, teacher=None, args=None):
        super().__init__()
        self.model = model
        self.teacher = teacher
        self.kd_loss = DistillKL(1)
        self.config = model.config
        self.args = args
        self.mse = torch.nn.MSELoss()
        self.dropout = nn.Dropout(0.1)

        if 'ldbr' in args.baseline:
            self.General_Encoder = nn.Sequential(
                nn.Linear(self.model.config.d_model, 128),
                nn.Tanh()
            )
            self.Specific_Encoder = nn.Sequential(
                nn.Linear(self.model.config.d_model, 128),
                nn.Tanh()
            )
            self.task_classifier = nn.Sequential(
                nn.Linear(128, args.ntasks)
            )
            self.cls_classifier = nn.Sequential(
                nn.Linear(2 * 128, self.model.config.num_labels)
            )

    def set_masked_label(self, masked_label):
        self.model.set_masked_label(masked_label)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            adapter_names=None,
            buffer=None,
            restrict_label=False,
            **kwargs
    ):
        outputs = self.model(input_ids,
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
                             output_hidden_states=True,
                             restrict_label=restrict_label)
        loss = outputs.loss
        logits = outputs.logits
        distill_loss = None

        # For experience replay, interleaving old samples with current data in training batches.
        if 'experience_replay' in self.args.baseline and buffer is not None and buffer.num_seen_examples > 0:
            replay_input_ids, replay_labels, replay_attention_mask = buffer.get_data(input_ids.size(0))
            replay_input_ids = replay_input_ids.to(self.model.device)
            replay_labels = replay_labels.to(self.model.device)
            replay_attention_mask = replay_attention_mask.to(self.model.device)
            replay_outputs = self.model(input_ids=replay_input_ids,
                                        labels=replay_labels,
                                        attention_mask=replay_attention_mask)
            # print('Add replay data successfully!')
            loss += replay_outputs.loss

        if 'derpp' in self.args.baseline and buffer is not None and buffer.num_seen_examples > 0:
            replay_input_ids, replay_labels, replay_logits, replay_attention_mask = buffer.get_data(input_ids.size(0))
            replay_input_ids = replay_input_ids.to(self.model.device)
            replay_labels = replay_labels.to(self.model.device)
            replay_logits = replay_logits.to(self.model.device)
            replay_attention_mask = replay_attention_mask.to(self.model.device)
            replay_outputs = self.model(input_ids=replay_input_ids,
                                        labels=replay_labels,
                                        attention_mask=replay_attention_mask)
            # Set alpha, beta to 0.5, 0.5
            loss = loss + 0.5 * replay_outputs.loss + 0.5 * F.mse_loss(replay_logits, outputs.logits)

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

        if 'ewc' in self.args.baseline and 'self_fisher' in kwargs:  # We don't need to do this in evaluation.
            loss_reg = 0
            if self.args.task > 0:
                for (name, param), (_, param_old) in zip(self.model.named_parameters(),
                                                         self.teacher.named_parameters()):
                    loss_reg += torch.sum(
                        kwargs['self_fisher']['model.' + name] * (param_old.cuda() - param.cuda()).pow(2)) / 2
            loss += self.args.lamb * loss_reg

        if 'ldbr' in self.args.baseline:
            sentence_embedding = outputs.decoder_hidden_states[-1][:, -1, :]
            general_features = self.General_Encoder(sentence_embedding)
            specific_features = self.Specific_Encoder(sentence_embedding)

            task_pred = self.task_classifier(specific_features)
            features = torch.cat([general_features, specific_features], dim=1)
            logits = self.cls_classifier(features)
            loss_fct = nn.CrossEntropyLoss()
            if labels is not None:
                loss = loss_fct(logits, labels)
            else:
                loss = None
        else:
            sentence_embedding = None
            task_pred = None
            general_features = None
            specific_features = None

        return ModelOutput(
            loss=loss,
            distill_loss=distill_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            total_g_fea=general_features,
            total_s_fea=specific_features,
            task_pred=task_pred,
            sentence_embedding=sentence_embedding
        )
