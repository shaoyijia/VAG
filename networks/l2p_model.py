"""Pytorch Implementation for
Learning to Prompt for Continual Learning (https://arxiv.org/pdf/2112.08654.pdf)"""
import math
import random
from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import ModelWithHeadsAdaptersMixin, BartPretrainedModel, BartConfig, \
    InvertibleAdaptersMixin
from transformers.adapters.models.bart import BartEncoderDecoderAdaptersMixin, BartModelAdaptersMixin
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput, Seq2SeqLMOutput, BaseModelOutput, \
    Seq2SeqModelOutput
from transformers.models.bart.modeling_bart import BartClassificationHead, shift_tokens_right, \
    BartLearnedPositionalEmbedding, BartEncoderLayer, _expand_mask, BartDecoder
from transformers.utils import logging

logger = logging.get_logger(__name__)


def sim_matrix(a, b, eps=1e-8):
    """Batch version of CosineSimilarity."""
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class L2PBartEncoder(InvertibleAdaptersMixin, BartEncoderDecoderAdaptersMixin, BartPretrainedModel):
    """
        Prompt pool (P): {P_1, ..., P_M}, P_i [Lp, embed_dim]
        Learnable key: {(k_1, P_1), ..., (k_M, P_M)}, k_i [last_hidden_dim]
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None, **kwargs):
        super().__init__(config)
        self.config = config

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()
        self.gradient_checkpointing = False

        self.M = kwargs['M'] if kwargs is not None and 'M' in kwargs else 10
        self.N = kwargs['M'] if kwargs is not None and 'N' in kwargs else 5
        self.Lp = kwargs['Lp'] if kwargs is not None and 'Lp' in kwargs else 12

        # Prompt pool and learnable keys.
        # We maintain {P_1, ..., P_M} as a large matrix of shape (M, Lp * hidden_dim) for efficient look-up.
        init_prompt_value = torch.FloatTensor(self.M, self.Lp * self.config.hidden_size).uniform_(-0.5, 0.5)
        self.prompt_pool = nn.Embedding(self.M, self.Lp * self.config.hidden_size)
        self.prompt_pool.weight = nn.parameter.Parameter(init_prompt_value)
        self.keys = nn.Embedding(self.M, self.config.hidden_size)

        # Hyperparameter for the loss function.
        self.lam = 0.5  # Follow the original paper.

    def _cat_selected_prompt_to_input(self, input_ids):
        """
        Selects prompts which minimize the matching function and concatenates them to the inputs.
        x_p = [P_s1; ... ; P_sN; x_e]
        """
        inputs_embeds = self.embed_tokens(input_ids)

        if len(inputs_embeds.shape) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # Use the frozen pre-trained model to get the query features: q(x) = f(x)[0,:]
        # Since BART contains an encoder and a decoder, we use the encoder to get the query feature.
        q = self.forward(inputs_embeds=inputs_embeds, add_prompt=False)[0][:, 0, :]
        sim = sim_matrix(q, self.keys.weight)
        # We don't want the similarity to go to negative.
        sim = torch.abs(sim)
        selection = torch.topk(sim, self.N, dim=1)
        matching_loss = selection.values.sum(dim=1).mean()
        selected_prompt = self.prompt_pool.weight[selection.indices].reshape(
            -1, self.Lp * self.N, self.config.hidden_size).to(input_ids.device)

        inputs_embeds = torch.cat([selected_prompt, inputs_embeds], dim=1)

        return inputs_embeds, matching_loss

    def _extend_attention_mask(self, attention_mask):
        """
        Extends attention_mask to match the input_ids's shape.
        """

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [
                torch.full(
                    (n_batches, self.Lp * self.N), 1).to(
                    attention_mask.device).long(),
                attention_mask
            ],
            dim=1,
        )

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            add_prompt=False,
            **kwargs
    ):
        """If add_prompt is True, we add the global prompt and the local prompt to the input_embeds."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if add_prompt:
            inputs_embeds, matching_loss = self._cat_selected_prompt_to_input(input_ids)
            attention_mask = self._extend_attention_mask(attention_mask)
            input_ids = None  # We add soft prompt to input_embeds, so now input_ids is useless.

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = self.invertible_adapters_forward(hidden_states)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        **kwargs,
                    )

                hidden_states = layer_outputs[0]
                attention_mask = self.adjust_attention_mask_for_parallel(hidden_states, attention_mask)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        outputs = BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

        if add_prompt:
            outputs.matching_loss = matching_loss
            outputs.attention_mask = attention_mask

        return outputs


class L2PBartModel(BartModelAdaptersMixin, BartPretrainedModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = L2PBartEncoder(config, self.shared, **kwargs)
        self.decoder = BartDecoder(config, self.shared)

        self._init_adapter_modules()

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

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
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            add_prompt=False,
            **kwargs
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.pre_transformer_forward(**kwargs)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                add_prompt=add_prompt,
                **kwargs,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        if add_prompt:
            # Also extend the attention_mask for the cross attention.
            attention_mask = torch.cat([
                torch.full((attention_mask.shape[0], self.encoder.Lp * self.encoder.N), 1).to(
                    attention_mask.device).long(),
                attention_mask], dim=1,
            )
        # inflate all decoder inputs according to encoder output
        decoder_input_ids, decoder_attention_mask, attention_mask = self.adjust_tensors_for_parallel(
            encoder_outputs[0], decoder_input_ids, decoder_attention_mask, attention_mask
        )
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        outputs = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        outputs.matching_loss = encoder_outputs.matching_loss

        return outputs


class L2PBartForSequenceClassification(ModelWithHeadsAdaptersMixin, BartPretrainedModel):
    def __init__(self, config: BartConfig, **kwargs):
        """
        Prompt pool (P): {P_1, ..., P_M}, P_i [Lp, embed_dim]
        Learnable key: {(k_1, P_1), ..., (k_M, P_M)}, k_i [last_hidden_dim]
        """
        super().__init__(config, **kwargs)
        self.model = L2PBartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)
        self.N = self.model.encoder.N
        self.Lp = self.model.encoder.Lp

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
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        # We move the shift_tokens_right here, so we don't need to pass the input_ids to self.model.
        # decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)

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
            add_prompt=True
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)
        eos_mask = self.model.encoder.adjust_attention_mask_for_parallel(hidden_states, eos_mask)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # Add matching loss:
        loss += self.model.encoder.lam * outputs.matching_loss

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


class L2PBartForConditionalGeneration(ModelWithHeadsAdaptersMixin, BartPretrainedModel):
    """In training, forward() will be called to pass the input_ids through both the encoder and the decoder.

    In inference (generation), generate() in generation_utils.py will first call get_encoder() and pass the input_ids
    through the encoder to get the last_hidden_state. Then forward() will be called in each generation step.

    To support both two cases, we implement the l2p in the encoder.
    """
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config)
        self.model = L2PBartModel(config, **kwargs)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

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
            add_prompt=False,
            **kwargs
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if input_ids is None and 'last_hidden_state' in encoder_outputs:
            # This will happen when generate() in generation_utils.py calls the model to decode step-by-step. The
            # hidden states of input_ids have already been calculated.
            if attention_mask is not None and attention_mask.shape[1] < encoder_outputs['last_hidden_state'].shape[1]:
                attention_mask = torch.cat([
                    torch.full((attention_mask.shape[0], self.model.encoder.Lp * self.model.encoder.N), 1).to(
                        attention_mask.device).long(), attention_mask], dim=1)

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
            add_prompt=add_prompt,
        )
        lm_logits = self.model.encoder.invertible_adapters_forward(outputs[0], rev=True)
        lm_logits = self.lm_head(lm_logits) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Add matching loss:
        if masked_lm_loss is not None and add_prompt:
            masked_lm_loss += self.model.encoder.lam * outputs.matching_loss

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
