# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
# Modified and updated by Yuxuan Wang

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import numpy
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers.modeling_bert import (BertEmbeddings,
                                                BertSelfAttention, BertAttention, BertEncoder, BertLayer,
                                                BertSelfOutput, BertLayerNorm, BertIntermediate, BertOutput,
                                                BertPooler, BertPreTrainedModel, BertOnlyMLMHead)
from modeling.modeling_utils import CaptionPreTrainedModel

logger = logging.getLogger(__name__)


class BertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(BertSelfAttention, self).__init__(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertAttention(BertAttention):
    """
    Modified from BertAttention to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(BertAttention, self).__init__(config)
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None,
                history_state=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask, history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(BertEncoder):
    """
    Modified from BertEncoder to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None,
                encoder_history_states=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            history_state = None if encoder_history_states is None else encoder_history_states[i]
            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i],
                history_state)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


class BertLayer(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(BertLayer, self).__init__(config)
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                                           head_mask, history_state)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertBoundaryModel(BertPreTrainedModel):
    """ Expand from BertModel to handle image region features as input
    """

    def __init__(self, config):
        super(BertBoundaryModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)

        self.obj_feature_dim = config.obj_feature_dim
        self.frame_feature_dim = config.frame_feature_dim
        self.act_feature_dim = config.act_feature_dim
        logger.info('BertBoundaryModel ------ Obj Dimension: {}, Frame Dimension: {}, Action Dimension: {}'.
                    format(self.obj_feature_dim, self.frame_feature_dim, self.act_feature_dim))
        if hasattr(config, 'use_vid_layernorm'):
            self.use_vid_layernorm = config.use_vid_layernorm
        else:
            self.use_vid_layernorm = None

        self.obj_embedding = nn.Linear(self.obj_feature_dim, self.config.hidden_size, bias=True)
        self.frame_embedding = nn.Linear(self.frame_feature_dim, self.config.hidden_size, bias=True)
        self.frame_diff_embedding = nn.Linear(self.frame_feature_dim, self.config.hidden_size, bias=True)
        self.act_embedding = nn.Linear(self.act_feature_dim, self.config.hidden_size, bias=True)
        self.act_diff_embedding = nn.Linear(self.act_feature_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_vid_layernorm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.apply(self.init_weights)

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,
                obj_feats=None, frame_feats=None, frame_feats_diff=None, act_feats=None, act_feats_diff=None,
                position_ids=None, head_mask=None, encoder_history_states=None):
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None and input_ids is not None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if encoder_history_states:
            assert obj_feats is None and frame_feats is None and frame_feats_diff is None and act_feats is None \
                   and act_feats_diff is None, "Cannot take image features while using encoder history states"

        cap_embedding_output = None
        if input_ids is not None:
            cap_embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                                                   token_type_ids=token_type_ids)

        vid_embedding_output = []
        if obj_feats is not None:
            obj_embedding_output = self.obj_embedding(obj_feats)
            if self.use_vid_layernorm:
                obj_embedding_output = self.LayerNorm(obj_embedding_output)
            vid_embedding_output.append(obj_embedding_output)
        if frame_feats is not None:
            frame_embedding_output = self.frame_embedding(frame_feats)
            if self.use_vid_layernorm:
                frame_embedding_output = self.LayerNorm(frame_embedding_output)
            vid_embedding_output.append(frame_embedding_output)
        if frame_feats_diff is not None:
            frame_diff_embedding_output = self.frame_diff_embedding(frame_feats_diff[:, :, self.frame_feature_dim:]) \
                                          - self.frame_diff_embedding(frame_feats_diff[:, :, :self.frame_feature_dim])

            if self.use_vid_layernorm:
                frame_diff_embedding_output = self.LayerNorm(frame_diff_embedding_output)
            vid_embedding_output.append(frame_diff_embedding_output)
        if act_feats is not None:
            act_embedding_output = self.act_embedding(act_feats)
            if self.use_vid_layernorm:
                act_embedding_output = self.LayerNorm(act_embedding_output)
            vid_embedding_output.append(act_embedding_output)
        if act_feats_diff is not None:
            act_diff_embedding_output = self.act_diff_embedding(act_feats_diff[:, :, self.act_feature_dim:]) \
                                        - self.act_diff_embedding(act_feats_diff[:, :, :self.act_feature_dim])
            if self.use_vid_layernorm:
                act_diff_embedding_output = self.LayerNorm(act_diff_embedding_output)
            vid_embedding_output.append(act_diff_embedding_output)

        if vid_embedding_output:
            # concatenate all vid embeddings
            vid_embedding_output = torch.cat(vid_embedding_output, 1)
            # add dropout on vid feature embedding
            vid_embedding_output = self.dropout(vid_embedding_output)

            # concatenate two embeddings
            if cap_embedding_output is not None:
                embedding_output = torch.cat((cap_embedding_output, vid_embedding_output), 1)
            else:
                embedding_output = vid_embedding_output
        else:
            # if no vid_embedding, you must have cap_embedding
            assert cap_embedding_output is not None, 'Exception: No input for the encoder in BertBoundaryModel'
            embedding_output = cap_embedding_output

        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask, head_mask=head_mask,
                                       encoder_history_states=encoder_history_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs


class BertCaptioningLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_smoothing = getattr(config, 'label_smoothing', 0)
        self.drop_worst_ratio = getattr(config, 'drop_worst_ratio', 0)
        self.drop_worst_after = getattr(config, 'drop_worst_after', 0)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='none')
        self.iter = 0

    def forward(self, logits, target):
        self.iter += 1
        eps = self.label_smoothing
        n_class = logits.size(1)
        one_hot = torch.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(logits)
        loss = self.kl(log_prb, one_hot).sum(1)

        if self.drop_worst_ratio > 0 and self.iter > self.drop_worst_after:
            loss, _ = torch.topk(loss,
                                 k=int(loss.shape[0] * (1 - self.drop_worst_ratio)),
                                 largest=False)

        loss = loss.mean()

        return loss


class BertForBoundaryCaptioning(CaptionPreTrainedModel):
    """
    Bert for Image Captioning.
    """

    def __init__(self, config):
        super(BertForBoundaryCaptioning, self).__init__(config)
        self.config = config
        self.bert = BertBoundaryModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.loss = BertCaptioningLoss(config)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        if hasattr(self.config, 'tie_weights') and self.config.tie_weights:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)
        freeze = False
        if hasattr(self.config, 'freeze_embedding'):
            freeze = self.config.freeze_embedding
        self.bert.embeddings.word_embeddings.weight.requires_grad = not freeze

    def forward(self, *args, **kwargs):
        is_decode = kwargs.get('is_decode', False)
        if is_decode:
            return self.generate(*args, **kwargs)
        else:
            return self.encode_forward(*args, **kwargs)

    def encode_forward(self, input_ids, attention_mask, obj_feats, frame_feats, frame_feats_diff,
                       act_feats, act_feats_diff, masked_pos, masked_ids=None, token_type_ids=None,
                       position_ids=None, head_mask=None, is_training=True, encoder_history_states=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, obj_feats=obj_feats, frame_feats=frame_feats,
                            frame_feats_diff=frame_feats_diff, act_feats=act_feats, act_feats_diff=act_feats_diff,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask, encoder_history_states=encoder_history_states)

        if is_training:
            sequence_output = outputs[0][:, :masked_pos.shape[-1], :]
            # num_masks_in_batch * hidden_size
            sequence_output_masked = sequence_output[masked_pos == 1, :]
            class_logits = self.cls(sequence_output_masked)
            masked_ids = masked_ids[masked_ids != 0]  # remove padding masks
            masked_loss = self.loss(class_logits.float(), masked_ids)
            outputs = (masked_loss, class_logits,) + outputs[2:]
        else:
            sequence_output = outputs[0][:, :input_ids.shape[-1], :]
            class_logits = self.cls(sequence_output)
            outputs = (class_logits,) + outputs[2:]
        return outputs

    def prepare_inputs_for_generation(self, curr_ids, past=None):
        # NOTE: if attention is on, it should be the token used to mask words in training
        mask_token_id = self.mask_token_id
        batch_size = curr_ids.shape[0]
        mask_ids = torch.full(
            (batch_size, 1), mask_token_id, dtype=torch.long, device=curr_ids.device
        )

        def _slice(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_token_len + self.od_labels_len)
            return t[:, start: end]

        def _remove_elements(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_token_len)
            return torch.cat([t[:, :start], t[:, end:]], dim=1)

        if past is None:
            input_ids = torch.cat([curr_ids, mask_ids], dim=1)

            curr_len = input_ids.shape[1]
            full_len = self.max_token_len
            if self.obj_feats is not None:
                full_len += self.obj_feats.shape[1]
            if self.frame_feats is not None:
                full_len += self.frame_feats.shape[1]
            if self.frame_feats_diff is not None:
                full_len += self.frame_feats_diff.shape[1]
            if self.act_feats is not None:
                full_len += self.act_feats.shape[1]
            if self.act_feats_diff is not None:
                full_len += self.act_feats_diff.shape[1]
            assert self.full_attention_mask.shape == (batch_size, full_len, full_len)

            def _remove_rows_cols(t, row_start, row_end, col_start, col_end):
                t00 = t[:, :row_start, :col_start]
                t01 = t[:, :row_start, col_end:]
                t10 = t[:, row_end:, :col_start]
                t11 = t[:, row_end:, col_end:]
                res = torch.cat([torch.cat([t00, t01], dim=2), torch.cat([t10, t11],
                                                                         dim=2)], dim=1)
                assert res.shape == (t.shape[0], t.shape[1] - row_end + row_start,
                                     t.shape[2] - col_end + col_start)
                return res

            seq_start = curr_len
            seq_end = self.max_token_len
            attention_mask = _remove_rows_cols(self.full_attention_mask, seq_start, seq_end, seq_start, seq_end)

            masked_pos = _remove_elements(self.full_masked_pos, seq_start, seq_end)
            token_type_ids = _remove_elements(self.full_token_type_ids, seq_start, seq_end)
            position_ids = _remove_elements(self.full_position_ids, seq_start, seq_end)
            obj_feats = self.obj_feats
            frame_feats = self.frame_feats
            frame_feats_diff = self.frame_feats_diff
            act_feats = self.act_feats
            act_feats_diff = self.act_feats_diff

        else:   # deprecated
            raise Exception
            last_token = curr_ids[:, -1:]
            # The representation of last token should be re-computed, because
            # it depends on both self-attention context and input tensor
            input_ids = torch.cat([last_token, mask_ids], dim=1)
            start_pos = curr_ids.shape[1] - 1
            end_pos = start_pos + input_ids.shape[1]
            masked_pos = _slice(self.full_masked_pos, start_pos, end_pos)
            token_type_ids = _slice(self.full_token_type_ids, start_pos, end_pos)
            position_ids = _slice(self.full_position_ids, start_pos, end_pos)

            img_feats = None
            assert past[0].shape[0] == batch_size
            if self.prev_encoded_layers is None:
                assert start_pos == 1  # the first token after BOS
                assert past[0].shape[1] == 2 + self.od_labels_len + self.img_seq_len
                # reorder to [od_labels, img_feats, sentence]
                self.prev_encoded_layers = [
                    torch.cat([x[:, 2:, :], x[:, :start_pos, :]], dim=1)
                    for x in past]
                s2s = self.full_attention_mask[:, :self.max_token_len,
                      :self.max_token_len]
                s2i = self.full_attention_mask[:, :self.max_token_len,
                      self.max_token_len:]
                i2s = self.full_attention_mask[:, self.max_token_len:,
                      :self.max_token_len]
                i2i = self.full_attention_mask[:, self.max_token_len:,
                      self.max_token_len:]
                self.full_attention_mask = torch.cat(
                    [torch.cat([i2i, i2s], dim=2),
                     torch.cat([s2i, s2s], dim=2)],
                    dim=1)
            else:
                assert start_pos > 1
                assert past[0].shape[1] == 2
                self.prev_encoded_layers = [torch.cat([x, p[:, :-1, :]], dim=1)
                                            for x, p in zip(self.prev_encoded_layers, past)]

            attention_mask = self.full_attention_mask[:,
                             self.od_labels_len + self.img_seq_len + start_pos: self.od_labels_len + self.img_seq_len + end_pos,
                             :self.od_labels_len + self.img_seq_len + end_pos]

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'obj_feats': obj_feats,
                'frame_feats': frame_feats, 'frame_feats_diff': frame_feats_diff, 'act_feats': act_feats,
                'act_feats_diff': act_feats_diff, 'masked_pos': masked_pos, 'token_type_ids': token_type_ids,
                'position_ids': position_ids, 'is_training': False, 'encoder_history_states': self.prev_encoded_layers}

    def get_output_embeddings(self):
        return self.decoder

    def generate(self, obj_feats, frame_feats, frame_feats_diff, act_feats,
                 act_feats_diff, attention_mask, masked_pos, token_type_ids=None,
                 position_ids=None, head_mask=None, input_ids=None, max_length=None,
                 do_sample=None, num_beams=None, temperature=None, top_k=None, top_p=None,
                 repetition_penalty=None, bos_token_id=None, pad_token_id=None,
                 eos_token_ids=None, mask_token_id=None, length_penalty=None,
                 num_return_sequences=None, num_keep_best=1, is_decode=None, fsm=None,
                 num_constraints=None, min_constraints_to_satisfy=None, use_hypo=False,
                 decoding_constraint_flag=None, bad_ending_ids=None, top_n_per_beam=2
                 ):
        """ Generates captions given image features
        """
        assert is_decode
        batch_size = input_ids.shape[0]
        self.max_token_len = max_length
        self.mask_token_id = mask_token_id
        self.prev_encoded_layers = None
        # NOTE: num_keep_best is not equavilant to num_return_sequences
        # num_keep_best is the number of hypotheses to keep in beam search
        # num_return_sequences is the repeating times of input, coupled with
        # do_sample=True can generate more than one samples per image
        self.num_keep_best = num_keep_best

        vocab_size = self.config.vocab_size
        num_fsm_states = 1

        assert input_ids.shape == (batch_size, self.max_token_len)
        input_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=input_ids.device
        )

        cur_len = input_ids.shape[1]
        if num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = self._expand_for_beams(input_ids, num_return_sequences)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if position_ids is None:
            position_ids = torch.arange(self.max_token_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand([batch_size, self.max_token_len])

        if token_type_ids is None:
            token_type_ids = torch.zeros([batch_size, self.max_token_len], dtype=torch.long, device=input_ids.device)

        num_expand = num_beams * num_fsm_states * num_return_sequences

        if obj_feats is None:
            self.obj_feats = None
        else:
            self.obj_feats = self._expand_for_beams(obj_feats, num_expand)

        if frame_feats is None:
            self.frame_feats = None
        else:
            self.frame_feats = self._expand_for_beams(frame_feats, num_expand)

        if frame_feats_diff is None:
            self.frame_feats_diff = None
        else:
            self.frame_feats_diff = self._expand_for_beams(frame_feats_diff, num_expand)

        if act_feats is None:
            self.act_feats = None
        else:
            self.act_feats = self._expand_for_beams(act_feats, num_expand)

        if act_feats_diff is None:
            self.act_feats_diff = None
        else:
            self.act_feats_diff = self._expand_for_beams(act_feats_diff, num_expand)

        self.full_attention_mask = self._expand_for_beams(attention_mask, num_expand)
        self.full_masked_pos = self._expand_for_beams(masked_pos, num_expand)
        self.full_token_type_ids = self._expand_for_beams(token_type_ids, num_expand)
        self.full_position_ids = self._expand_for_beams(position_ids, num_expand)
        self.full_head_mask = self._expand_for_beams(head_mask, num_expand)

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                length_penalty,
                num_beams,
                top_n_per_beam,
                vocab_size,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
            )
        return output

    def _expand_for_beams(self, x, num_expand):
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_expand, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x

    def _do_output_past(self, outputs):
        return len(outputs) > 1


class BertForBoundaryCaptioningAdvanced(CaptionPreTrainedModel):
    """
    Bert for Image Captioning.
    """

    def __init__(self, config):
        super(BertForBoundaryCaptioningAdvanced, self).__init__(config)
        self.config = config
        self.bert = BertBoundaryModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.loss = BertCaptioningLoss(config)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        if hasattr(self.config, 'tie_weights') and self.config.tie_weights:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)
        freeze = False
        if hasattr(self.config, 'freeze_embedding'):
            freeze = self.config.freeze_embedding
        self.bert.embeddings.word_embeddings.weight.requires_grad = not freeze

    def forward(self, *args, **kwargs):
        is_decode = kwargs.get('is_decode', False)
        if is_decode:
            return self.generate(*args, **kwargs)
        else:
            return self.encode_forward(*args, **kwargs)

    def encode_forward(self, input_ids, attention_mask, obj_feats, frame_feats, frame_feats_diff,
                       act_feats, act_feats_diff, masked_pos, masked_ids=None, token_type_ids=None,
                       position_ids=None, head_mask=None, is_training=True, encoder_history_states=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, obj_feats=obj_feats, frame_feats=frame_feats,
                            frame_feats_diff=frame_feats_diff, act_feats=act_feats, act_feats_diff=act_feats_diff,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask, encoder_history_states=encoder_history_states)

        if is_training:
            sequence_output = outputs[0][:, :masked_pos.shape[-1], :]
            # num_masks_in_batch * hidden_size
            sequence_output_masked = sequence_output[masked_pos == 1, :]
            class_logits = self.cls(sequence_output_masked)
            masked_ids = masked_ids[masked_ids != 0]  # remove padding masks
            masked_loss = self.loss(class_logits.float(), masked_ids)
            outputs = (masked_loss, class_logits,) + outputs[2:]
        else:
            sequence_output = outputs[0][:, :input_ids.shape[-1], :]
            class_logits = self.cls(sequence_output)
            outputs = (class_logits,) + outputs[2:]
        return outputs

    def prepare_inputs_for_generation(self, curr_ids, past=None):
        # NOTE: if attention is on, it should be the token used to mask words in training
        mask_token_id = self.mask_token_id
        batch_size = curr_ids.shape[0]
        mask_ids = torch.full(
            (batch_size, 1), mask_token_id, dtype=torch.long, device=curr_ids.device
        )

        def _slice(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_token_len + self.od_labels_len)
            return t[:, start: end]

        def _remove_elements(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_token_len)
            return torch.cat([t[:, :start], t[:, end:]], dim=1)

        if past is None:
            input_ids = torch.cat([curr_ids, mask_ids], dim=1)

            curr_len = input_ids.shape[1]
            full_len = self.max_token_len
            if self.obj_feats is not None:
                full_len += self.obj_feats.shape[1]
            if self.frame_feats is not None:
                full_len += self.frame_feats.shape[1]
            if self.frame_feats_diff is not None:
                full_len += self.frame_feats_diff.shape[1]
            if self.act_feats is not None:
                full_len += self.act_feats.shape[1]
            if self.act_feats_diff is not None:
                full_len += self.act_feats_diff.shape[1]
            assert self.full_attention_mask.shape == (batch_size, full_len, full_len)

            def _remove_rows_cols(t, row_start, row_end, col_start, col_end):
                t00 = t[:, :row_start, :col_start]
                t01 = t[:, :row_start, col_end:]
                t10 = t[:, row_end:, :col_start]
                t11 = t[:, row_end:, col_end:]
                res = torch.cat([torch.cat([t00, t01], dim=2), torch.cat([t10, t11],
                                                                         dim=2)], dim=1)
                assert res.shape == (t.shape[0], t.shape[1] - row_end + row_start,
                                     t.shape[2] - col_end + col_start)
                return res

            seq_start = curr_len
            seq_end = self.max_token_len
            attention_mask = _remove_rows_cols(self.full_attention_mask, seq_start, seq_end, seq_start, seq_end)

            masked_pos = _remove_elements(self.full_masked_pos, seq_start, seq_end)
            token_type_ids = _remove_elements(self.full_token_type_ids, seq_start, seq_end)
            position_ids = _remove_elements(self.full_position_ids, seq_start, seq_end)
            obj_feats = self.obj_feats
            frame_feats = self.frame_feats
            frame_feats_diff = self.frame_feats_diff
            act_feats = self.act_feats
            act_feats_diff = self.act_feats_diff

        else:   # deprecated
            raise Exception
            last_token = curr_ids[:, -1:]
            # The representation of last token should be re-computed, because
            # it depends on both self-attention context and input tensor
            input_ids = torch.cat([last_token, mask_ids], dim=1)
            start_pos = curr_ids.shape[1] - 1
            end_pos = start_pos + input_ids.shape[1]
            masked_pos = _slice(self.full_masked_pos, start_pos, end_pos)
            token_type_ids = _slice(self.full_token_type_ids, start_pos, end_pos)
            position_ids = _slice(self.full_position_ids, start_pos, end_pos)

            img_feats = None
            assert past[0].shape[0] == batch_size
            if self.prev_encoded_layers is None:
                assert start_pos == 1  # the first token after BOS
                assert past[0].shape[1] == 2 + self.od_labels_len + self.img_seq_len
                # reorder to [od_labels, img_feats, sentence]
                self.prev_encoded_layers = [
                    torch.cat([x[:, 2:, :], x[:, :start_pos, :]], dim=1)
                    for x in past]
                s2s = self.full_attention_mask[:, :self.max_token_len,
                      :self.max_token_len]
                s2i = self.full_attention_mask[:, :self.max_token_len,
                      self.max_token_len:]
                i2s = self.full_attention_mask[:, self.max_token_len:,
                      :self.max_token_len]
                i2i = self.full_attention_mask[:, self.max_token_len:,
                      self.max_token_len:]
                self.full_attention_mask = torch.cat(
                    [torch.cat([i2i, i2s], dim=2),
                     torch.cat([s2i, s2s], dim=2)],
                    dim=1)
            else:
                assert start_pos > 1
                assert past[0].shape[1] == 2
                self.prev_encoded_layers = [torch.cat([x, p[:, :-1, :]], dim=1)
                                            for x, p in zip(self.prev_encoded_layers, past)]

            attention_mask = self.full_attention_mask[:,
                             self.od_labels_len + self.img_seq_len + start_pos: self.od_labels_len + self.img_seq_len + end_pos,
                             :self.od_labels_len + self.img_seq_len + end_pos]

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'obj_feats': obj_feats,
                'frame_feats': frame_feats, 'frame_feats_diff': frame_feats_diff, 'act_feats': act_feats,
                'act_feats_diff': act_feats_diff, 'masked_pos': masked_pos, 'token_type_ids': token_type_ids,
                'position_ids': position_ids, 'is_training': False, 'encoder_history_states': self.prev_encoded_layers}

    def get_output_embeddings(self):
        return self.decoder

    def generate(self, obj_feats, frame_feats, frame_feats_diff, act_feats,
                 act_feats_diff, attention_mask, masked_pos, token_type_ids=None,
                 position_ids=None, head_mask=None, input_ids=None, max_length=None,
                 do_sample=None, num_beams=None, temperature=None, top_k=None, top_p=None,
                 repetition_penalty=None, bos_token_id=None, pad_token_id=None,
                 eos_token_ids=None, mask_token_id=None, length_penalty=None,
                 num_return_sequences=None, num_keep_best=1, is_decode=None, fsm=None,
                 num_constraints=None, min_constraints_to_satisfy=None, use_hypo=False,
                 decoding_constraint_flag=None, bad_ending_ids=None, top_n_per_beam=2
                 ):
        """ Generates captions given image features
        """
        assert is_decode
        batch_size = input_ids.shape[0]
        self.max_token_len = max_length
        self.mask_token_id = mask_token_id
        self.prev_encoded_layers = None
        # NOTE: num_keep_best is not equavilant to num_return_sequences
        # num_keep_best is the number of hypotheses to keep in beam search
        # num_return_sequences is the repeating times of input, coupled with
        # do_sample=True can generate more than one samples per image
        self.num_keep_best = num_keep_best

        vocab_size = self.config.vocab_size
        num_fsm_states = 1

        assert input_ids.shape == (batch_size, self.max_token_len)
        input_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=input_ids.device
        )

        cur_len = input_ids.shape[1]
        if num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = self._expand_for_beams(input_ids, num_return_sequences)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if position_ids is None:
            position_ids = torch.arange(self.max_token_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand([batch_size, self.max_token_len])

        if token_type_ids is None:
            token_type_ids = torch.zeros([batch_size, self.max_token_len], dtype=torch.long, device=input_ids.device)

        num_expand = num_beams * num_fsm_states * num_return_sequences

        if obj_feats is None:
            self.obj_feats = None
        else:
            self.obj_feats = self._expand_for_beams(obj_feats, num_expand)

        if frame_feats is None:
            self.frame_feats = None
        else:
            self.frame_feats = self._expand_for_beams(frame_feats, num_expand)

        if frame_feats_diff is None:
            self.frame_feats_diff = None
        else:
            self.frame_feats_diff = self._expand_for_beams(frame_feats_diff, num_expand)

        if act_feats is None:
            self.act_feats = None
        else:
            self.act_feats = self._expand_for_beams(act_feats, num_expand)

        if act_feats_diff is None:
            self.act_feats_diff = None
        else:
            self.act_feats_diff = self._expand_for_beams(act_feats_diff, num_expand)

        self.full_attention_mask = self._expand_for_beams(attention_mask, num_expand)
        self.full_masked_pos = self._expand_for_beams(masked_pos, num_expand)
        self.full_token_type_ids = self._expand_for_beams(token_type_ids, num_expand)
        self.full_position_ids = self._expand_for_beams(position_ids, num_expand)
        self.full_head_mask = self._expand_for_beams(head_mask, num_expand)

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                length_penalty,
                num_beams,
                top_n_per_beam,
                vocab_size,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
            )
        return output

    def _expand_for_beams(self, x, num_expand):
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_expand, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x

    def _do_output_past(self, outputs):
        return len(outputs) > 1


class BertForVideoRetrieval(CaptionPreTrainedModel):
    """
    Bert for Image Captioning.
    """

    def __init__(self, config):
        super(BertForVideoRetrieval, self).__init__(config)
        self.config = config
        self.bert = BertBoundaryModel(config)
        self.ctx_bert = BertBoundaryModel(config)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        if hasattr(self.config, 'tie_weights') and self.config.tie_weights:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)
        freeze = False
        if hasattr(self.config, 'freeze_embedding'):
            freeze = self.config.freeze_embedding
        self.bert.embeddings.word_embeddings.weight.requires_grad = not freeze

    def forward(self, input_ids, attention_mask, obj_feats, frame_feats, frame_feats_diff, act_feats,
                act_feats_diff, token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                do_cap=True, do_ctx=True):

        # get caption embedding
        if do_cap:
            cap_attention_mask = attention_mask[:, :input_ids.shape[1], :input_ids.shape[1]]
            cap_output = self.bert(input_ids, attention_mask=cap_attention_mask, position_ids=position_ids,
                                      token_type_ids=token_type_ids, head_mask=head_mask,
                                      encoder_history_states=encoder_history_states)
            cap_embedding = cap_output[0][:, 0, :]
        else:
            cap_embedding = None

        # get visual embedding
        if do_ctx:
            if input_ids is not None:
                ctx_attention_mask = attention_mask[:, input_ids.shape[1]:, input_ids.shape[1]:]
            else:
                ctx_attention_mask = attention_mask
            ctx_output = self.ctx_bert(attention_mask=ctx_attention_mask, obj_feats=obj_feats, frame_feats=frame_feats,
                                       frame_feats_diff=frame_feats_diff, act_feats=act_feats,
                                       act_feats_diff=act_feats_diff, head_mask=head_mask,
                                       encoder_history_states=encoder_history_states)
            ctx_embedding = ctx_output[0][:, 0, :]
        else:
            ctx_embedding = None

        return cap_embedding, ctx_embedding


class BertForVideoRetrievalOneStream(CaptionPreTrainedModel):
    """
    Bert for Image Captioning.
    """

    def __init__(self, config):
        super(BertForVideoRetrievalOneStream, self).__init__(config)
        self.config = config
        self.bert = BertBoundaryModel(config)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        if hasattr(self.config, 'tie_weights') and self.config.tie_weights:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)
        freeze = False
        if hasattr(self.config, 'freeze_embedding'):
            freeze = self.config.freeze_embedding
        self.bert.embeddings.word_embeddings.weight.requires_grad = not freeze

    def forward(self, input_ids, attention_mask, obj_feats, frame_feats, frame_feats_diff, act_feats,
                act_feats_diff, token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                do_cap=True, do_ctx=True):

        # get caption embedding
        if do_cap:
            cap_attention_mask = attention_mask[:, :input_ids.shape[1], :input_ids.shape[1]]
            cap_output = self.bert(input_ids, attention_mask=cap_attention_mask, position_ids=position_ids,
                                      token_type_ids=token_type_ids, head_mask=head_mask,
                                      encoder_history_states=encoder_history_states)
            cap_embedding = cap_output[0][:, 0, :]
        else:
            cap_embedding = None

        # get visual embedding
        if do_ctx:
            if input_ids is not None:
                ctx_attention_mask = attention_mask[:, input_ids.shape[1]:, input_ids.shape[1]:]
            else:
                ctx_attention_mask = attention_mask
            ctx_output = self.bert(attention_mask=ctx_attention_mask, obj_feats=obj_feats, frame_feats=frame_feats,
                                       frame_feats_diff=frame_feats_diff, act_feats=act_feats,
                                       act_feats_diff=act_feats_diff, head_mask=head_mask,
                                       encoder_history_states=encoder_history_states)
            ctx_embedding = ctx_output[0][:, 0, :]
        else:
            ctx_embedding = None

        return cap_embedding, ctx_embedding


class BertForPWLocating(CaptionPreTrainedModel):

    def __init__(self, config):
        super(BertForPWLocating, self).__init__(config)
        self.config = config
        self.bert = BertBoundaryModel(config)
        self.cls = nn.Linear(config.hidden_size, 2)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        if hasattr(self.config, 'tie_weights') and self.config.tie_weights:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)
        freeze = False
        if hasattr(self.config, 'freeze_embedding'):
            freeze = self.config.freeze_embedding
        self.bert.embeddings.word_embeddings.weight.requires_grad = not freeze

    def forward(self, input_ids, attention_mask, obj_feats, frame_feats, frame_feats_diff, act_feats, act_feats_diff,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, obj_feats=obj_feats, frame_feats=frame_feats,
                            frame_feats_diff=frame_feats_diff, act_feats=act_feats, act_feats_diff=act_feats_diff,
                            position_ids=position_ids, token_type_ids=token_type_ids, head_mask=head_mask,
                            encoder_history_states=encoder_history_states)

        output = self.cls(outputs[0][:, 0, :])

        return output