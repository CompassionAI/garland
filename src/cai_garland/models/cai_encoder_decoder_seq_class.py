import torch

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .cai_encoder_decoder import CAIEncoderDecoderConfig, CAIEncoderDecoderModel
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from transformers.models.bart.modeling_bart import BartClassificationHead
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqSequenceClassifierOutput


class CAIEncoderDecoderForSequenceClassification(PreTrainedModel):
    """A wrapper class for an encoder-decoder model used for sequence classification."""

    config_class = CAIEncoderDecoderConfig
    base_model_prefix = "encoder_decoder_for_sequence_classification"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = [r"encoder.version", r"decoder.version"]

    def _init_weights(self, module):
        std = self.config.decoder.init_std
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def __init__(
        self,
        config=None,
        encoder=None,
        decoder=None,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        model = CAIEncoderDecoderModel(config, encoder, decoder)
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.classification_head = BartClassificationHead(
            config.decoder.d_model,
            config.decoder.d_model,
            config.decoder.num_labels,
            config.decoder.classifier_dropout,
        )
        self._init_weights(self.classification_head.dense)
        self._init_weights(self.classification_head.out_proj)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            raise ValueError("We should never land here with encoder-decoder sequence classification")

        # Decode
        decoder_outputs = self.decoder.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )
        hidden_states = decoder_outputs.hidden_states[-1]  # last hidden state

        eos_mask = decoder_input_ids.eq(self.config.decoder.eos_token_id)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=getattr(decoder_outputs, "decoder_attentions", None),
            cross_attentions=getattr(decoder_outputs, "cross_attentions", None),
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=getattr(encoder_outputs, "hidden_states", None),
            encoder_attentions=getattr(encoder_outputs, "attentions", None)
        )

# This injects our new model type into the Hugging Face factory for models without having to modify their code base. If
#   ever this is contributed to the Transformers main branch, this should be moved.
AutoModelForSequenceClassification.register(CAIEncoderDecoderConfig, CAIEncoderDecoderForSequenceClassification)
