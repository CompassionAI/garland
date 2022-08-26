from transformers import Seq2SeqTrainer


class CAISeq2SeqTrainer(Seq2SeqTrainer):
    """Wrapper to accomodate changes needed for our encoder-decoder models."""

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None
    ):
        with self.model.prepare_model_for_generation(
            inputs.data['context_embedding'],
            inputs.data['context_embedding_mask']
        ):
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)