from transformers import Seq2SeqTrainer


class CAISeq2SeqTrainer(Seq2SeqTrainer):
    """Wrapper to accomodate changes needed for our encoder-decoder models."""

    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics
        )
        self.generation_kwargs = {}

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None
    ):
        if hasattr(self.model, "prepare_model_for_generation") and "context_embedding" in inputs.data:
            with self.model.prepare_model_for_generation(
                inputs.data['context_embedding'],
                inputs.data['context_embedding_mask']
            ):
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
        **gen_kwargs
    ):
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **(gen_kwargs | self.generation_kwargs))