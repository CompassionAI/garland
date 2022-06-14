# Project Garland

Machine translation from classical literary Tibetan. Current focus is on:

- A curated approach to machine translation that's very task-focused
- Custom neural machine translation models in Huggingface

Eventually these techniques may generalize to other low resource languages.

## A note on configuration files

The configuration files for runnable code should be maintained to be the current champion methodology for whatever that code is doing. For training, the configs should contain the hyperparameters of the champion model. For inference, the configs should fix the example of how to best do whatever task.