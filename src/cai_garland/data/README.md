---
dataset_info:
- config_name: no_registers
  features:
  - name: source
    dtype: string
  - name: target
    dtype: string
  splits:
  - name: train
    num_bytes: 66611039
    num_examples: 109458
  - name: validation
    num_bytes: 358098
    num_examples: 687
  - name: test
  download_size: 0
  dataset_size: 66969137
- config_name: no_registers_no_splits
  features:
  - name: source
    dtype: string
  - name: target
    dtype: string
  splits:
  - name: train
    num_bytes: 66969137
    num_examples: 110145
  download_size: 0
  dataset_size: 66969137
- config_name: registers_3_only
  features:
  - name: source
    dtype: string
  - name: target
    dtype: string
  splits:
  - name: train
    num_bytes: 118153022
    num_examples: 116371
  - name: validation
    num_bytes: 582773
    num_examples: 695
  - name: test
  download_size: 0
  dataset_size: 118735795
- config_name: registers_3_only_no_splits
  features:
  - name: source
    dtype: string
  - name: target
    dtype: string
  splits:
  - name: train
    num_bytes: 118735795
    num_examples: 117066
  download_size: 0
  dataset_size: 118735795
- config_name: registers_2_only
  features:
  - name: source
    dtype: string
  - name: target
    dtype: string
  splits:
  - name: train
    num_bytes: 80919439
    num_examples: 113136
  - name: validation
    num_bytes: 414543
    num_examples: 694
  - name: test
  download_size: 0
  dataset_size: 81333982
- config_name: raw_with_context
  features:
  - name: source
    dtype: string
  - name: target
    dtype: string
  splits:
  - name: train
    num_bytes: 48422317
    num_examples: 140717
  - name: validation
    num_bytes: 221192
    num_examples: 820
  - name: test
  download_size: 0
  dataset_size: 48643509
- config_name: raw_with_context_cased
  features:
  - name: source
    dtype: string
  - name: target
    dtype: string
  splits:
  - name: train
    num_bytes: 69840871
    num_examples: 148841
  - name: validation
    num_bytes: 311565
    num_examples: 836
  - name: test
  download_size: 0
  dataset_size: 70152436
- config_name: raw_with_bidirectional_context
  features:
  - name: source
    dtype: string
  - name: target
    dtype: string
  splits:
  - name: train
    num_bytes: 89142800
    num_examples: 142957
  - name: validation
    num_bytes: 430937
    num_examples: 832
  - name: test
  download_size: 0
  dataset_size: 89573737
- config_name: nllb_augmentation
  features:
  - name: source
    dtype: string
  - name: target
    dtype: string
  splits:
  - name: bod_Tibt.ita_Latn
    num_bytes: 53006009
    num_examples: 301924
  - name: bod_Tibt.eng_Latn
    num_bytes: 52804520
    num_examples: 301748
  download_size: 0
  dataset_size: 105810529
- config_name: mined
  features:
  - name: source
    dtype: string
  - name: target
    dtype: string
  - name: score
    dtype: float32
  splits:
  - name: train
    num_bytes: 89595387
    num_examples: 234897
  download_size: 0
  dataset_size: 89595387
- config_name: largest_parallel_with_context
  features:
  - name: source
    dtype: string
  - name: target
    dtype: string
  splits:
  - name: train
    num_bytes: 201755790
    num_examples: 361308
  - name: validation
    num_bytes: 375791
    num_examples: 831
  - name: test
  download_size: 0
  dataset_size: 202131581
---
