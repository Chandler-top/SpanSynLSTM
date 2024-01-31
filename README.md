# SpanSynLSTM
Pytorch implementation of Span prediction model and Sequence labeling framework for Named Entity Recognition

# Model Architecture

# Requirement
Python 3.7

Pytorch 1.4.0

Transformers 3.3.1

CUDA 10.1, 10.2

# Performance

| Model  | Dataset | F1 |
| ------------- | ------------- |------------- |
| Roberta-base-Syn-LSTM-CRF  | Chinese  |  -  |
| Roberta-base-Syn-LSTM-Span | Chinese  |  -  |
| Roberta-base-CRF  | Chinese  | - |
| Roberta-base-Span  | Chinese  | - |

| Model  | Dataset | F1 |
| ------------- | ------------- |------------- |
| Roberta-base-Syn-LSTM-CRF  | English  |  -  |
| Roberta-base-Syn-LSTM-Span | English  |  -  |
| Roberta-base-CRF  | English  | - |
| Roberta-base-Span  | English  | - |

# Running

    python transformers_trainer.py
    
To train the model with other datasets:

    python transformers_trainer.py --mode=train --dataset=ontonotes --embedder_type=roberta-base

For more detailed usage, please refer to the [pytorch_neural_crf project](https://github.com/allanj/pytorch_neural_crf)

# Related Repo
The code are created based on the codes of the paper ["SPANNER: Named Entity Re-/Recognition as Span Prediction"](https://github.com/neulab/spanner), ACL 2021
