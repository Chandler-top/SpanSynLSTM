# SpanSynLSTM
Pytorch implementation of Span prediction model and Sequence labeling framework for Named Entity Recognition, which incorporating the contextual information and the structured dependency trees information with Synergized-LSTM (Syn-LSTM).

# Model Architecture

# Requirement
Python 3.7

Pytorch 1.4.0

AllenNLP 1.4.0

Transformers 3.3.1

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
Default:

    python transformers_trainer.py
    
To train the model with other datasets, parser_mode crf or span, dep_model dggcn or none:

    python transformers_trainer.py --mode=train --dataset=ontonotes --embedder_type=roberta-base --parser_mode span --dep_model dggcn

For more detailed usage, please refer to the [pytorch_neural_crf project](https://github.com/allanj/pytorch_neural_crf)

# Related Repo
The code are created based on the codes of the paper ["SPANNER: Named Entity Re-/Recognition as Span Prediction"](https://github.com/neulab/spanner), ACL 2021
["SPANNER: Named Entity Re-/Recognition as Span Prediction"](https://github.com/neulab/spanner), ACL 2021
["Better Feature Integration for Named Entity Recognition"](https://github.com/xuuuluuu/SynLSTM-for-NER?tab=readme-ov-file#related-repo), NAACL 2021

