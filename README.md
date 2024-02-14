# SpanSynLSTM
Pytorch implementation of Span prediction model and Sequence labeling framework for Named Entity Recognition, which incorporating the contextual information and the structured dependency trees information with Synergized-LSTM (Syn-LSTM).

# Model Architecture

# Requirement
Python 3.7

Pytorch 1.4.0

Transformers 3.3.1

# Performance

| Model  | Dataset | Pretrained Model |P | R | F1 |
| ------------- | ------------- |-------------|------------- |------------- |------------- |
| Roberta-base-Syn-LSTM-CRF  | Chinese  | chinese-electra-180g-base-discriminator |-  |-  |-  |
| Roberta-base-Syn-LSTM-Span | chinese  | chinese-electra-180g-base-discriminator |78.37  |81.55  | 79.92 |
| Roberta-base-CRF  | Chinese  |chinese-electra-180g-base-discriminator |- |-  |-  |
| Roberta-base-Span  | Chinese  |chinese-electra-180g-base-discriminator |79.46 |80.73  |80.09 |

| Model  | Dataset |  Pretrained Model |P | R | F1 |
| ------------- | --------------|--------------|--------------|--------------|------------- |
| Roberta-base-Syn-LSTM-CRF  | OntoNotes 5  |RoBERTa-base|  90.19  | 90.94  | 90.56  |
| Roberta-base-Syn-LSTM-Span | OntoNotes 5  |RoBERTa-base|  90.59  | 90.97  | 90.78  |
| Roberta-base-CRF  | OntoNotes 5  |RoBERTa-base|  90.17  | 91.34  | 90.75  |
| Roberta-base-Span | OntoNotes 5  |RoBERTa-base|  90.64  | 90.89  | 90.77  |
| Roberta-base-CRF  | CoNLL-2003  |RoBERTa-base|92.00 | 93.04  | 92.52  |
| Roberta-base-Span  | CoNLL-2003  | RoBERTa-base|92.47 | 92.81 | 92.65 |

# Running
Default:

    python transformers_trainer.py
    
To train the model with other datasets, parser_mode crf or span, dep_model dggcn or none:

    python transformers_trainer.py --mode=train --dataset=ontonotes --embedder_type=roberta-base --parser_mode span --dep_model dggcn

For more detailed usage, please refer to the [pytorch_neural_crf project](https://github.com/allanj/pytorch_neural_crf)

# Related Repo
The code are created based on the codes of the paper ["SPANNER: Named Entity Re-/Recognition as Span Prediction"](https://github.com/neulab/spanner), ACL 2021
["Better Feature Integration for Named Entity Recognition"](https://github.com/xuuuluuu/SynLSTM-for-NER?tab=readme-ov-file#related-repo), NAACL 2021

