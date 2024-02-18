# SpanSynLSTM
Pytorch implementation of Span prediction model and Sequence labeling framework for Named Entity Recognition, which incorporating the contextual information and the structured dependency trees information with Synergized-LSTM (Syn-LSTM).

# Model Architecture

# Requirement
Python 3.7

Pytorch 1.4.0

Transformers 3.3.1

# Performance
learning_rate: 2e-05, batch_size: 20(30)

F1:torch.cat((all_span_rep, att_span_emb) > torch.cat((all_span_rep, spanlen_rep) under ELECTRA-base-Syn-LSTM-Span

all_span_rep = self._endpoint_span_extractor(feature_out, all_span_ids.long())  
att_span_emb = self.attentive_span_extractor(feature_out, all_span_ids.long())  
all_span_rep = torch.cat((all_span_rep, att_span_emb), dim=-1)  

all_span_rep = self._endpoint_span_extractor(feature_out, all_span_ids.long())  
spanlen_rep = self.spanLen_embedding(all_span_lens)  # (bs, n_span, len_dim)  
spanlen_rep = functional.relu(spanlen_rep)  
all_span_rep = torch.cat((all_span_rep, spanlen_rep), dim=-1)

| Model  | Dataset | Pretrained Model |P | R | F1 |
| ------------- | ------------- |-------------|------------- |------------- |------------- |
| ELECTRA-base-Syn-LSTM-CRF  | Chinese  | chinese-electra-180g-base-discriminator |77.58  |80.80  |79.16  |
| ELECTRA-base-Syn-LSTM-Span | chinese  | chinese-electra-180g-base-discriminator |78.79  |82.05  | 80.39 |
| ELECTRA-base-CRF  | Chinese  |chinese-electra-180g-base-discriminator |- |-  |-  |
| ELECTRA-base-Span  | Chinese  |chinese-electra-180g-base-discriminator |79.46 |80.73  |80.09 |

| Model  | Dataset |  Pretrained Model |P | R | F1 |
| ------------- | --------------|--------------|--------------|--------------|------------- |
| Roberta-base-Syn-LSTM-CRF  | OntoNotes 5  |RoBERTa-base|  90.19  | 90.94  | 90.56  |
| Roberta-base-Syn-LSTM-Span | OntoNotes 5  |RoBERTa-base|  90.30  | 91.35  | 90.82  |
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

