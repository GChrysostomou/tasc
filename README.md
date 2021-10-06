**## TaSc**



Repository for the paper "Improving Attention Interpretability with Task-specific Information in Text Classification" published at ACL 2021 available at : ("Improving Attention Interpretability with Task-specific Information in Text Classification")[https://aclanthology.org/2021.acl-long.40/]



**## Prerequisites**



Install necessary packages by using the files  [conda_reqs.txt](https://github.com/GChrysostomou/eacl_tasc/blob/master/conda_reqs.txt) and  [pip_reqs.txt](https://github.com/GChrysostomou/eacl_tasc/blob/master/pip_reqs.txt)  



\```

conda create --name tasc --file  conda_reqs.txt

conda activate tasc

pip install -r pip_reqs.txt

python -m spacy download en

\```



**## Training and Evaluating the models**



You can train and save the models with [train_eval_bc.py](https://github.com/GChrysostomou/eacl_tasc/blob/master/train_eval_bc.py) script with the following options:



\* dataset : **{sst, twitter, mimicanemia, imdb, agnews}**

\* encoder : **{lstm, gru, mlp, cnn, bert}** 

\* data_dir : **directory for your data followed by** /

\* model_dir : **direcrory to save models followed by** /

\* experiments_dir : **direcrory to save experiment results followed by** /

\* mechanisms: **selection of attention mechanism with options {tanh, dot}**

\* operation : **operation over tasc score generation with option {sum-over, max-pool, mean-pool}**

\* lin : **apply Lin-TaSc**

\* feat : **apply Feat-TaSc**

\* conv : **apply Conv-TaSc**



Example script (without TaSc):



\```python train_eval_bc.py -dataset sst -encoder lstm -mechanism dot -data_dir data/ -model_dir models/ ```



Example script (with Lin-TaSc):



\```

python train_eval_bc.py -dataset mimicanemia -encoder bert -mechanism tanh -data_dir data/ -model_dir models/ -lin

\```



**## Summarising results**



Following the evaluation for multiple models / attention mechanisms , with and without TaSc, you can use [produce_reports.py](https://github.com/GChrysostomou/eacl_tasc/blob/master/produce_reports.py) to create tables in latex and as csv for the full stack of results (as a comparison), a table for comparing with other explanation techniques, results across attention mechanism, encoder and dataset. The file can be run with the following options:



\* datasets: **list of datasets with permissible datasets listed above**

\* encoders: **list of encoders with permissible encoders listed above**

\* experiments_dir: **directory that contains saved results to use for summarising followed by /**

\* mechanisms: **selection of attention mechanism with options {Tanh, Dot}**

\* tasc_ver : **tasc version with options {lin, feat, conv}**



**## Citation**



Please cite:



\```

@inproceedings{chrysostomou-aletras-2021-improving,

​    title = "Improving the Faithfulness of Attention-based Explanations with Task-specific Information for Text Classification",

​    author = "Chrysostomou, George  and

​      Aletras, Nikolaos",

​    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",

​    month = aug,

​    year = "2021",

​    address = "Online",

​    publisher = "Association for Computational Linguistics",

​    url = "https://aclanthology.org/2021.acl-long.40",

​    doi = "10.18653/v1/2021.acl-long.40",

​    pages = "477--488",

​    abstract = "Neural network architectures in natural language processing often use attention mechanisms to produce probability distributions over input token representations. Attention has empirically been demonstrated to improve performance in various tasks, while its weights have been extensively used as explanations for model predictions. Recent studies (Jain and Wallace, 2019; Serrano and Smith, 2019; Wiegreffe and Pinter, 2019) have showed that it cannot generally be considered as a faithful explanation (Jacovi and Goldberg, 2020) across encoders and tasks. In this paper, we seek to improve the faithfulness of attention-based explanations for text classification. We achieve this by proposing a new family of Task-Scaling (TaSc) mechanisms that learn task-specific non-contextualised information to scale the original attention weights. Evaluation tests for explanation faithfulness, show that the three proposed variants of TaSc improve attention-based explanations across two attention mechanisms, five encoders and five text classification datasets without sacrificing predictive performance. Finally, we demonstrate that TaSc consistently provides more faithful attention-based explanations compared to three widely-used interpretability techniques.",

}

\```
