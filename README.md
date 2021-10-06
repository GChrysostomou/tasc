## TaSc

Repository for the paper "Improving Attention Interpretability with Task-specific Information in Text Classification" published at ACL 2021 available at : ["Improving Attention Interpretability with Task-specific Information in Text Classification"](https://aclanthology.org/2021.acl-long.40/)



## Prerequisites



Install necessary packages by using the files  [conda_reqs.txt](https://github.com/GChrysostomou/eacl_tasc/blob/master/conda_reqs.txt) and  [pip_reqs.txt](https://github.com/GChrysostomou/eacl_tasc/blob/master/pip_reqs.txt)  

\```

conda create --name tasc --file  conda_reqs.txt

conda activate tasc

pip install -r pip_reqs.txt

python -m spacy download en

\```

## Training and Evaluating the models

You can train and save the models with [train_eval_bc.py](https://github.com/GChrysostomou/eacl_tasc/blob/master/train_eval_bc.py) script with the following options:

* dataset : **{sst, twitter, mimicanemia, imdb, agnews}**
* encoder : **{lstm, gru, mlp, cnn, bert}** 

* data_dir : **directory for your data followed by** /

* model_dir : **direcrory to save models followed by** /

* experiments_dir : **direcrory to save experiment results followed by** /

* mechanisms: **selection of attention mechanism with options {tanh, dot}**

* operation : **operation over tasc score generation with option {sum-over, max-pool, mean-pool}**

* lin : **apply Lin-TaSc**
*  feat : **apply Feat-TaSc**

* conv : **apply Conv-TaSc**



Example script (without TaSc):



```python
python train_eval_bc.py -dataset sst -encoder lstm -mechanism dot -data_dir data/ -model_dir models/ 
```



Example script (with Lin-TaSc):

```python
python train_eval_bc.py -dataset mimicanemia -encoder bert -mechanism tanh -data_dir data/ -model_dir models/ -lin
```



## Summarising results

Following the evaluation for multiple models / attention mechanisms , with and without TaSc, you can use [produce_reports.py](https://github.com/GChrysostomou/eacl_tasc/blob/master/produce_reports.py) to create tables in latex and as csv for the full stack of results (as a comparison), a table for comparing with other explanation techniques, results across attention mechanism, encoder and dataset. The file can be run with the following options:

* datasets: **list of datasets with permissible datasets listed above**

* encoders: **list of encoders with permissible encoders listed above**

* experiments_dir: **directory that contains saved results to use for summarising followed by /**

* mechanisms: **selection of attention mechanism with options {Tanh, Dot}**

* tasc_ver : **tasc version with options {lin, feat, conv}**



## Citation

Please cite:

```latex
@inproceedings{chrysostomou-aletras-2021-improving,
	title = "Improving the Faithfulness of Attention-based Explanations with Task-specific Information for Text Classification",
	author = "Chrysostomou, George  and Aletras, Nikolaos",
	booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
	month = aug,
	year = "2021",
	address = "Online",
	publisher = "Association for Computational Linguistics",
	url = "https://aclanthology.org/2021.acl-long.40",
	doi = "10.18653/v1/2021.acl-long.40",
	pages = "477--488"
}
```


