import glob 
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.offline as pyo
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def capture_all_fraction(results_dir = str, datasets = list(), encoders = list(), tasc_approach = list(), mechanisms = list()):
    
    
    tasc_mapper = {
        "" : "baseline",
        "lin_" : "lin",
        "feat_" : "feat",
        "conv_" : "conv"
    }
    
    
    cwd = os.getcwd()

    all_results = []

    
    for dataset in datasets:
        for encoder in encoders:
            for tasc_ in tasc_approach:
                for mechanism in mechanisms:

                    path = os.path.join(cwd, results_dir, tasc_ + dataset + "/")
                    file_name = encoder + "_" + mechanism + "Attention_decision-flip-set-summary.csv"

                    results = pd.read_csv(path + file_name)

                    collect = {
                        "dataset" : dataset,
                        "encoder": encoder,
                        "method": tasc_mapper[tasc_],
                        "mechanism": mechanism,
                        "attention" : results[results["Unnamed: 0"] == "attention"]["mean percentage"].values[0],
                        "random" :  results[results["Unnamed: 0"] == "random"]["mean percentage"].values[0],
                        "gradOfatt" :  results[results["Unnamed: 0"] == "attention gradients"]["mean percentage"].values[0],
                        "attention*gradients" :  results[results["Unnamed: 0"] == "scaled attention"]["mean percentage"].values[0]
                    }

                    if tasc_ == "lin_" or tasc_ == "":

                        try:

                            collect["gradients"] = results[results["Unnamed: 0"] == "grad"]["mean percentage"].values[0]
                            collect["omission"] = results[results["Unnamed: 0"] == "omission"]["mean percentage"].values[0]
                            collect["ig"] = results[results["Unnamed: 0"] == "ig"]["mean percentage"].values[0]

                        except:

                            print("-- No results for dataset : {} | encoder {} | tasc_ {} | mechanism {}".format(
                                dataset,
                                encoder,
                                tasc_,
                                mechanism
                            ))

                    all_results.append(collect)
    
    return pd.DataFrame(all_results)


def capture_all_flip(results_dir = str, datasets = list(), encoders = list(), tasc_approach = list(), mechanisms = list()):

    tasc_mapper = {
        "" : "baseline",
        "lin_" : "lin",
        "feat_" : "feat",
        "conv_" : "conv"
    }


    cwd = os.getcwd()

    all_results = []


    for dataset in datasets:
        for encoder in encoders:
            for tasc_ in tasc_approach:
                for mechanism in mechanisms:

                    path = os.path.join(cwd, results_dir, tasc_ + dataset + "/")
                    file_name = encoder + "_" + mechanism + "Attention_decision-flip-single-summary.csv"

                    results = pd.read_csv(path + file_name)

                    if "True" not in results.columns:

                        results["True"] = 0.

                    collect = {
                                "dataset" : dataset,
                                "encoder": encoder,
                                "method": tasc_mapper[tasc_],
                                "mechanism": mechanism,
                                "attention" : results[results["Unnamed: 0"] == "attention"]["True"].values[0],
                                "random" :  results[results["Unnamed: 0"] == "random_source"]["True"].values[0],
                                "gradOfatt" :  results[results["Unnamed: 0"] == "attention gradients"]["True"].values[0],
                                "attention*gradients" :  results[results["Unnamed: 0"] == "scaled attention"]["True"].values[0]
                            }

                    all_results.append(collect)


    return pd.DataFrame(all_results)



def plot_radars(graph_loc = "decision-flip-set/", results = pd.DataFrame(), plot_by_what = str):


    results = results.groupby([plot_by_what, "method"]).mean()
    results.reset_index(inplace = True)
    results[results.method == "baseline"]["attention"].values


    data_mapper = {
        "agnews" : "AG",
        "twitter" : "ADR", 
        "sst" : "SST",
        "mimicanemia": "MIMIC",
        "imdb" : "IMDB"
    }

    path = graph_loc + plot_by_what + "/unfiltered/"

    os.makedirs(path , exist_ok=True)
    
    for method in results.columns[-3:]:


        categories = results[plot_by_what].unique()
        categories = [*categories, categories[0]]

        if plot_by_what == "dataset":

            categories = [data_mapper[x] for x in categories]

        else: categories = [x.upper() for x in categories]

        methods = {}

        base_a = results[results.method == "baseline"][method].values
        lina = results[results.method == "lin"][method].values
        feata = results[results.method == "feat"][method].values
        conva = results[results.method == "conv"][method].values

        base_a = [*base_a, base_a[0]]
        lina = [*lina, lina[0]]
        feata = [*feata, feata[0]]
        conva = [*conva, conva[0]]




        fig = go.Figure(
            data=[
                go.Scatterpolar(r=np.asarray(base_a), theta=categories, name='No-TaSc', line = {
                    "color" : "blue"}),#, "dash": "dash"})
                go.Scatterpolar(r=np.asarray(lina), theta=categories, name='Lin-TaSc', line = {
                    "color" : "green", "dash": "dash"}),
                go.Scatterpolar(r=np.asarray(feata), theta=categories, name='Feat-TaSc', line = {
                    "color" : "orange", "dash": "dash"}),
                go.Scatterpolar(r=np.asarray(conva), theta=categories, name='Conv-TaSc', line = {
                    "color" : "black", "dash": "dash"}),

            ],
            layout=go.Layout(
                title=go.layout.Title(text=method),
                polar={'radialaxis': {'visible': True}},
                showlegend=True,
                font = {"size": 20}
            )
        )

        plt.tight_layout()
        fig.write_image(path + method + "_per_encoder.pdf")
        
    return

if __name__ == "__main__":
    
    
    results_dir = "dev_test_exp/"
    encoders = ["bert", "lstm", "gru","mlp", "cnn"]
    datasets = ["mimicanemia", "imdb", "sst", "twitter", "agnews"]
    tasc_approach = ["", "lin_", "feat_", "conv_"]
    mechanisms = ["Tanh", "Dot"]

    os.getcwd()
    
    ### decision flip
    all_res = capture_all_flip(
        results_dir = results_dir,
        encoders = encoders,
        datasets = datasets, 
        tasc_approach = tasc_approach,
        mechanisms = mechanisms
    )
    
    ## plot per encoder 
    
    filt = all_res[["dataset", "encoder", "method", "mechanism", "attention" ,  "gradOfatt",  "attention*gradients"]]
    
    plot_radars(
        graph_loc = "decision-flip-single/", 
        results = filt, 
        plot_by_what = "encoder"
    )
    
    ## plot per dataset
    
    plot_radars(
        graph_loc = "decision-flip-single/", 
        results = filt, 
        plot_by_what = "dataset"
    )
    
    
    ### fraction of flip
    all_res = capture_all_fraction(
        results_dir = results_dir,
        encoders = encoders,
        datasets = datasets, 
        tasc_approach = tasc_approach,
        mechanisms = mechanisms
    )
    
    ## plot per encoder 
    
    filt = all_res[["dataset", "encoder", "method", "mechanism", "attention" ,  "gradOfatt",  "attention*gradients"]]
    
    plot_radars(
        graph_loc = "decision-flip-set/", 
        results = filt, 
        plot_by_what = "encoder"
    )
    
    ## plot per dataset
    
    plot_radars(
        graph_loc = "decision-flip-set/", 
        results = filt, 
        plot_by_what = "dataset"
    )