#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import matplotlib
import os


    
def plot(results_directory, dataset, encoder, mechanism, tasc_approach):

    save_dir = "summarised_results/" + tasc_approach + "-tasc_boxplots/"

    try:
    
        os.makedirs(save_dir)

    except:
        
        pass

    datasets = glob.glob(results_directory + dataset + "/"+ encoder + "*" +mechanism +"*decision-flip-set.csv")
    df = pd.read_csv(datasets[0])
    
    df = df[["attention", "attention gradients", "scaled attention"]]
   
    datasets = glob.glob(results_directory + "/" + tasc_approach + "_" + dataset + "/"+ encoder + "*" +mechanism +"*decision-flip-set.csv")
    
    df2 = pd.read_csv(datasets[0])[["attention", "attention gradients", "scaled attention"]]

    matplotlib.rc('xtick', labelsize=40)     
    matplotlib.rc('ytick', labelsize=30)


    df["extension"] = mechanism
    df2["extension"] = mechanism + "+"
   
    df = df.replace(np.nan, 0)
    df2 = df2.replace(np.nan, 0)

    both = pd.concat([df, df2], 0)

    mapper = {"attention":"α","attention gradients":"∇α", "scaled attention":"α∇α"}
    
    
    both = both.rename(mapper, axis= 1)

    df_long = pd.melt(both, "extension", var_name="a", value_name="c")
    df_long.extension = df_long.extension.apply(lambda x: x + " (No-TaSc)" if "+" not in x else x.rstrip("+") + " (Lin-TaSc)")


    plt.figure(figsize = (18,10))


    box_plot = sns.boxplot(x="a", y="c", data = df_long, hue = "extension")



    plt.xlabel("Importance Metrics", fontsize=35)
    plt.ylabel("Fraction of tokens removed",  fontsize=35)

    plt.legend(prop={'size': 25}, loc = 1 )
    
    plt.savefig(save_dir + dataset + "-" + encoder + "-" + mechanism + "-" + "dec_flip_set.png",
               bbox_inches = "tight", dpi = 100)

    plt.close()

def produce_boxplots(results_dir, datasets, encoders, mechanisms, tasc_approach):
    
    for dataset in datasets:
        
        for encoder in encoders:
            
            for mechanism in mechanisms:
                
                try:
                    plot(results_dir, dataset, encoder, mechanism, tasc_approach)
                    
                except:
                    
                    print(dataset + "-" + encoder + "--" + mechanism)

