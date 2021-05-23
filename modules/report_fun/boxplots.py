#!/usr/bin/env python
# coding: utf-8

# In[5]:


# import pandas
import pandas as pd
# import matplotlib
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns

import glob

import os


    
def plot(results_directory, dataset, encoder, mechanism, tasc_approach):

    save_dir = "summarised_results/" + tasc_approach + "-tasc_boxplots/"

    try:
    
        os.makedirs(save_dir)

    except:
        
        pass

    datasets = glob.glob(results_directory + dataset + "/"+ encoder + "*" +mechanism +"*decision-flip-set.csv")
    df = pd.read_csv(datasets[0])
    
    df = df[["max_source","att_grad", "att*grad"]]
   
    datasets = glob.glob(results_directory + "/" + tasc_approach + "_" + dataset + "/"+ encoder + "*" +mechanism +"*decision-flip-set.csv")
    
    df2 = pd.read_csv(datasets[0])[["max_source", "att_grad", "att*grad"]]

    import matplotlib

    matplotlib.rc('xtick', labelsize=40)     
    matplotlib.rc('ytick', labelsize=30)


    df["extension"] = mechanism
    df2["extension"] = mechanism + "+"
    import numpy as np
    df = df.replace(np.nan, 0)
    df2 = df2.replace(np.nan, 0)

    both = pd.concat([df, df2], 0)

    mapper = {"max_source":"α","att_grad":"∇α", "att*grad":"α∇α"}
    
    
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

