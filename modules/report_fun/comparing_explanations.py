#!/usr/bin/env python
# coding: utf-8


import glob
import pandas as pd
import os

# datasets = ["agnews"]#["sst","twitter", "imdb", "agnews", "mimicanemia"]
# encoders = ["bert", "lstm", "gru", "mlp", "cnn"]
# mechanisms = ["Tanh", "Dot"]
# tasc_approach = "lin"


def comparison(results_dir, datasets, encoders, mechanisms, tasc_approach):

    save_dir = "summarised_results/" + tasc_approach + "-tasc_comparing_explanations/"

    try:
    
        os.makedirs(save_dir)

    except:
    
        pass


    set_results = []

    header = ["dataset","encoder", 
            "omission (Tanh)", "grad (Tanh)", "IG (Tanh)", "Attn_Grad*Attn (Tanh +)",
            "omission (Dot)", "grad (Dot)", "IG (Dot)", "Attn_Grad*Attn (Dot +)"
            ]
    set_results.append(header)

    for dataset in datasets:

        for encoder in encoders:
            
            mech_temp = {}
            
            for mechanism in mechanisms:

                nontasc_files = glob.glob(results_dir  + dataset + "/"+encoder+ "*" + mechanism + "*decision-flip-set-summary.csv")

                tasc_files = glob.glob(results_dir + tasc_approach + "_" + dataset + "/"+encoder+ "*" + mechanism + "*decision-flip-set-summary.csv")

                nontasc = dict(pd.read_csv(nontasc_files[0]).values)

                tasc = dict(pd.read_csv(tasc_files[0]).values)

                mech_temp[mechanism] = {}

                mech_temp[mechanism]["Attn*Attn_Grad +"] = round(tasc["att*grad"],2)

                mech_temp[mechanism]["omission"] = round(nontasc["omission"],2)

                mech_temp[mechanism]["grad"] = round(nontasc["grad"],2)

                mech_temp[mechanism]["IG"] = round(nontasc["IG"],2)



            set_results.append([dataset, encoder, 
                                mech_temp["Tanh"]["omission"],  mech_temp["Tanh"]["grad"],  mech_temp["Tanh"]["IG"], mech_temp["Tanh"]["Attn*Attn_Grad +"],
                                mech_temp["Dot"]["omission"],  mech_temp["Dot"]["grad"],  mech_temp["Dot"]["IG"], mech_temp["Dot"]["Attn*Attn_Grad +"],

                            ])


    set_of_w = pd.DataFrame(set_results)

    pd.options.display.float_format = lambda x : '{:.0f}'.format(x) if int(x) == x else '{:,.2f}'.format(x)



    new_header = set_of_w.iloc[0] #grab the first row for the header
    set_of_w = set_of_w[1:] #take the data less the header row
    set_of_w.columns = new_header


    set_of_w.to_latex(save_dir + "explanation-comparison.tex", index = False, escape=False)
    set_of_w.to_csv(save_dir + "explanation-comparison.csv", index = False)

