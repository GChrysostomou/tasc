#!/usr/bin/env python
# coding: utf-8



import glob
import pandas as pd
import operator
import os

def return_full_table(results_dir , datasets,  encoders , mechanisms, tasc_approach, format_for_tex = False, save = True):

    save_dir = "summarised_results/" + tasc_approach + "-tasc_comparisons_single/"

    try:
    
        os.makedirs(save_dir)

    except:
        
        pass


    #  ## Instructions
    # 
    # The following window generates the full stack of results for attention based explanation comparison (Tables in Appendix)
    # 
    # ```use_of_format``` parameter for use with latex.


    set_results = []

    header = ["dataset","encoder", 
            "attn (Tanh)", "attn(Dot)", "attn (Tanh +)", "attn(Dot +)",
            "attn_grad (Tanh)", "attn_grad(Dot)", 
            "attn_grad (Tanh +)", "attn_grad(Dot +)",
            "attn_gradattn (Tanh)", "attn_gradattn(Dot)", 
            "attn_gradattn (Tanh +)", "attn_gradattn(Dot +)",
            ]
    set_results.append(header)

    use_of_format = format_for_tex

    for dataset in datasets:

        for encoder in encoders:
            
            mech_temp = {}
            
            for mechanism in mechanisms:
            
                nontasc_files = glob.glob(results_dir  + dataset + "/"+encoder+ "*" + mechanism + "*decision-flip-single-summary.csv")

                tasc_files = glob.glob(results_dir + tasc_approach + "_" + dataset + "/"+encoder+ "*" + mechanism + "*decision-flip-single-summary.csv")
                
                nontasc = dict(pd.read_csv(nontasc_files[0]).drop(columns = "False").values)

                tasc = dict(pd.read_csv(tasc_files[0]).drop(columns = "False").values)

                mech_temp[mechanism] = {}
                

                mech_temp[mechanism]["attn"] = round(nontasc["max_source"],1)
                mech_temp[mechanism]["attn_grad"] = round(nontasc["att_grad"],1)
                mech_temp[mechanism]["attn_gradattn"] = round(nontasc["att*grad"],1)


                mech_temp[mechanism]["attn +"] = round(tasc["max_source"],1)
                mech_temp[mechanism]["attn_grad +"] = round(tasc["att_grad"],1)
                mech_temp[mechanism]["attn_gradattn +"] = round(tasc["att*grad"],1)
                
                
                
                if use_of_format:
                
                    max_key = max(mech_temp[mechanism].items(), key=operator.itemgetter(1))[0]

                    mech_temp[mechanism][max_key] = r"\textbf{" + str(mech_temp[mechanism][max_key])                + r"}"

                    if tasc["max_source"] > nontasc["max_source"]:

                        try:

                            mech_temp[mechanism]["attn +"] = r"\underline{" + str(mech_temp[mechanism]["attn +"])                         + r"}"
                        except:
                            mech_temp[mechanism]["attn +"] = r"\underline{" + mech_temp[mechanism]["attn +"]                         + r"}"

                    if tasc["att_grad"] > nontasc["att_grad"]:
                        try:

                            mech_temp[mechanism]["attn_grad +"] = r"\underline{" + str(mech_temp[mechanism]["attn_grad +"])                          + r"}"

                        except:

                            mech_temp[mechanism]["attn_grad +"] = r"\underline{" + mech_temp[mechanism]["attn_grad +"]                          + r"}"

                    if tasc["att*grad"] > nontasc["att*grad"]:

                        try:
                            mech_temp[mechanism]["attn_gradattn +"] = r"\underline{" + str(mech_temp[mechanism]["attn_gradattn +"])                         + r"}"
                        except:
                            mech_temp[mechanism]["attn_gradattn +"] = r"\underline{" + mech_temp[mechanism]["attn_gradattn +"]                         + r"}"

                
            set_results.append([dataset, encoder, 
                            mech_temp["Tanh"]["attn"], mech_temp["Dot"]["attn"],
                                mech_temp["Tanh"]["attn +"], mech_temp["Dot"]["attn +"],
                                mech_temp["Tanh"]["attn_grad"], mech_temp["Dot"]["attn_grad"],
                                mech_temp["Tanh"]["attn_grad +"], mech_temp["Dot"]["attn_grad +"],
                                mech_temp["Tanh"]["attn_gradattn"], mech_temp["Dot"]["attn_gradattn"],
                                mech_temp["Tanh"]["attn_gradattn +"], mech_temp["Dot"]["attn_gradattn +"],
                            ])


    set_of_w = pd.DataFrame(set_results)

    pd.options.display.float_format = lambda x : '{:.0f}'.format(x) if int(x) == x else '{:,.2f}'.format(x)

    new_header = set_of_w.iloc[0] #grab the first row for the header
    set_of_w = set_of_w[1:] #take the data less the header row
    set_of_w.columns = new_header

    if save:

        if use_of_format:

            set_of_w.to_latex(save_dir + "full-table.tex", index = False, escape=False)
        
        set_of_w.to_csv(save_dir + "full-table.csv", index = False)

    return set_of_w, save_dir


def r_imp(x1, x2):
    
    """
    format for relative improvement
    """

    new_x2 = str(round(x2,2)) + " (" + str(round(x2/x1,1)) + ")"
    
    return new_x2

def produce_per_mechanism(results_dir , datasets,  encoders , mechanisms, tasc_approach):

    set_of_w, save_dir  = return_full_table(results_dir =results_dir, 
                                            datasets = datasets,  
                                            encoders = encoders, 
                                            mechanisms = mechanisms,
                                            tasc_approach = tasc_approach, 
                                            format_for_tex = False, save = False)


    nontasc_columns = [x for x in set_of_w.columns if "+" not in x]
    tasc_columns = [x for x in set_of_w.columns if "+" in x]


    per_mechanism = {}

        
    per_mechanism["no-tasc"] = dict(set_of_w[nontasc_columns].drop(columns = ["dataset", "encoder"]).mean().items())

    no_plus = {}
    for key, item in dict(set_of_w[tasc_columns].mean().items()).items():

        new_key = key.split(" +")[0] + ")"
        
        no_plus[new_key] = item

    per_mechanism[tasc_approach + "-tasc"] = no_plus

    per_mechanism = pd.DataFrame(per_mechanism)[["no-tasc", tasc_approach + "-tasc"]]


    per_mechanism[tasc_approach + "-tasc"] = per_mechanism.apply(lambda x : r_imp(x["no-tasc"], x[tasc_approach + "-tasc"]), axis = 1)

    per_mechanism.to_latex(save_dir + "across-mechanisms.tex", escape=False)
    per_mechanism.to_latex(save_dir + "across-mechanisms.csv")


def produce_per_encoder(results_dir , datasets,  encoders , mechanisms, tasc_approach):

    set_of_w, save_dir  = return_full_table(results_dir =results_dir, 
                                            datasets = datasets,  
                                            encoders = encoders, 
                                            mechanisms = mechanisms,
                                            tasc_approach = tasc_approach, 
                                            format_for_tex = False, save = False)

    asfloat = set_of_w.columns.drop("dataset").drop("encoder")
    set_of_w[asfloat] = set_of_w[asfloat].astype(float)

    per_encoder = set_of_w[set_of_w.columns.drop("dataset")]

    avgs = per_encoder.groupby("encoder").mean()

    nontasc_columns = [x for x in set_of_w.columns if "+" not in x][2:]
    tasc_columns = [x for x in set_of_w.columns if "+" in x]

    nontasc_dot = avgs[[x for x in nontasc_columns if "Tanh" not in x]] 
    import re
    renamed_cols = dict(zip(nontasc_dot.columns, [re.sub("Dot", "", x) for x in nontasc_dot.columns]))
    nontasc_dot = nontasc_dot.rename(columns = renamed_cols)
    nontasc_tanh = avgs[[x for x in nontasc_columns if "Dot" not in x]]
    renamed_cols = dict(zip(nontasc_tanh.columns, [re.sub(" ", "", re.sub("Tanh", "", x)) for x in nontasc_tanh.columns]))
    nontasc_tanh = nontasc_tanh.rename(columns = renamed_cols)


    nontasc_tanh = nontasc_tanh.to_dict()
    nontasc_dot = nontasc_dot.to_dict()


    nontasc_avg = {}

    for key in nontasc_tanh.keys():
        
        nontasc_avg[key] = {}
        
        for enc in nontasc_tanh[key].keys():
            
            nontasc_avg[key][enc] = (nontasc_tanh[key][enc] + nontasc_dot[key][enc]) / 2



    tasc_dot = avgs[[x for x in tasc_columns if "Tanh" not in x]] 
    renamed_cols = dict(zip(tasc_dot.columns, [re.sub("Dot ", "", x) for x in tasc_dot.columns]))
    tasc_dot = tasc_dot.rename(columns = renamed_cols)
    tasc_tanh = avgs[[x for x in tasc_columns if "Dot" not in x]]
    renamed_cols = dict(zip(tasc_tanh.columns, [re.sub(" ", "", re.sub("Tanh", "", x)) for x in tasc_tanh.columns]))
    tasc_tanh = tasc_tanh.rename(columns = renamed_cols)

    tasc_tanh = tasc_tanh.to_dict()
    tasc_dot = tasc_dot.to_dict()

    tasc_avg = {}

    for key in tasc_tanh.keys():
        
        tasc_avg[key] = {}
        
        for enc in tasc_tanh[key].keys():
            
            tasc_avg[key][enc] = (tasc_tanh[key][enc] + tasc_dot[key][enc]) / 2


    tasc_avg = pd.DataFrame(tasc_avg)
    nontasc_avg = pd.DataFrame(nontasc_avg)

    tasc_avg["encoder"] = tasc_avg.index
    nontasc_avg["encoder"] = nontasc_avg.index

    merged = nontasc_avg.merge(tasc_avg, on = "encoder")

    merged["attn(+)"] = merged.apply(lambda x : r_imp(x["attn()"], x["attn(+)"]), axis = 1)
    merged["attn_grad(+)"] = merged.apply(lambda x : r_imp(x["attn_grad()"], x["attn_grad(+)"]), axis = 1)
    merged["attn_gradattn(+)"] = merged.apply(lambda x : r_imp(x["attn_gradattn()"], x["attn_gradattn(+)"]), axis = 1)

    tasc = merged.drop(columns = ["attn()", "attn_grad()", "attn_gradattn()"])

    new_header = tasc.T.iloc[0] #grab the first row for the header
    tasc = tasc.T[1:] #take the data less the header row
    tasc.columns = new_header
    tasc = tasc[encoders]

    nontasc_avg = nontasc_avg.drop(columns = "encoder")
    nontasc_avg = nontasc_avg.T[encoders]
    

    tasc.T.to_latex(save_dir + "tasc-across-encoders.tex", escape=False)
    nontasc_avg.T.to_latex(save_dir + "non-tasc-across-encoders.tex", escape=False)


def produce_per_dataset(results_dir , datasets,  encoders , mechanisms, tasc_approach):

    set_of_w, save_dir  = return_full_table(results_dir =results_dir, 
                                            datasets = datasets,  
                                            encoders = encoders, 
                                            mechanisms = mechanisms,
                                            tasc_approach = tasc_approach, 
                                            format_for_tex = False, save = False)

    asfloat = set_of_w.columns.drop("dataset").drop("encoder")
    set_of_w[asfloat] = set_of_w[asfloat].astype(float)

    per_dataset = set_of_w[set_of_w.columns.drop("encoder")]

    avgs = per_dataset.groupby("dataset").mean()

    nontasc_columns = [x for x in set_of_w.columns if "+" not in x][2:]
    tasc_columns = [x for x in set_of_w.columns if "+" in x]

    nontasc_dot = avgs[[x for x in nontasc_columns if "Tanh" not in x]] 
    import re
    renamed_cols = dict(zip(nontasc_dot.columns, [re.sub("Dot", "", x) for x in nontasc_dot.columns]))
    nontasc_dot = nontasc_dot.rename(columns = renamed_cols)
    nontasc_tanh = avgs[[x for x in nontasc_columns if "Dot" not in x]]
    renamed_cols = dict(zip(nontasc_tanh.columns, [re.sub(" ", "", re.sub("Tanh", "", x)) for x in nontasc_tanh.columns]))
    nontasc_tanh = nontasc_tanh.rename(columns = renamed_cols)

    nontasc_tanh = nontasc_tanh.to_dict()
    nontasc_dot = nontasc_dot.to_dict()

    nontasc_avg = {}

    for key in nontasc_tanh.keys():
        
        nontasc_avg[key] = {}
        
        for dat in nontasc_tanh[key].keys():
            
            nontasc_avg[key][dat] = (nontasc_tanh[key][dat] + nontasc_dot[key][dat]) / 2
            

    tasc_dot = avgs[[x for x in tasc_columns if "Tanh" not in x]] 
    renamed_cols = dict(zip(tasc_dot.columns, [re.sub("Dot ", "", x) for x in tasc_dot.columns]))
    tasc_dot = tasc_dot.rename(columns = renamed_cols)
    tasc_tanh = avgs[[x for x in tasc_columns if "Dot" not in x]]
    renamed_cols = dict(zip(tasc_tanh.columns, [re.sub(" ", "", re.sub("Tanh", "", x)) for x in tasc_tanh.columns]))
    tasc_tanh = tasc_tanh.rename(columns = renamed_cols)

    tasc_tanh = tasc_tanh.to_dict()
    tasc_dot = tasc_dot.to_dict()

    tasc_avg = {}

    for key in tasc_tanh.keys():
        
        tasc_avg[key] = {}
        
        for dat in tasc_tanh[key].keys():
            
            tasc_avg[key][dat] = (tasc_tanh[key][dat] + tasc_dot[key][dat]) / 2



    tasc_avg = pd.DataFrame(tasc_avg)
    nontasc_avg = pd.DataFrame(nontasc_avg)

    tasc_avg["dataset"] = tasc_avg.index
    nontasc_avg["dataset"] = nontasc_avg.index

    merged = nontasc_avg.merge(tasc_avg, on = "dataset")


    merged["attn(+)"] = merged.apply(lambda x : r_imp(x["attn()"], x["attn(+)"]), axis = 1)
    merged["attn_grad(+)"] = merged.apply(lambda x : r_imp(x["attn_grad()"], x["attn_grad(+)"]), axis = 1)
    merged["attn_gradattn(+)"] = merged.apply(lambda x : r_imp(x["attn_gradattn()"], x["attn_gradattn(+)"]), axis = 1)

    tasc = merged.drop(columns = ["attn()", "attn_grad()", "attn_gradattn()"])

    new_header = tasc.T.iloc[0] #grab the first row for the header
    tasc = tasc.T[1:] #take the data less the header row
    tasc.columns = new_header
    tasc = tasc[datasets]

    nontasc_avg = nontasc_avg.drop(columns = "dataset")
    nontasc_avg = nontasc_avg.T[datasets]
    
    tasc.T.to_latex(save_dir + "tasc-across-datasets.tex", escape=False)
    nontasc_avg.T.to_latex(save_dir + "non-tasc-across-datasets.tex", escape=False)

def decision_single_experiments(results_dir , datasets,  encoders , mechanisms, tasc_approach):

    return_full_table(results_dir = results_dir, datasets = datasets, 
            encoders = encoders, mechanisms = mechanisms, tasc_approach = tasc_approach, format_for_tex=True)

    produce_per_mechanism(results_dir = results_dir, datasets = datasets, 
            encoders = encoders, mechanisms = mechanisms, tasc_approach = tasc_approach)

    produce_per_encoder(results_dir = results_dir, datasets = datasets, 
            encoders = encoders, mechanisms = mechanisms, tasc_approach = tasc_approach)


    produce_per_dataset(results_dir = results_dir, datasets = datasets, 
            encoders = encoders, mechanisms = mechanisms, tasc_approach = tasc_approach)