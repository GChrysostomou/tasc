#!/usr/bin/env python
# coding: utf-8
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-datasets", nargs='+',help = "select dataset / task", default = ["sst", "twitter", "agnews", "imdb", "mimicanemia"])
parser.add_argument("-encoders", nargs='+', help = "select encoder", default = ["bert", "lstm", "gru", "mlp", "cnn"])
parser.add_argument("-experiments_dir", type = str, help = "where to load results from", default = "test_experiment_results/")
parser.add_argument("-mechanisms", type = str, help = "choose mechanism", default = ["Tanh", "Dot"], choices = ["Tanh", "Dot"] )
parser.add_argument("-tasc_ver", type = str, help = "choose tasc mechanism", default = "lin", choices = ["lin", "feat", "conv"] )


print("\n", vars(parser.parse_args()), "\n")

args = vars(parser.parse_args())


from modules.report_fun.comparing_explanations import comparison
from modules.report_fun.boxplots import produce_boxplots
from modules.report_fun.decision_set_experiments import decision_set_experiments
from modules.report_fun.decision_single_experiments import decision_single_experiments

comparison(results_dir = args["experiments_dir"], datasets = args["datasets"], 
             encoders = args["encoders"], mechanisms = args["mechanisms"], tasc_approach = args["tasc_ver"])


produce_boxplots(results_dir = args["experiments_dir"], datasets = args["datasets"], 
             encoders = args["encoders"], mechanisms = args["mechanisms"], tasc_approach = args["tasc_ver"])

decision_set_experiments(results_dir = args["experiments_dir"], datasets = args["datasets"], 
            encoders = args["encoders"], mechanisms = args["mechanisms"], tasc_approach = args["tasc_ver"])

decision_single_experiments(results_dir = args["experiments_dir"], datasets = args["datasets"], 
            encoders = args["encoders"], mechanisms = args["mechanisms"], tasc_approach = args["tasc_ver"])
