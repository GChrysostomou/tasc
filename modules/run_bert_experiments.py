#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import modules.experiments_bc_bert.decision_single as eoo
import modules.experiments_bc_bert.decision_set as perc
import modules.experiments_bc_bert.set_tp as dmp

class evaluate:
    
    def __init__(self, classifier, loss_function, data, save_path, bayesian = None):
        
        self.classifier = classifier
        self.loss_function = loss_function
        self.dataset_name, dataset = data
        self.testing = dataset.testing
        self.save_path = save_path
        self.bayesian = bayesian
        
        self.encode_sel = classifier.encoder.encode_sel
        self.mechanism_name = classifier.attention.__class__.__name__
       
        self.save_dir = self.save_path + \
        self.encode_sel + "_" + self.mechanism_name


    def decision_flip_single(self):
        
        """
        Removing the most informative token
        """
        
        eoo.effect_on_output(self.testing, 
                   self.classifier, 
                   save_path = self.save_dir)
        
        
    def decision_flip_set(self):
        
        """
        Recording the fraction of tokens required to cause a decision flip
        """
        
        perc.percentage_removed(self.testing, 
                   self.classifier, 
                   save_path = self.save_dir)
        

    def correct_classified_set(self, data_size, largest = True):
        """
        Conducts a decision flip experiment
        on correctly classified instances
        """
        dmp.degrading_model_perf(data = self.testing, 
            model = self.classifier, 
            save_path  = self.save_dir,
            data_size = data_size,
            largest = largest)
