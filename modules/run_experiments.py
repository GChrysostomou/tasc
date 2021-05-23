import modules.experiments_bc.decision_single as eoo
import modules.experiments_bc.decision_set as perc
import modules.experiments_bc.set_tp as dmp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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


    def topbotk_s(self, w2ix, k = 10):
        """
        returns scores for explanations of topkwords
        """
        
        assert self.classifier.tasc_attention == True
        
        explanations = dict(enumerate((self.classifier.explanations.unsqueeze(-1) * self.classifier.encoder.embedding.weight).sum(-1) ))
 
        ix2w = {v:k for k,v in w2ix.items()}
        
        word_scores = tuple({ix2w[k]:v.item() for k,v in explanations.items()}.items())
        
        top_words = list(sorted(word_scores, key = lambda x: x[1], reverse = True))[:k]
        bottom_words = list(sorted(word_scores, key = lambda x: x[1], reverse = False))[:k]
        bottom_words = list(sorted(bottom_words, key = lambda x: x[1], reverse = True))
        
        total = top_words + bottom_words
        
        plt.clf()
        df = pd.DataFrame(total)
        df.to_csv(self.save_dir + "_topbottomk.csv")
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(1, 1, 1)
        
        
        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        
        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        sns.barplot(data= df, x = 1, y = 0)
        plt.xlabel("scores (u)")
        plt.ylabel("")
        
        plt.savefig(self.save_dir + "_topbottomk_s.png")
        
    def topbotk_u(self, w2ix, k = 10):
        """
        returns scores for explanations of topkwords
        """
        
        assert self.classifier.tasc_attention == True
        
        explanations = dict(enumerate((self.classifier.explanations)))
 
        ix2w = {v:k for k,v in w2ix.items()}
        
        word_scores = tuple({ix2w[k]:v.item() for k,v in explanations.items()}.items())
        
        top_words = list(sorted(word_scores, key = lambda x: x[1], reverse = True))[:k]
        bottom_words = list(sorted(word_scores, key = lambda x: x[1], reverse = False))[:k]
        bottom_words = list(sorted(bottom_words, key = lambda x: x[1], reverse = True))
        
        total = top_words + bottom_words
        
        plt.clf()
        df = pd.DataFrame(total)
        df.to_csv(self.save_dir + "_topbottomk.csv")
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(1, 1, 1)
        
        
        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        
        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        sns.barplot(data= df, x = 1, y = 0)
        plt.xlabel("scores (u)")
        plt.ylabel("")
        
        plt.savefig(self.save_dir + "_topbottomk_u.png")
