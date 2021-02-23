import numpy as np
import matplotlib.pyplot as plt
from time import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import get_scorer

import tensorflow as tf
from tensorflow import keras

class RandomSeedSearchCV:
    '''
    Generalized random-search model tuner! Uses random seeds and user-made model-maker function to search
    through acceptable models and find those that achieve the best cross_validation score.

    This class is just a shell to execute the search. It doesn't know anything about the models
    or the parameter space to search through; all of that is encapsulated in the user-provided model-making
    function.
    '''
    def __init__(self,model_maker,n_iter=50,validation=0.1,cv=5,scoring='mean_squared_error',
            metric=None,metric_needs_weights=False,plot_summary=True,shield_seed=True,verbose=True,
            random_state=None,custom_fit=None,fit_params=None,**model_maker_kwargs):
        '''
        INPUTS:
        model_maker          -  Function that takes integer and returns model. Returned model must implement
                                 fit and predict methods, like standard sklearn estimators do.

                                This should be a function defined by the user ahead of time. The function should
                                 accept a random seed and then internally construct a model with randomly drawn
                                 hyperparameters. The idea behind this is to take responsibility of knowing the parameter
                                 space and generating models within it off of the searcher and instead placing that
                                 responsibility on the user-provided model-maker. randomseed_searchCV will never know
                                 what type of model you're fitting or what range of which parameters you're sampling.
                                 All it will know is that it can give model_maker an integer, and it will
                                 return a model that can be fit and used to predict.

        **model_maker_kwargs - Any additional arguments to be passed to the model maker.

        X_train,y_train      - Training data and labels.
        validation           - Fraction to split off for validation testing. Default is 0.1. If None,False,0,or 1
                                are provided, validation testing will not be done.
        cv                   - Integer N for N-fold cross validation.
        scoring              - Metric to use when ranking models.
        plot_summary         - Boolean whether or not to produce a beautiful summary plot!
        shield_seed          - Boolean whether or not to forcibly prevent the model maker from globally changing
                                the numpy random seed.
        verbose              - Boolean whether or not to show progress.
        random_state         - Random state to use for splitting off validation dataset.
        '''

        self.model_maker = model_maker
        self.n_iter = n_iter
        self.validation = validation
        self.cv = cv
        self.scoring = scoring

        self.plot_summary = plot_summary
        self.verbose = verbose

        self.shield_seed = shield_seed
        self.random_state = random_state

        self.custom_fit = custom_fit

        self.fit_params = fit_params
        self.model_maker_kwargs = model_maker_kwargs

    def fit(self,X,y):
        if self.verbose: progress = simple_progress()

        if self.shield_seed:
            modmkr = preserve_state(self.model_maker)
        else:
            modmkr = self.model_maker

        use_custom_fit = not self.custom_fit is None

        if self.fit_params is None:
            self.fit_params = {}

        #Get metric callable from sklearn. L8r I'm gonna make it
        # so the user can provide a callable metric themselves, because I
        # don't like that sklearn uses negative mse instead of positive. It drives me bonkers.
        metric = get_metric(self.scoring)

        #Split validation set out of the training set, if
        if self.validation is None or self.validation==False or self.validation>=1 or self.validation <=0:
            use_val = False
            Xtra,ytra = X,y
            Xval,yval = None,None
        else:
            use_val = True
            Xtra,Xval,ytra,yval = train_test_split(X,y,
                                                test_size=self.validation,
                                                random_state=self.random_state)

        #Draw random seeds to use for model generation.
        seeds = np.random.choice(10*self.n_iter,self.n_iter,replace=False)

        #Create empty lists to store model performance measures.
        if self.cv > 1 :cv_scores = np.array([])
        train_metric = np.array([])
        valid_metric = np.array([])
        times = np.array([])

        #For each model, find cv_score and metrics. Also store training time.
        for i,seed in enumerate(seeds):
            if self.verbose: progress.update("%d/%d: Seed = %d"%(i+1,self.n_iter,seed))
            model = modmkr(seed,**self.model_maker_kwargs)
            if self.cv > 1:
                cv_score = np.mean(cross_val_score(model,Xtra,ytra,scoring=metric,cv=self.cv,fit_params=self.fit_params))
                cv_scores = np.append(cv_scores,cv_score)

            start_time = time()
            if use_custom_fit:
                model = custom_fit(model,Xtra,ytra,**self.fit_params)
            else:
                model.fit(Xtra,ytra,**self.fit_params)
            train_metric = np.append(train_metric, metric(model,Xtra,ytra))
            if use_val:
                valid_metric = np.append(valid_metric, metric(model,Xval,yval))
            times = np.append(times, time()-start_time)

        #Plot a summary when done.
        if self.plot_summary:
            fig,ax = plt.subplots()
            scat = ax.scatter(train_metric,valid_metric,c=times,cmap='coolwarm',s=50)
            cbar = fig.colorbar(scat,ax=ax)
            ax.set_xlabel("Training Metric")
            ax.set_ylabel("Validation Metric")
            cbar.ax.set_ylabel("Training Time (s)")
            mi,ma = 0,1.1*np.max([np.max(train_metric),np.max(valid_metric)])
            ax.set_xlim(mi,ma)
            ax.set_ylim(mi,ma)
            ax.plot([mi,ma],[mi,ma],ls='--',color='black')

        if self.cv > 1:
            sort = np.argsort(cv_scores)
            return np.c_[seeds[sort],cv_scores[sort],train_metric[sort],valid_metric[sort],times[sort]]
        else:
            sort = np.argsort(valid_metric)
            return np.c_[seeds[sort],train_metric[sort],valid_metric[sort],times[sort]]
