import pickle
import talos as ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras.initializers as ki

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from keras.callbacks import EarlyStopping
from vnet_model import optimiseCNN
from custom_loss import dice_coef_loss, surface_loss
from talos import Deploy, Evaluate, Analyze

def sweep_hyperparams(train_generator, validation_generator, 
                      target_size=(128,128,128), batchsize=6):
    p = {'ds': [4],
         'dropout': [0.0],
         'epochs': [50],
         'lr': [1e-5, 1e-6, 1e-8],
         'loss': [dice_coef_loss],
         'activation': ['prelu'],
         'decay': [1e-7, 1e-8, 1e-9],
         'k_initializer': ['lecun_normal','he_uniform'],
         'f_layer': ['softmax']}
    
    model_checkpoint = [EarlyStopping(monitor='val_loss', patience=10,
                                     mode='min', min_delta=1e-4)]
    
    model = optimiseCNN(train_generator, validation_generator, 
                        targetsize=target_size, batch_size=batchsize,
                        callbacks=model_checkpoint)
    
    dummyx, dummyy = train_generator.__getitem__(0)
    testx, testy = validation_generator.__getitem__(0)
    scan_object = ta.Scan(x = dummyx,
                              y = dummyy,
                              params = p,
                              model = model,
                              x_val = testx,
                              y_val = testy,
                              experiment_name = 'rsom_roi_hyperparam',
                              fraction_limit = 0.4,
                              random_method = 'quantum',
                              reduction_method='correlation',
                              reduction_interval=20,
                              reduction_window=40,
                              reduction_threshold=0.2,
                              reduction_metric='val_loss',
                              minimize_loss=True,
                              print_params = True)
    scan_model = Deploy(scan_object, 'rsom_roi_hyperparam', metric='val_dice_coef', asc=True)
    # print('Best model ... ', best_model(scan_object, 'val_dice_coef', False))
    # # activate_model(scan_object, best_model(scan_object, 'val_acc', False)).predict(X)
    
    
    # # p = ta.Predict(scan_object)
    # # evaluation = ta.Evaluate(scan_model).evaluate(testx, testy, metric='')
    
    
    # save_object(scan_object, 'example.pickle')
    # # tt = load_object('example.pickle')
    hyperparam_stats(p, scan_object)
    # # e = ta.Evaluate(scan_object)   

def hyperparam_stats(p, scan_object):
    
    df = scan_object.data
    
    # index = 0
    # for key in p:
    #     # print(index, key)
    #     print(df[f'{key}'])
    #     if index <= 3:
    #         plt.figure()
    #         # print(key)
    
    #         # print(df.val_loss)
    #         # print(df[f'{key}'])
    #         ax = sns.boxplot(x=key, y=df.val_loss, data=df[f'{key}'])
    #         ax.set_title(f'Log-loss improvement as a function of {key}')
    #         ax.set_yscale('log')
    #     index += 1
   
    r = ta.Analyze(scan_object)
    
    # rr = [data for data in r.data.rows if str(r.data) != 'nan']
    
    
    exclude = ['val_recall','val_precision','round_epochs','loss','binary_accuracy',
               None]
    newcols = _cols(r, metric=[], exclude=exclude)

    bp = best_params(r, 'val_binary_accuracy', exclude=[], n=1, ascending=False)
    
    r.plot_corr(metric=newcols, exclude=exclude, color_grades=5)
    r.plot_box('val_binary_accuracy', ['lr'], hue=None)
    # r.plot_hist('val_binary_accuracy', bins = 10)
    
    # r.plot_line(['val_binary_accuracy','epochs'])
    
    r.plot_regs(r.data['val_binary_accuracy'], r.data['epochs'])
    # r.plot_regs(r.data['val_binary_accuracy'], r.data['lr'])
    # r.plot_box(r.data['ds'], r.data['dropout'])
    # X = df[['ds', 'dropout', 'loss', 'activation', 'epochs', 'lr']]
    # scaler = MinMaxScaler()
    # y = scaler.fit_transform(np.reshape(df.val_loss, 1,-1))
    
    # reg = RandomForestRegressor(max_depth=3,n_estimators=100)
    # reg.fit(X,y)
    # pd.Series(reg.feature_importances_, index=X.columns)
    # df.sort_values(ascending=True).plot.barh(color='grey',
    #                                       title='Feature Importance of Hyperparameters')

def project_object(obj,*attributes):
	out={}
	for a in attributes:
		out[a]=getattr(obj,a)
	return out
    
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, protocol=2)

def load_object(filename):
        with open(filename, 'rb') as f:
                return pickle.load(f)

def print_hyperparameter_search_stats(t):
        print(" *** params: ",{ p:(v if len(v)<200 else [v[0],v[1],v[2],'...',v[-1]]) for p,v in t['params'].items()})
        print()
        print(" *** data ",type(t['data']),len(t['data']))
        print(t['data'].sort_values('val_acc',ascending=False).to_string())
        print()
        distinct_data=t['data']
        nunique = distinct_data.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        distinct_data = distinct_data.drop(cols_to_drop, axis=1)
        print(nunique,cols_to_drop)
        print(" *** distinct data ",type(distinct_data),len(distinct_data))
        print(distinct_data.sort_values('val_acc',ascending=False).to_string())
        print()
        print(" *** details ",type(t['details']),len(t['details']))
        print(t['details'])
        print()
        
def best_params(self, metric, exclude, n=10, ascending=False):

        '''Get the best parameters of the experiment based on a metric.
        Returns a numpy array with the values in a format that can be used
        with the talos backend in Scan(). Adds an index as the last column.
        metric | str or list | Column labels for the metric to correlate with
        exclude | list | Column label/s to be excluded from the correlation
        n | int | Number of hyperparameter permutations to be returned
        ascending | bool | Set to True when `metric` is to be minimized eg. loss
        '''

        cols = self._cols(metric, exclude)
        out = self.data[cols].sort_values(metric, ascending=ascending)
        out = out.drop(metric, axis=1).head(n)
        out.insert(out.shape[1], 'index_num', range(len(out)))

        return out.values
        
def _cols(self, metric, exclude):

        '''Helper to remove other than desired metric from data table'''

        cols = [col for col in self.data.columns if col not in exclude + [metric]]

        if isinstance(metric, list) is False:
            metric = [metric]
        for i, metric in enumerate(metric):
            cols.insert(i, metric)

        # make sure only unique values in col list
        cols = list(set(cols))
        
        cols = list(filter(None, cols))

        return cols