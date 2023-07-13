import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.core.numeric import Infinity 
from scipy import stats
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split,cross_validate,cross_val_score, StratifiedShuffleSplit
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.linear_model import LinearRegression
import pickle
import math 
import shap

# *** Default Plotting Parameters ***
TICK_SIZE = 16
AXES_SIZE = 16
TITLE_SIZE = 22
DPI = 150
FIG_SIZE = (5, 4)
FACECOLOR = 'white'
SCATTER_ALPHA = 0.5
plt.rc('font', size=TICK_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=AXES_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=AXES_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=TICK_SIZE)    # legend fontsize
plt.rc('figure', titlesize=TITLE_SIZE)  # fontsize of the figure title

use_data_without_messy_datum = True # It is reccomended you do not change this. 
use_cep = False
use_log_scale_y = False
split_test_train = True
def remove_annoying_variables(X:np.array,Y:np.array):
    """Removes all elements with a y (ZPR [ev]) < 0.7, which are 
    likely due to bandgap convergene failures and should not be used.
    """
    thresh = 0.7
    bad_boy_list = [] # one might ask why bad datum are/can be gendered. I do not know why either. 
    for index,element in enumerate(Y):
        if element>thresh:
            bad_boy_list.append(index)
    new_x = np.zeros((X.shape[0]-len(bad_boy_list),X.shape[1]))
    new_y = np.zeros((Y.shape[0]-len(bad_boy_list),))
    index = 0
    for count,row in enumerate(X):
        if not count in bad_boy_list:
            new_x[index] = X[count]
            new_y[index] = Y[count]
            index += 1
    return new_x,new_y
def remove_below_10mev(X:np.array,Y:np.array):
    """ Returns all elements below 10 meV. Used in a 
    conference presentation, not used in final work. 
    """
    thresh = 0.1
    bad_boy_list = [] 
    for index,element in enumerate(Y):
        if element>thresh:
            bad_boy_list.append(index)
    new_x = np.zeros((X.shape[0]-len(bad_boy_list),X.shape[1]))
    new_y = np.zeros((Y.shape[0]-len(bad_boy_list),))
    index = 0
    for count,row in enumerate(X):
        if not count in bad_boy_list:
            new_x[index] = X[count]
            new_y[index] = Y[count]
            index += 1
    return new_x,new_y
def dict_to_xy(input_dict):
    """Converts dictionary-type storage to Numpy-naitive data structures. 
    """
    y = input_dict['zprs']
    x = []
    lst = input_dict.files
    for item in lst:
        if (item != 'zprs') and (item != 'names') and (item != 'zgerr'):
            x.append(input_dict[item])
    return np.transpose(x),y
def get_mse(y_pred,y_real):
    return np.mean((y_pred-y_real)**2)
def get_mae(y_pred,y_real):
    return np.mean(np.abs((y_pred-y_real)))
def plot_zprs_v_gaps(zprs,gaps,path=''):
    
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI, facecolor=FACECOLOR)
    #fig = plt.figure(figsize=(5, 4))
    ax.set_xlabel('Bandgaps [eV]')
    ax.set_ylabel('ZPRS [eV]')
    ax.set_title('ZPRS vs Static Gaps')
    ax.scatter(gaps, zprs )
    #matplotx.line_labels()
    #plt.figure(figsize=(5, 4), dpi=250, facecolor='white')
    plt.savefig(path + '/' + 'ZPRS vs Static Gaps',bbox_inches="tight")
def plot_zprs_v_gaps_colored(zprs,gaps,prop,prop_name = 'Property X',path=''):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI, facecolor=FACECOLOR)
    ax.set_xlabel('Bandgaps [eV]')
    ax.set_ylabel('ZPRS [eV]')
    ax.set_title('ZPRS vs Static Gaps w/' + str(prop_name))
    ax.scatter(gaps,zprs,c=prop)
    # make legend? might be hard 
    
    plt.savefig(path + '/' + 'ZPRS vs Static Gaps w ' + str(prop_name),bbox_inches="tight")
def plot_gap_v_epsilon(gaps, epsilon,path=''):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI, facecolor=FACECOLOR)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Bandgaps [eV]')
    ax.set_title('Gap vs Epsilon')
    ax.scatter(epsilon,gaps)
    
    plt.savefig(path + '/' + 'Gap vs Epsilon',bbox_inches="tight")
def plot_zpr_v_epsilon(zprs,epsilon,path=''):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI, facecolor=FACECOLOR)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('ZPRS [eV]')
    ax.set_title('ZPRS vs Epsilon')
    ax.scatter(epsilon,zprs)
    
    plt.savefig(path + '/' + 'ZPRS vs Epsilon',bbox_inches="tight")
def plot_zpr_epsilon_ens(zprs,epsilon,ens,path=''):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI, facecolor=FACECOLOR)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Electronegativity')
    #ax.set_zlabel('ZPRS (eV)')
    ax.set_title('ZPRS vs Epsilon, Electronegativity')
    ax.scatter(epsilon,zprs,c=ens)
    
    plt.savefig(path + '/' + 'ZPRS vs Epsilon, Electronegativity',bbox_inches="tight")
def train_linear_regression(X,y):
    reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)
    weights = reg.coef_
    y_pred = reg.predict(X)
    return r2,weights,y_pred
def plot_regression(y,y_guess,name = '',path=''):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI, facecolor=FACECOLOR)
    def myfunc(x):
        return x

    mymodel = list(map(myfunc, y))
    #ax.legend(['Blue - Training Data','Yellow - Test Data'])
    #ax.legend()
    ax.set_xlabel(name + ' Original')
    ax.set_ylabel(name + ' Predicted')
    ax.set_title(name + ' Original vs Predicted: Linear Regression')

    ax.plot(y, mymodel,c='black',label='Truth Line')

    #print("Slope: " + str(slope))

    ax.scatter(y,y_guess,c=['#1f77b4'],label='Train',alpha=SCATTER_ALPHA)
    #x.scatter(y_test,y_test_guess,c=['#bcbd22'],label='Test',alpha=SCATTER_ALPHA)
    ax.legend(loc="upper left")
    plt.show()
    
    plt.savefig(path + '/' + name + ' Original vs Predicted: Linear Regression',bbox_inches="tight")
def plot_pred_v_truth(y_test,y_train,y_test_guess,y_train_guess,name = '',split:int=-1,path=''):
    if split == -1:
        split = int(len(y_test)/(len(y_train)+len(y_test)))
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI, facecolor=FACECOLOR)
    slope, intercept, r, p, std_err = stats.linregress(y_train_guess, y_train)
    def myfunc(x):
        return x

    mymodel = list(map(myfunc, y_train))
    #ax.legend(['Blue - Training Data','Yellow - Test Data'])
    #ax.legend()
    ax.set_xlabel('ZPRs' + ' Original')
    ax.set_ylabel('ZPRs' + ' Predicted')
    ax.set_title(name + ' Original vs Predicted: ' + str(split) + ' Split')

    ax.plot(y_train, mymodel,c='black',label='Truth Line')

    #print("Slope: " + str(slope))

    ax.scatter(y_train,y_train_guess,c=['#1f77b4'],label='Train',alpha=SCATTER_ALPHA)
    ax.scatter(y_test,y_test_guess,c=['#bcbd22'],label='Test',alpha=SCATTER_ALPHA)
    ax.legend(loc="upper left")
    #plt.show()
    
    plt.savefig('plots/'+str(name),bbox_inches="tight")
def train_final_model(X_train,y_train,is_full=True,use_bayes=False):


    est = ExtraTreesRegressor(
        #n_estimators=1000, #sort of optimal... idk  # CHANGE BACK TO 100
        
        criterion='squared_error',
        #max_features = 6, #experimentally optimal
        n_jobs=-1,
        bootstrap = False,
        )
    y_train = y_train.ravel()
    if is_full:
        parameters = {'n_estimators':[10,50,100,200],
                    'max_depth':[None,1,3,5],
                    'min_samples_split':[2,5,10],
                    'min_samples_leaf':[1,2,5,10],
                    'max_features': [1,2,5,9],
                    }
    else:
        parameters = {'n_estimators':[10,50,100,200],
                    'max_depth':[None,1,3,5],
                    'min_samples_split':[2,5,10],
                    'min_samples_leaf':[1,2,5,10],
                    'max_features': [1,2,5],
                    }
    if use_bayes:
        clf = BayesSearchCV(est, parameters,cv=10)
    else:
        clf = GridSearchCV(est, parameters,cv=10)
    clf.fit(X_train, y_train)
    #print(clf.best_estimator_)
    best_est = clf.best_estimator_
    best_est.fit(X_train,y_train)
    return  best_est
def generate_shap_beehive(model,X,Y,is_full=True):
    fig, ax = plt.subplots(facecolor=FACECOLOR, dpi=DPI)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_values = np.array(shap_values.values)
    print(shap_values.shape)
    if is_full:
        names = ['epsilons',
                'ens',
                'becs',
                'mean_atomic_masses',
                'mean_ph_freqs',
                'first_optical_ph_freqs',
                'eq_gaps',
                'atoms_per_areas',
                'areal_densities']
        labels = [r'$\epsilon$',
                  r'$\Delta \chi$',
                  r'$Z^*$',
                  r'$\bar{M}$',
                  r'$\bar{\omega}$',
                  r'$\omega_1$',
                  r'$E_g$',
                  r'$\bar{N}$',
                  r'$\rho_{2D}$']
        names = labels
    else:
        names = ['ens','mean_atomic_masses',  'eq_gaps','atoms_per_areas','areal_densities']
        labels = [r'$\bar{M}$',
                  r'$\omega_1$',
                  r'$E_g$',
                  r'$\bar{N}$',
                  r'$\rho_{2D}$']
        names = labels
    y_pos = np.arange(len(names))
    #print(avg_stuff)
    shap.summary_plot(shap_values, X,feature_names=names,show=False)
    plt.tick_params(labelsize=15)
    if is_full:
        plt.savefig('plots/SHAP_violin_full',bbox_inches="tight")
    else:
        plt.savefig('plots/SHAP_violin_reduced',bbox_inches="tight")
def generate_shap_avg(model,X,Y,is_full=True):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_values = np.array(shap_values.values)
    if is_full:
        names = ['epsilons',
                'ens',
                'becs',
                'mean_atomic_masses',
                'mean_ph_freqs',
                'first_optical_ph_freqs',
                'eq_gaps',
                'atoms_per_areas',
                'areal_densities']
    else:
        names = ['ens','mean_atomic_masses',  'eq_gaps','atoms_per_areas','areal_densities']
        labels = [r'$\bar{M}$',
                  r'$\omega_1$',
                  r'$E_g$',
                  r'$\bar{N}$',
                  r'$\rho_{2D}$']
        names = labels
    names = np.array(names)
    fig, ax = plt.subplots(facecolor='white', dpi=DPI)
    avg_stuff = np.mean(np.absolute(shap_values),axis=0)
    # sorts everything
    sorted_indices = np.argsort(avg_stuff)
    avg_stuff = np.take_along_axis(avg_stuff,sorted_indices,axis=None)
    names = np.take_along_axis(names,sorted_indices,axis=None)
    ax.barh(names,avg_stuff)
    ax.set_xlabel('mean absolute SHAP value ')
    if is_full:
        ax.set_title('HZTG Data Feature Importance ')
    else:
        ax.set_title('HZTG Reduced Data Feature Importance ')
    plt.tick_params(labelsize=15)
    if is_full:
        plt.savefig('plots/SHAP_avg_full',bbox_inches="tight")
    else:
        plt.savefig('plots/SHAP_avg_reduced',bbox_inches="tight")
def evaluate_model(model,X_train,y_train,X_test,y_test,title):
    print(title)
    print('Train Score: ',model.score(X_train,y_train))
    print('Test Score: ',model.score(X_test,y_test))
    print('Train MSE: ',get_mse(model.predict(X_train),y_train))
    print('Test MSE: ',get_mse(model.predict(X_test),y_test))
    print('Test/Train Ratio: ',get_mse(model.predict(X_test),y_test)/get_mse(model.predict(X_train),y_train))
def evaluate_model_below_10mev(model,X_train,y_train,X_test,y_test,title):
    print(title,' below .1 EV: ')
    X_train,y_train = remove_below_10mev(X_train,y_train)
    X_test,y_test = remove_below_10mev(X_test,y_test)
    print(X_train.shape)
    print('Train Score: ',model.score(X_train,y_train))
    print('Test Score: ',model.score(X_test,y_test))
    print('Train MSE: ',get_mse(model.predict(X_train),y_train))
    print('Test MSE: ',get_mse(model.predict(X_test),y_test))
    print('Test/Train Ratio: ',get_mse(model.predict(X_test),y_test)/get_mse(model.predict(X_train),y_train))
def generate_paper_fig_2(y_test,y_train,y_test_guess,y_train_guess,model,X,Y,name = '',split:int=-1,path='',is_full=True):
    fig, (ax1, ax3) = plt.subplots(1,2,figsize=(9,4), dpi=DPI, facecolor=FACECOLOR)
    # **** Pred vs True **** 
    if split == -1:
        split = int(len(y_test)/(len(y_train)+len(y_test)))
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI, facecolor=FACECOLOR)
    slope, intercept, r, p, std_err = stats.linregress(y_train_guess, y_train)
    def myfunc(x):
        return x

    mymodel = list(map(myfunc, y_train))
    #ax.legend(['Blue - Training Data','Yellow - Test Data'])
    #ax.legend()
    ax1.set_xlabel('ZPR Ground Truth [eV]')
    ax1.set_ylabel('ZPRs Predicted [eV]')
    #ax1.set_title(name + ' Original vs Predicted: ' + str(split) + ' Split')

    ax1.plot(y_train, mymodel,c='black',label='Truth Line')

    #print("Slope: " + str(slope))

    ax1.scatter(y_train,y_train_guess,c=['#1f77b4'],label='Train',alpha=SCATTER_ALPHA)
    ax1.scatter(y_test,y_test_guess,c=['#bcbd22'],label='Test',alpha=SCATTER_ALPHA)
    ax1.legend(loc="upper left",frameon=False)

    # **** SHAP Average **** 
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_values = np.array(shap_values.values)
    if is_full:
        names = ['epsilons',
                'ens',
                'becs',
                'mean_atomic_masses',
                'mean_ph_freqs',
                'first_optical_ph_freqs',
                'eq_gaps',
                'atoms_per_areas',
                'areal_densities']
        labels = [r'$\epsilon$',
                  r'$\Delta \chi$',
                  r'$Z^*$',
                  r'$\bar{M}$',
                  r'$\bar{\omega}$',
                  r'$\omega_1$',
                  r'$E_g$',
                  r'$\bar{N}$',
                  r'$\rho_{2D}$']
        names = labels
    else:
        names = ['ens','mean_atomic_masses',  'eq_gaps','atoms_per_areas','areal_densities']
        labels = [r'$\bar{M}$',
                  r'$\omega_1$',
                  r'$E_g$',
                  r'$\bar{N}$',
                  r'$\rho_{2D}$']
        names = labels
    #names = ['','','','','','','','','']
    # names = [r'$Delta \Chi$',r'$\bar{M}$',r'$E_g$',r'$\bar{N}$',r'','6','7','8','9']
    names = np.array(names)
    #print(names.shape)
    avg_stuff = np.mean(np.absolute(shap_values),axis=0)
    # sorts everything
    sorted_indices = np.argsort(avg_stuff)
    print(len(sorted_indices))
    print(len(names))
    avg_stuff = np.take_along_axis(avg_stuff,sorted_indices,axis=None)
    names = np.take_along_axis(names,sorted_indices,axis=None)
    ax3.barh(names,avg_stuff)
    ax3.set_xlabel('Mean Absolute SHAP Val [eV]')