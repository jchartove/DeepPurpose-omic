import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols

import sys

from DeepPurpose.pybiomed_helper import _GetPseudoAAC, CalculateAADipeptideComposition, \
calcPubChemFingerAll, CalculateConjointTriad, GetQuasiSequenceOrder
import torch
from torch.utils import data
from torch.autograd import Variable
try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
except:
    raise ImportError("Please install pip install git+https://github.com/bp-kelley/descriptastorus.")
from DeepPurpose.chemutils import get_mol, atom_features, bond_features, MAX_NB, ATOM_FDIM, BOND_FDIM
from subword_nmt.apply_bpe import BPE
import codecs
import pickle
import wget
from zipfile import ZipFile 
import os

from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from torch import nn 

from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, log_loss
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import pickle 
#torch.manual_seed(2)
#np.random.seed(3)
import copy
from prettytable import PrettyTable
import collections

import os

from DeepPurpose.utils import *
from DeepPurpose.model_helper import Encoder_MultipleLayers, Embeddings        
from DeepPurpose.encoders import *
from DeepPurpose import DTI

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

from matplotlib import pyplot as plt

def process_BindingDB_omic(path = None, df = None, temp_ph = True):
    """
    Fit all three models
            
    :param X: train features
    :param y: train targets
        
    TODO: parallelize this code across processors
    """
    #TODO: allow imputer, allow binary classification
    
    if not os.path.exists(path):
        os.makedirs(path)

    if df is not None:
        print('Loading Dataset from pandas input...')
    else:
        print('Loading Dataset from path...')
        df = pd.read_csv(path, sep = '\t', error_bad_lines=False)
    print('Beginning Processing...')
    df = df[df['Number of Protein Chains in Target (>1 implies a multichain complex)'] == 1.0]
    df = df[df['Ligand SMILES'].notnull()]
    
    df = df[['BindingDB Reactant_set_id', 'Ligand InChI', 'Ligand SMILES',\
                  'PubChem CID', 'UniProt (SwissProt) Primary ID of Target Chain',\
                  'Target Source Organism According to Curator or DataSource',\
                  'BindingDB Target Chain  Sequence', 'Kd (nM)', 'IC50 (nM)', 'Ki (nM)',\
                  'EC50 (nM)', 'kon (M-1-s-1)', 'koff (s-1)','pH','Temp (C)']]
    df.rename(columns={'BindingDB Reactant_set_id':'ID',
                        'Ligand SMILES':'SMILES',
                        'Ligand InChI':'InChI',
                        'PubChem CID':'PubChem_ID',
                        'UniProt (SwissProt) Primary ID of Target Chain':'UniProt_ID',
                        'BindingDB Target Chain  Sequence': 'Target Sequence',
                        'Target Source Organism According to Curator or DataSource': 'Organism',
                        'Kd (nM)':'Kd',
                        'IC50 (nM)':'IC50',
                        'Ki (nM)':'Ki',
                        'EC50 (nM)':'EC50',
                        'kon (M-1-s-1)':'kon',
                        'koff (s-1)':'koff',
                        'Temp (C)':'Temp'}, 
                        inplace=True)
    df['Temp'] = df['Temp'].str.rstrip('C')
    
    df = df[df['IC50'].notnull()]
    if temp_ph:
        df = df[df['Temp'].notnull()]
        df = df[df['pH'].notnull()]
        idx_str = ['IC50','Temp']
    else:
        idx_str = ['IC50']

    for label in idx_str:
        df[label] = df[label].astype(str).str.replace('>', '')
        df[label] = df[label].astype(str).str.replace('<', '')
        df[label] = df[label].apply(pd.to_numeric, errors='coerce')
    
    df['pH'] = df['pH'].apply(pd.to_numeric, errors='coerce')
    print('There are ' + str(len(df)) + ' drug target pairs.')

    df['pIC50'] = -np.log10(df['IC50'].astype(np.float32)*1e-9 + 1e-10)
    
    return df, df.SMILES.values, df['Target Sequence'].values, np.array(df['pIC50'])
    
def feature_select(df_data, drug_func_list = [drug2emb_encoder,smiles2daylight], 
                    prot_func_list = [CalculateConjointTriad, protein2emb_encoder]):
                    
    """
    Fit all three models
            
    :param X: train features
    :param y: train targets
        
    TODO: parallelize this code across processors
    """
    start = time()

    column_name = 'SMILES'
    for func in drug_func_list:
        print('Encoding: ', func.__name__)
        save_column_name = func.__name__
        unique = pd.Series(df_data[column_name].unique()).apply(func)
        unique_dict = dict(zip(df_data[column_name].unique(), unique))
        df_data[save_column_name] = [unique_dict[i] for i in df_data[column_name]]
        end = time()
        print("Elapsed time: ", end - start)
        
    column_name = 'Target Sequence'
    for func in prot_func_list:
        print('Encoding: ', func.__name__)
        save_column_name = func.__name__
        AA = pd.Series(df_data[column_name].unique()).apply(func)
        AA_dict = dict(zip(df_data[column_name].unique(), AA))
        df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
        end = time()
        print("Elapsed time: ", end - start)
    
    return df_data

def flattener(x):
    """
    Fit all three models
            
    :param X: train features
    :param y: train targets
        
    TODO: parallelize this code across processors
    """
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flattener(i)]
    else:
        return [x]

def data_process_omic(df_data, first_pass, include_org = True):
    """
    Fit all three models
            
    :param X: train features
    :param y: train targets
        
    TODO: parallelize this code across processors
    """
    if include_org:
        cat_list = pd.get_dummies(df_data['Organism'], prefix='var')
        df_data=df_data.join(cat_list)
    else:
        cat_list = 0
    
    discard=['SMILES','Target Sequence','Organism','IC50','pIC50','ID','InChI','PubChem_ID','UniProt_ID','Kd','Ki','EC50','kon','koff']
    df_vars=df_data.columns.values.tolist()
    to_keep=[i for i in df_vars if i not in discard]
    X=df_data[to_keep]
    
    #idxlist = df_data.index.astype(int)
    #first_pass = first_pass[idxlist]
    X[len(X.columns)] = first_pass
    
    print('Converting features to vectors (this takes a while)')
    #note to reader: i know i could be using "apply" for this but it actually takes longer
    Z = np.empty(shape=[len(X),len(flattener(X.iloc[0]))])
    for n in range(len(X)):
        Z[n] = flattener(X.iloc[n])
    
    #TODO: allow adjusting this
    tag = np.random.binomial(n=1, p=.8, size=len(X)) == 1

    # assign True indices to idx1 and False indices to index 2
    idx = np.array( range( len(X) ) )
    idx1, idx2 = idx[ tag ], idx[ np.logical_not( tag ) ]

    # sample from idx1 and idx2
    #TODO: allow adjusting this
    i1, i2 = np.random.choice( idx1, size=10000 ), np.random.choice( idx2, size=10000 )
    
    X_train = Z[idx1]
    X_test = Z[idx2]
    y_train =df_data['pIC50'].iloc[idx1].to_numpy(dtype = object)
    y_test = df_data['pIC50'].iloc[idx2].to_numpy(dtype = object)
    
    test_list = pd.isnull(y_train)
    res = [i for i, val in enumerate(test_list) if val] 
    y_train = np.delete(y_train,res,0)
    X_train = np.delete(X_train,res,0)

    test_list = pd.isnull(y_test)
    res = [i for i, val in enumerate(test_list) if val] 
    y_test = np.delete(y_test,res,0)
    X_test = np.delete(X_test,res,0)
        
    return X_train,X_test,y_train,y_test,cat_list
    
def get_typelist(X):
    """
    Fit all three models
            
    :param X: train features
    :param y: train targets
        
    TODO: parallelize this code across processors
    """
    typelist = []
    for i in list(X):
        if isinstance(X[i].iloc[0],np.ndarray):
            typelist.extend([i]*len(X[i].iloc[0]))
        elif isinstance(X[i].iloc[0],tuple):
            for n in X[i].iloc[0]:
                if isinstance(n,np.ndarray):
                     typelist.extend([i]*len(n))
        elif i[:4] == 'var_':
            typelist.append('organism')
        else:
            typelist.append(i)
    return typelist

def model_metrics(model, X_test, y_test, typelist = None):
    """
    Fit all three models
            
    :param X: train features
    :param y: train targets
        
    TODO: parallelize this code across processors
    """
    # Record actual values on test set
    #TODO: feature importances
    predictions = pd.DataFrame(y_test, columns = ['actual'])
    
    if isinstance(model,GBoostModel):
        predictions = model.predict(X_test)
    else:
        predictions['mid'] = pd.Series(model.predict(X_test))
        
    predictions['actual'] = y_test
    
    
    predictions['error'] = abs(predictions['actual'] - predictions['mid'])
    error1 = sum((predictions['error']<1)/len(predictions))
    error2 = sum((predictions['error']<2)/len(predictions))
    r = pearsonr(predictions['actual'],predictions['mid'])
    
    print("Average error: ", np.mean(predictions['error']))
    print("Fraction correct within one order of magnitude: ", error1 )
    print("Fraction correct within two orders of magnitude: ", error2 )
    print("Pearson's R = ",r)
    
    plt.scatter(predictions["actual"].astype(np.float), predictions["mid"].astype(np.float))#, yerr=[(predictions["upper"]-predictions["mid"]), (predictions["mid"]-predictions["lower"])])
    plt.xlabel('Reported pIC50')
    plt.ylabel('Predicted pIC50')
    plt.show()
    
    importances = None
    type_importance = None
    if typelist is not None:
        importances = pd.DataFrame({'importance':np.round(model.mid_model.feature_importances_,3)})
        out = importances.sort_values('importance',ascending=False)
        for i in range(len(importances)-len(typelist)):
                typelist.append('organism')

        importances["Type"] = typelist
        type_importance = importances.groupby(by=['Type']).sum()
        type_importance.sort_values('importance',ascending=False)
    return importances, type_importance


def model_usage(model, model2, X_drug, X_target, temp = 35, pH = 7, org = None):
    """
    Fit all three models
            
    :param X: train features
    :param y: train targets
        
    TODO: parallelize this code across processors
    """
    data = data_process(X_drug, X_target, [0,0], 
                        model.drug_encoding, model.target_encoding, 
                        split_method='no_split')
    first_pass = model.predict(data)
    pred = model2.predict_drug(X_drug,X_target,first_pass,temp = 35, pH = 7, org = None)
    print("Estimated pIC50: ", pred['mid'].values)
    print("Lower bound: ", pred['lower'].values)
    print("Upper bound: ", pred['upper'].values)
    return pred


class GBoostModel(BaseEstimator):
    """
    Adapted from https://github.com/WillKoehrsen/Data-Analysis/blob/master/prediction-intervals/prediction_intervals.ipynb
    Model that produces prediction intervals with a Scikit-Learn inteface
    
    :param lower_alpha: lower quantile for prediction, default=0.1
    :param upper_alpha: upper quantile for prediction, default=0.9
    :param **kwargs: additional keyword arguments for creating a GradientBoostingRegressor model
    """

    def __init__(self, lower_alpha=0.1, upper_alpha=0.9, drug_func_list= [drug2emb_encoder,smiles2daylight], 
                    prot_func_list = [CalculateConjointTriad, protein2emb_encoder], temp_ph = True, org_list = None,
                    n_estimators = 10, init_model = None, **kwargs):
        """
        Fit all three models
            
        :param X: train features
        :param y: train targets
        
        TODO: parallelize this code across processors
        """
        self.lower_alpha = lower_alpha
        self.upper_alpha = upper_alpha
        self.init_model = init_model

        self.drug_func_list = drug_func_list
        self.prot_func_list = prot_func_list
        self.temp_ph = temp_ph
        if org_list is not None:
            self.org_list = org_list
        else:
            self.org_list = []
            #TODO: figure out a way to do a default here
            
        # Three separate models
        self.lower_model = GradientBoostingRegressor(loss="quantile", alpha=self.lower_alpha, n_estimators = n_estimators, **kwargs)
        self.mid_model = GradientBoostingRegressor(loss="ls", n_estimators = n_estimators, **kwargs)
        self.upper_model = GradientBoostingRegressor(loss="quantile", alpha=self.upper_alpha, n_estimators = n_estimators, **kwargs)
        self.predictions = None
        self.scaler = StandardScaler()
        self.typelist = []

    def fit(self, X_train, y_train):
        """
        Fit all three models
            
        :param X: train features
        :param y: train targets
        
        TODO: parallelize this code across processors
        """
        
        train_scaled = self.scaler.fit_transform(X_train)
        
        print("Calculating estimates...")
        self.mid_model.fit(train_scaled, y_train)
        print("Getting lower bound...")
        self.lower_model.fit(train_scaled, y_train)
        print("Getting upper bound...")
        self.upper_model.fit(train_scaled, y_train)

    def predict(self, Z):
        """
        Fit all three models
            
        :param X: train features
        :param y: train targets
        
        TODO: parallelize this code across processors
        """
        predictions = pd.DataFrame()
        predictions["lower"] = self.lower_model.predict(Z)
        predictions["mid"] = self.mid_model.predict(Z)
        predictions["upper"] = self.upper_model.predict(Z)
        self.predictions = predictions
        return predictions
    
    def predict_drug(self, X_drug, X_target, first_pass = None, temp = 35, pH = 7, org = None):
        """
        Predict with all 3 models 
        
        :param X: test features
        :param y: test targets
        :return predictions: dataframe of predictions
        
        TODO: parallelize this code across processors
        """
        if first_pass is None:
            proc = DeepPurpose.utils.data_process(X_drug, X_target, 0, 
                                self.init_model.drug_encoding, model.target_encoding, 
                                split_method='no_split')
            first_pass = self.init_model.predict(proc)
        
        df_data = pd.DataFrame()
        df_data['SMILES'] = X_drug
        df_data['Target Sequence'] = X_target
        if self.temp_ph:
            df_data['pH'] = pH
            df_data['Temp'] = temp
        
        df_data = feature_select(df_data, self.drug_func_list, self.prot_func_list)
        df_data=df_data.join(self.org_list)
        if org is not None:
            df_data['var_'+org] = 1
            
        discard=['SMILES','Target Sequence','Organism','IC50','pIC50','ID','InChI','PubChem_ID','UniProt_ID','Kd','Ki','EC50','kon','koff']
        df_vars=df_data.columns.values.tolist()
        to_keep=[i for i in df_vars if i not in discard]
        X=df_data[to_keep]
        X['estimate'] = first_pass
        self.typelist = get_typelist(X)
        Z = flattener(np.asarray(X))
        typelist = get_typelist(X)
        Z = np.asarray(flattener(np.asarray(X)))
        s=np.isnan(Z)
        Z[s]=0.0

        predictions = self.predict(Z.reshape(1, -1))

        return predictions

    def plot_intervals(self, mid=False, start=None, stop=None):
        """
        Plot the prediction intervals
        
        :param mid: boolean for whether to show the mid prediction
        :param start: optional parameter for subsetting start of predictions
        :param stop: optional parameter for subsetting end of predictions
    
        :return fig: plotly figure
        """

        if self.predictions is None:
            raise ValueError("This model has not yet made predictions.")
            return
        
        fig = plot_intervals(predictions, mid=mid, start=start, stop=stop)
        return fig
    
    def calculate_and_show_errors(self):
        """
        Calculate and display the errors associated with a set of prediction intervals
        
        :return fig: plotly boxplot of absolute error metrics
        """
        if self.predictions is None:
            raise ValueError("This model has not yet made predictions.")
            return
        
        calculate_error(self.predictions)
        fig = show_metrics(self.predictions)
        return fig