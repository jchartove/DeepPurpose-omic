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

#uncomment the following for testing reproducibility
#torch.manual_seed(2)
#np.random.seed(3)

def process_BindingDB_omic(path = None, df = None, temp_ph = True):
    """
    Read BindingDB data that contains desired fields into a pandas dataframe.
    This function replaces DeepPurpose.process_BindingDB
            
    :param path: Filepath to BindingDB data in tsv format
    :param df: Loads dataset from an existing dataframe
    :param temp_ph: 'True' means you only want to select entries that contain temperature and pH info
        
    :output df: The full dataframe obtained from the 
    :output df.SMILES.values: Array of drugs in the dataframe
    :output df['Target Sequence'].values: Array of protein targets in the dataframe
    :output np.array(df['pIC50']): Array of IC50 values
    
    TODO: allow imputer, allow binary classification
    """
    
    if not os.path.exists(path):
        os.makedirs(path)

    if df is not None:
        print('Loading Dataset from pandas input...')
    else:
        print('Loading Dataset from path...')
        df = pd.read_csv(path, sep = '\t', error_bad_lines=False)
    print('Beginning Processing...')
    
    #DeepPurpose only works for single chain complexes
    df = df[df['Number of Protein Chains in Target (>1 implies a multichain complex)'] == 1.0]
    df = df[df['Ligand SMILES'].notnull()]
    
    #List of values drawn from BindingDB data
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
    
    #Discard data that doesn't contain desired fields
    df = df[df['IC50'].notnull()]
    if temp_ph:
        df = df[df['Temp'].notnull()]
        df = df[df['pH'].notnull()]
        idx_str = ['IC50','Temp']
    else:
        idx_str = ['IC50']
    
    #Force numeric
    for label in idx_str:
        df[label] = df[label].astype(str).str.replace('>', '')
        df[label] = df[label].astype(str).str.replace('<', '')
        df[label] = df[label].apply(pd.to_numeric, errors='coerce')
    df['pH'] = df['pH'].apply(pd.to_numeric, errors='coerce')
    print('There are ' + str(len(df)) + ' drug target pairs.')

    #convert to pIC50
    df['pIC50'] = -np.log10(df['IC50'].astype(np.float32)*1e-9 + 1e-10)
    
    return df, df.SMILES.values, df['Target Sequence'].values, np.array(df['pIC50'])
    
def feature_select(df_data, drug_func_list = [drug2emb_encoder,smiles2daylight], 
                    prot_func_list = [CalculateConjointTriad, protein2emb_encoder]):
                    
    """
    Adds encodings of drugs and targets to the dataframe
            
    :param df_data: A dataframe with a 'SMILES' column and a 'Target Sequence' column
    :param drug_func_list: List of drug encodings to include 
        (supported drug encodings: smiles2morgan, drug2emb_encoder, calcPubChemFingerAll, smiles2daylight)

    :param prot_func_list: List of protein encodings to include
        (supported protein encodings: CalculateConjointTriad, protein2emb_encoder, target2quasi)
        
    :output df_data: Dataframe that includes selected encodings as columns
    """
    start = time()

    column_name = 'SMILES'
    for func in drug_func_list:
        print('Encoding: ', func.__name__)
        #column name is name of encoder function
        save_column_name = func.__name__
        #iterate over unique drugs, applying encoder
        unique = pd.Series(df_data[column_name].unique()).apply(func)
        #save encoding per drug for lookup in a dictionary
        unique_dict = dict(zip(df_data[column_name].unique(), unique))
        df_data[save_column_name] = [unique_dict[i] for i in df_data[column_name]]
        end = time()
        print("Elapsed time: ", end - start)
        
    column_name = 'Target Sequence'
    for func in prot_func_list:
        print('Encoding: ', func.__name__)
        #column name is name of encoder function
        save_column_name = func.__name__
        #iterate over unique proteins, applying encoder
        AA = pd.Series(df_data[column_name].unique()).apply(func)
        #save encoding per protein for lookup in a dictionary
        AA_dict = dict(zip(df_data[column_name].unique(), AA))
        df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
        end = time()
        print("Elapsed time: ", end - start)
    
    return df_data

def flattener(x):
    """
    Recursive function that turns an iterable containing iterables 
    (eg an array containing lists, etc) into a 1D list
            
    :param x: any iterable
    :output x: iterable as list
    """
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flattener(i)] #list comprehension
    else:
        return [x]

def data_process_omic(df_data, first_pass, include_org = True):
    """
    Formats input data as feature vectors for use in model training
        and performs train/test split
    Replaces DeepPurpose.data_process
            
    :param df_data: Dataframe containing drug and target encodings 
        as well as other input features such as temperature and pH
    :param first_pass: Predictions from the initial DeepPurpose model
    :param include_org: Whether to include target source organism as a feature

    :output X_train: Training data array containing input features
    :output X_test: Testing data array containing input features
    :output y_train: Training data array containing pIC50
    :output y_test: Testing data array containing pIC50
    :output cat_list: List of organisms as categorical feature variables
    
    TODO: allow train/test split values to be input parameter
    TODO: if for some reason you want to train on something other than pIC50, allow for that
    """
    if include_org:
        #add organisms to the list of features as categorical dummy variables
        cat_list = pd.get_dummies(df_data['Organism'], prefix='var')
        df_data=df_data.join(cat_list)
    else:
        cat_list = 0
    
    #discarding non-numerical data, target (IC50/pIC50) data, and other kinetics data
    discard=['SMILES','Target Sequence','Organism','IC50','pIC50','ID','InChI','PubChem_ID','UniProt_ID','Kd','Ki','EC50','kon','koff']
    df_vars=df_data.columns.values.tolist()
    to_keep=[i for i in df_vars if i not in discard]
    X=df_data[to_keep]
    
    #adding in initial DeepPurpose predictions
    X[len(X.columns)] = first_pass
    
    print('Converting features to vectors (this takes a while)')
    #note: i know i could be using "apply" for this but that actually takes longer
    Z = np.empty(shape=[len(X),len(flattener(X.iloc[0]))])
    for n in range(len(X)):
        Z[n] = flattener(X.iloc[n])
    
    #TODO: allow adjusting this, this is the train/test probabilities
    #sklearn has a train/test split function but it's much slower because it's in-place
    tag = np.random.binomial(n=1, p=.8, size=len(X)) == 1

    # assign True indices to idx1 and False indices to index 2
    idx = np.array( range( len(X) ) )
    idx1, idx2 = idx[ tag ], idx[ np.logical_not( tag ) ]

    # sample from idx1 and idx2
    #TODO: allow adjusting and using this for if you don't have time to train on the entire dataset
    i1, i2 = np.random.choice( idx1, size=10000 ), np.random.choice( idx2, size=10000 )
    
    #apply split
    X_train = Z[idx1]
    X_test = Z[idx2]
    y_train =df_data['pIC50'].iloc[idx1].to_numpy(dtype = object)
    y_test = df_data['pIC50'].iloc[idx2].to_numpy(dtype = object)
    
    #good riddance to bad data
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
    Make a list of which encoding or input variable each feature came from for importance calculations
            
    :param X: dataframe with column names, incluing encodings
        
    :output typelist: List of where each feature came from (name of encoding or data column)
    """
    typelist = []
    for i in list(X):
        if isinstance(X[i].iloc[0],np.ndarray):
            #repeat column name for however long the encoding is
            typelist.extend([i]*len(X[i].iloc[0]))
        elif isinstance(X[i].iloc[0],tuple):
            #repeat column name for however long the encoding is
            for n in X[i].iloc[0]:
                if isinstance(n,np.ndarray):
                     typelist.extend([i]*len(n))
        elif i[:4] == 'var_': #categorical dummy variables
            typelist.append('organism')
        else: #name of numerical column
            typelist.append(i)
    return typelist

def model_metrics(model, X_test, y_test, typelist = None):
    """
    Assess model performance (error, accuracy, Pearson's R), plot actual data vs predictions,
        get feature importances for interpretable models
            
    :param model: model to be assessed (either DeepPurpose or Gradient Boosted)
    :param X_test: Testing data array containing input features
    :param y_test: Testing data array containing pIC50
    :param typelist: List of where each feature came from (see get_typelist)
        
    :output importances: Dataframe of feature importances
    :output type_importance: Dataframe of total feature importance per type
    """
    predictions = pd.DataFrame(y_test, columns = ['actual'])
    
    #the GBoostModel class gives predictions in the form of a 3 column dataframe
    if isinstance(model,GBoostModel):
        predictions = model.predict(X_test)
    else: #otherwise we can expect predictions to be a 1D array
        predictions['mid'] = pd.Series(model.predict(X_test))
        
    predictions['actual'] = y_test
    
    #absolute error
    predictions['error'] = abs(predictions['actual'] - predictions['mid'])
    
    #error in terms of order of magnitude (only works with pIC50)
    error1 = sum((predictions['error']<1)/len(predictions))
    error2 = sum((predictions['error']<2)/len(predictions))
    r = pearsonr(predictions['actual'],predictions['mid'])
    
    print("Average error: ", np.mean(predictions['error']))
    print("Fraction correct within one order of magnitude: ", error1 )
    print("Fraction correct within two orders of magnitude: ", error2 )
    print("Pearson's R = ",r)
    
    #plot actual vs predicted
    plt.scatter(predictions["actual"].astype(np.float), predictions["mid"].astype(np.float))
    #for error bars: yerr=[(predictions["upper"]-predictions["mid"]), (predictions["mid"]-predictions["lower"])])
    
    plt.xlabel('Reported pIC50')
    plt.ylabel('Predicted pIC50')
    plt.show()
    
    importances = None
    type_importance = None
    
    #get feature importances if possible
    if isinstance(model,GBoostModel):
        importances = pd.DataFrame({'importance':np.round(model.mid_model.feature_importances_,3)})
        #importances = importances.sort_values('importance',ascending=False)
    
    #get type importances if typelist is available
    if typelist is not None:
        for i in range(len(importances)-len(typelist)):
                typelist.append('organism')
        importances["Type"] = typelist
        type_importance = importances.groupby(by=['Type']).sum()
        type_importance = type_importance.sort_values('importance',ascending=False)
    return importances, type_importance


def model_usage(model, model2, X_drug, X_target, temp = 35, pH = 7, org = None):
    """
    Wrapper for using combined DeepPurpose/ Gradient Boosted models
            
    :param model: First pass model (DeepPurpose drug binding target affinity model)
    :param model2: Second pass model (GBoostModel)
    :param X_drug: Drug to be tested
    :param X_target: Target protein to be tested
    :param temp: Temperature in celsius (if available)
    :param pH: pH level of testing (if available)
    :param org: Target protein source organism (if available)
        
    :output pred: Predicted pIC50 (with lower and upper 80% prediction interval)
    """
    #format for use with DeepPurpose
    data = data_process(X_drug, X_target, [0,0], 
                        model.drug_encoding, model.target_encoding, 
                        split_method='no_split')
    first_pass = model.predict(data) #generate initial DeepPurpose estimate
    #generate Gradient Boosted estimate
    pred = model2.predict_drug(X_drug,X_target,first_pass,temp = 35, pH = 7, org = None)
    print("Estimated pIC50: ", pred['mid'].values)
    print("Lower bound: ", pred['lower'].values)
    print("Upper bound: ", pred['upper'].values)
    return pred


class GBoostModel(BaseEstimator):
    """
    Adapted from https://github.com/WillKoehrsen/Data-Analysis/blob/master/prediction-intervals/prediction_intervals.ipynb
    Gradient boosting model that produces pIC50 prediction intervals with a Scikit-Learn inteface
    
    :param lower_alpha: lower quantile for prediction, default=0.1
    :param upper_alpha: upper quantile for prediction, default=0.9
    :param drug_func_list: List of drug encodings to include 
        (supported drug encodings: smiles2morgan, drug2emb_encoder, calcPubChemFingerAll, smiles2daylight)
    :param prot_func_list: List of protein encodings to include
        (supported protein encodings: CalculateConjointTriad, protein2emb_encoder, target2quasi)
    :param temp_ph: Whether to include temperature and pH values as features
    :param org_list: List of model organism features
    :param n_estimators: Number of trees to generate when searching for optimal decision tree
    :param init_model: The DeepPurpose model used as a first pass
    :param **kwargs: additional keyword arguments for creating a GradientBoostingRegressor model
    """

    def __init__(self, lower_alpha=0.1, upper_alpha=0.9, drug_func_list= [drug2emb_encoder,smiles2daylight], 
                    prot_func_list = [CalculateConjointTriad, protein2emb_encoder], temp_ph = True, org_list = None,
                    n_estimators = 10, init_model = None, **kwargs):
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
            
        # Three separate models for lower bound/ prediction/ upper bound
        self.lower_model = GradientBoostingRegressor(loss="quantile", alpha=self.lower_alpha, n_estimators = n_estimators, **kwargs)
        self.mid_model = GradientBoostingRegressor(loss="ls", n_estimators = n_estimators, **kwargs)
        self.upper_model = GradientBoostingRegressor(loss="quantile", alpha=self.upper_alpha, n_estimators = n_estimators, **kwargs)
        self.predictions = None
        #self.scaler = StandardScaler()
        self.typelist = []

    def fit(self, X_train, y_train):
        """
        Fit all three models
            
        :param X: training features (array)
        :param y: training targets (pIC50)
        
        TODO: parallelize this code across processors
        """
        
        #probably don't actually need to scale
        #train_scaled = self.scaler.fit_transform(X_train)
        
        print("Calculating estimates...")
        self.mid_model.fit(X_train, y_train)
        print("Getting lower bound...")
        self.lower_model.fit(X_train, y_train)
        print("Getting upper bound...")
        self.upper_model.fit(X_train, y_train)

    def predict(self, Z):
        """
        Predict with all 3 models 
            
        :param Z: drug and target information formatted as vector
        
        :output predictions: predicted pIC50 values
        """
        predictions = pd.DataFrame()
        predictions["lower"] = self.lower_model.predict(Z)
        predictions["mid"] = self.mid_model.predict(Z)
        predictions["upper"] = self.upper_model.predict(Z)
        self.predictions = predictions
        return predictions
    
    def predict_drug(self, X_drug, X_target, first_pass = None, temp = 35, pH = 7, org = None):
        """
        Preprocessing wrapper to get drugs into proper format for model predictions
        
        :param X_drug: Drug to be tested
        :param X_target: Target protein to be tested
        :param first_pass: Predicted pIC50 from the initial DeepPurpose model (optional)
        :param temp: Temperature in celsius (if available)
        :param pH: pH level of testing (if available)
        :param org: Target protein source organism (if available)
        
        :output predictions: dataframe of predictions
        """
        #run DeepPurpose model if it hasn't been run yet
        if first_pass is None:
            proc = DeepPurpose.utils.data_process(X_drug, X_target, 0, 
                                self.init_model.drug_encoding, model.target_encoding, 
                                split_method='no_split')
            first_pass = self.init_model.predict(proc)
        
        df_data = pd.DataFrame()
        
        #get predictive features
        df_data['SMILES'] = X_drug
        df_data['Target Sequence'] = X_target
        if self.temp_ph:
            df_data['pH'] = pH
            df_data['Temp'] = temp
        
        #get encodings
        df_data = feature_select(df_data, self.drug_func_list, self.prot_func_list)
        df_data=df_data.join(self.org_list)
        if org is not None: #organism categorical variable feature
            df_data['var_'+org] = 1
        
        #discarding non-numerical data, target (IC50/pIC50) data, and other kinetics data
        discard=['SMILES','Target Sequence','Organism','IC50','pIC50','ID','InChI','PubChem_ID','UniProt_ID','Kd','Ki','EC50','kon','koff']
        df_vars=df_data.columns.values.tolist()
        to_keep=[i for i in df_vars if i not in discard]
        X=df_data[to_keep]
        
        X['estimate'] = first_pass #include DeepPurpose estimate
        self.typelist = get_typelist(X) #get predictor types
        Z = np.asarray(flattener(np.asarray(X)))  #vectorize
        s=np.isnan(Z) #remove NaNs
        Z[s]=0.0

        predictions = self.predict(Z.reshape(1, -1)) #reshape for correct orientation

        return predictions