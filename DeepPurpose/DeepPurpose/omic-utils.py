import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
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
import sys

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
torch.manual_seed(2)
np.random.seed(3)
import copy
from prettytable import PrettyTable

import os

from DeepPurpose.utils import *
from DeepPurpose.model_helper import Encoder_MultipleLayers, Embeddings        
from DeepPurpose.encoders import *


# ESPF encoding
vocab_path = './DeepPurpose/ESPF/drug_codes_chembl_freq_1500.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./DeepPurpose/ESPF/subword_units_map_chembl_freq_1500.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

vocab_path = './DeepPurpose/ESPF/protein_codes_uniprot_2000.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
#sub_csv = pd.read_csv(dataFolder + '/subword_units_map_protein.csv')
sub_csv = pd.read_csv('./DeepPurpose/ESPF/subword_units_map_uniprot_2000.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

from DeepPurpose.chemutils import get_mol, atom_features, bond_features, MAX_NB

class Classifier_o(nn.Sequential):
	def __init__(self, feature_list, model_list, **config):
		super(Classifier, self).__init__()
		
		for feat in feature_list
			self.input_dim[feat] = 256 #un-hard-code this later

		self.model_list = model_list

		self.dropout = nn.Dropout(0.1)

		self.hidden_dims = config['cls_hidden_dims']
		layer_size = len(self.hidden_dims) + 1
		dims = sum(self.input_dim) + self.hidden_dims + [1]
		
		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

	def forward(self, v, feature_list):
		# each encoding
		for feat in feature_list
			v[feat] = self.model_list[feat]
		# concatenate and classify
		v_f = torch.cat(v, 1)
		for i, l in enumerate(self.predictor):
			if i==(len(self.predictor)-1):
				v_f = l(v_f)
			else:
				v_f = F.relu(self.dropout(l(v_f)))
		return v_f

def model_initialize_o(**config):
	model = DBTA_o(**config)
	return 

def data_process_o(X_drug = None, X_target = None, y=None, drug_encoding=None, target_encoding=None, 
				 split_method = 'random', frac = [0.7, 0.1, 0.2], random_seed = 1, sample_frac = 1, mode = 'DTI', X_drug_ = None, X_target_ = None):

	if split_method == 'repurposing_VS':
		y = [-1]*len(X_drug) # create temp y for compatitibility
	
	print('Drug Target Interaction Prediction Mode...')
	if isinstance(X_target, str):
		X_target = [X_target]
	if len(X_target) == 1:
		# one target high throughput screening setting
		X_target = np.tile(X_target, (length_func(X_drug), ))

	df_data = pd.DataFrame(zip(X_drug, X_target, y))
	df_data.rename(columns={0:'SMILES',
							1: 'Target Sequence',
							2: 'Label'}, 
							inplace=True)
	print('in total: ' + str(len(df_data)) + ' drug-target pairs')

	if sample_frac != 1:
		df_data = df_data.sample(frac = sample_frac).reset_index(drop = True)
		print('after subsample: ' + str(len(df_data)) + ' data points...') 
		
	df_data = encode_drug(df_data, drug_encoding)
	df_data = encode_protein(df_data, target_encoding)

	# dti split
	if DTI_flag:
		if split_method == 'repurposing_VS':
			pass
		else:
			print('splitting dataset...')

		if split_method == 'random': 
			train, val, test = create_fold(df_data, random_seed, frac)
		elif split_method == 'cold_drug':
			train, val, test = create_fold_setting_cold_drug(df_data, random_seed, frac)
		elif split_method == 'HTS':
			train, val, test = create_fold_setting_cold_drug(df_data, random_seed, frac)
			val = pd.concat([val[val.Label == 1].drop_duplicates(subset = 'SMILES'), val[val.Label == 0]])
			test = pd.concat([test[test.Label == 1].drop_duplicates(subset = 'SMILES'), test[test.Label == 0]])        
		elif split_method == 'cold_protein':
			train, val, test = create_fold_setting_cold_protein(df_data, random_seed, frac)
		elif split_method == 'repurposing_VS':
			train = df_data
			val = df_data
			test = df_data
		elif split_method == 'no_split':
			print('do not do train/test split on the data for already splitted data')
			return df_data.reset_index(drop=True)
		else:
			raise AttributeError("Please select one of the three split method: random, cold_drug, cold_target!")
	print('Done.')
	return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
	
class DBTA_o:
	'''
		Drug Target Binding Affinity 
	'''

	def __init__(self, **config):
		#drug_encoding = config['drug_encoding']
		#target_encoding = config['target_encoding']

		self.model_drug.MLP = MLP(config['input_dim_drug'], config['hidden_dim_drug'], config['mlp_hidden_dims_drug'])
        self.model_drug.CNN = CNN('drug', **config)
        self.model_drug.CNN_RNN = CNN_RNN('drug', **config)
        self.model_drug.transformer = transformer('drug', **config)
        self.model_drug.MPNN = MPNN(config['hidden_dim_drug'], config['mpnn_depth'])

		self.model_protein.MLP = MLP(config['input_dim_protein'], config['hidden_dim_protein'], config['mlp_hidden_dims_target'])
        self.model_protein.CNN = CNN('protein', **config)
        self.model_protein.CNN_RNN = CNN_RNN('protein', **config)
        self.model_protein.transformer = transformer('protein', **config)

		self.model = Classifier_o(self.model_drug, self.model_protein, **config)
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		#self.drug_encoding = drug_encoding
		#self.target_encoding = target_encoding
		self.result_folder = config['result_folder']
		if not os.path.exists(self.result_folder):
			os.mkdir(self.result_folder)            
		self.binary = False
		if 'num_workers' not in self.config.keys():
			self.config['num_workers'] = 0
		if 'decay' not in self.config.keys():
			self.config['decay'] = 0

	def test_(self, data_generator, model, repurposing_mode = False, test = False):
		y_pred = []
		y_label = []
		model.eval()
		for i, (v_d, v_p, label) in enumerate(data_generator):
			if self.drug_encoding == "MPNN" or self.drug_encoding == 'Transformer':
				v_d = v_d
			else:
				v_d = v_d.float().to(self.device)                
			if self.target_encoding == 'Transformer':
				v_p = v_p
			else:
				v_p = v_p.float().to(self.device)                
			score = self.model(v_d, v_p)
			if self.binary:
				m = torch.nn.Sigmoid()
				logits = torch.squeeze(m(score)).detach().cpu().numpy()
			else:
				logits = torch.squeeze(score).detach().cpu().numpy()
			label_ids = label.to('cpu').numpy()
			y_label = y_label + label_ids.flatten().tolist()
			y_pred = y_pred + logits.flatten().tolist()
			outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
		model.train()
		if self.binary:
			if repurposing_mode:
				return y_pred
			## ROC-AUC curve
			if test:
				roc_auc_file = os.path.join(self.result_folder, "roc-auc.jpg")
				plt.figure(0)
				roc_curve(y_pred, y_label, roc_auc_file, self.drug_encoding + '_' + self.target_encoding)
				plt.figure(1)
				pr_auc_file = os.path.join(self.result_folder, "pr-auc.jpg")
				prauc_curve(y_pred, y_label, pr_auc_file, self.drug_encoding + '_' + self.target_encoding)

			return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), log_loss(y_label, outputs), y_pred
		else:
			if repurposing_mode:
				return y_pred
			return mean_squared_error(y_label, y_pred), pearsonr(y_label, y_pred)[0], pearsonr(y_label, y_pred)[1], concordance_index(y_label, y_pred), y_pred

	def train(self, train, val, test = None, verbose = True):
		if len(train.Label.unique()) == 2:
			self.binary = True
			self.config['binary'] = True

		lr = self.config['LR']
		decay = self.config['decay']
		BATCH_SIZE = self.config['batch_size']
		train_epoch = self.config['train_epoch']
		if 'test_every_X_epoch' in self.config.keys():
			test_every_X_epoch = self.config['test_every_X_epoch']
		else:     
			test_every_X_epoch = 40
		loss_history = []

		self.model = self.model.to(self.device)

		# support multiple GPUs
		if torch.cuda.device_count() > 1:
			if verbose:
				print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
			self.model = nn.DataParallel(self.model, dim = 0)
		elif torch.cuda.device_count() == 1:
			if verbose:
				print("Let's use " + str(torch.cuda.device_count()) + " GPU!")
		else:
			if verbose:
				print("Let's use CPU/s!")
		# Future TODO: support multiple optimizers with parameters
		opt = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = decay)
		if verbose:
			print('--- Data Preparation ---')

		params = {'batch_size': BATCH_SIZE,
	    		'shuffle': True,
	    		'num_workers': self.config['num_workers'],
	    		'drop_last': False}
		if (self.drug_encoding == "MPNN"):
			params['collate_fn'] = mpnn_collate_func

		training_generator = data.DataLoader(data_process_loader(train.index.values, train.Label.values, train, **self.config), **params)
		validation_generator = data.DataLoader(data_process_loader(val.index.values, val.Label.values, val, **self.config), **params)
		
		if test is not None:
			info = data_process_loader(test.index.values, test.Label.values, test, **self.config)
			params_test = {'batch_size': BATCH_SIZE,
					'shuffle': False,
					'num_workers': self.config['num_workers'],
					'drop_last': False,
					'sampler':SequentialSampler(info)}
        
			if (self.drug_encoding == "MPNN"):
				params_test['collate_fn'] = mpnn_collate_func
			testing_generator = data.DataLoader(data_process_loader(test.index.values, test.Label.values, test, **self.config), **params_test)

		# early stopping
		if self.binary:
			max_auc = 0
		else:
			max_MSE = 10000
		model_max = copy.deepcopy(self.model)

		valid_metric_record = []
		valid_metric_header = ["# epoch"] 
		if self.binary:
			valid_metric_header.extend(["AUROC", "AUPRC", "F1"])
		else:
			valid_metric_header.extend(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
		table = PrettyTable(valid_metric_header)
		float2str = lambda x:'%0.4f'%x
		if verbose:
			print('--- Go for Training ---')
		t_start = time() 
		for epo in range(train_epoch):
			for i, (v_d, v_p, label) in enumerate(training_generator):
				if self.target_encoding == 'Transformer':
					v_p = v_p
				else:
					v_p = v_p.float().to(self.device) 
				if self.drug_encoding == "MPNN" or self.drug_encoding == 'Transformer':
					v_d = v_d
				else:
					v_d = v_d.float().to(self.device)                
					#score = self.model(v_d, v_p.float().to(self.device))
               
				score = self.model(v_d, v_p)
				label = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)

				if self.binary:
					loss_fct = torch.nn.BCELoss()
					m = torch.nn.Sigmoid()
					n = torch.squeeze(m(score), 1)
					loss = loss_fct(n, label)
				else:
					loss_fct = torch.nn.MSELoss()
					n = torch.squeeze(score, 1)
					loss = loss_fct(n, label)
				loss_history.append(loss.item())

				opt.zero_grad()
				loss.backward()
				opt.step()

				if verbose:
					if (i % 100 == 0):
						t_now = time()
						print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + \
							' with loss ' + str(loss.cpu().detach().numpy())[:7] +\
							". Total time " + str(int(t_now - t_start)/3600)[:7] + " hours") 
						### record total run time

			##### validate, select the best model up to now 
			with torch.set_grad_enabled(False):
				if self.binary:  
					## binary: ROC-AUC, PR-AUC, F1, cross-entropy loss
					auc, auprc, f1, loss, logits = self.test_(validation_generator, self.model)
					lst = ["epoch " + str(epo)] + list(map(float2str,[auc, auprc, f1]))
					valid_metric_record.append(lst)
					if auc > max_auc:
						model_max = copy.deepcopy(self.model)
						max_auc = auc   
					if verbose:
						print('Validation at Epoch '+ str(epo + 1) + ' , AUROC: ' + str(auc)[:7] + \
						  ' , AUPRC: ' + str(auprc)[:7] + ' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
						  str(loss)[:7])
				else:  
					### regression: MSE, Pearson Correlation, with p-value, Concordance Index  
					mse, r2, p_val, CI, logits = self.test_(validation_generator, self.model)
					lst = ["epoch " + str(epo)] + list(map(float2str,[mse, r2, p_val, CI]))
					valid_metric_record.append(lst)
					if mse < max_MSE:
						model_max = copy.deepcopy(self.model)
						max_MSE = mse
					if verbose:
						print('Validation at Epoch '+ str(epo + 1) + ' , MSE: ' + str(mse)[:7] + ' , Pearson Correlation: '\
						 + str(r2)[:7] + ' with p-value: ' + str(p_val)[:7] +' , Concordance Index: '+str(CI)[:7])
			table.add_row(lst)


		# load early stopped model
		self.model = model_max

		#### after training 
		prettytable_file = os.path.join(self.result_folder, "valid_markdowntable.txt")
		with open(prettytable_file, 'w') as fp:
			fp.write(table.get_string())

		if test is not None:
			if verbose:
				print('--- Go for Testing ---')
			if self.binary:
				auc, auprc, f1, loss, logits = self.test_(testing_generator, model_max, test = True)
				test_table = PrettyTable(["AUROC", "AUPRC", "F1"])
				test_table.add_row(list(map(float2str, [auc, auprc, f1])))
				if verbose:
					print('Validation at Epoch '+ str(epo + 1) + ' , AUROC: ' + str(auc)[:7] + \
					  ' , AUPRC: ' + str(auprc)[:7] + ' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
					  str(loss)[:7])				
			else:
				mse, r2, p_val, CI, logits = self.test_(testing_generator, model_max)
				test_table = PrettyTable(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
				test_table.add_row(list(map(float2str, [mse, r2, p_val, CI])))
				if verbose:
					print('Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) 
					  + ' with p-value: ' + str(p_val) +' , Concordance Index: '+str(CI))
			np.save(os.path.join(self.result_folder, str(self.drug_encoding) + '_' + str(self.target_encoding) 
				     + '_logits.npy'), np.array(logits))                
	
			######### learning record ###########

			### 1. test results
			prettytable_file = os.path.join(self.result_folder, "test_markdowntable.txt")
			with open(prettytable_file, 'w') as fp:
				fp.write(test_table.get_string())

		### 2. learning curve 
		fontsize = 16
		iter_num = list(range(1,len(loss_history)+1))
		plt.figure(3)
		plt.plot(iter_num, loss_history, "bo-")
		plt.xlabel("iteration", fontsize = fontsize)
		plt.ylabel("loss value", fontsize = fontsize)
		pkl_file = os.path.join(self.result_folder, "loss_curve_iter.pkl")
		with open(pkl_file, 'wb') as pck:
			pickle.dump(loss_history, pck)

		fig_file = os.path.join(self.result_folder, "loss_curve.png")
		plt.savefig(fig_file)
		if verbose:
			print('--- Training Finished ---')
          

	def predict(self, df_data):
		'''
			utils.data_process_repurpose_virtual_screening 
			pd.DataFrame
		'''
		print('predicting...')
		info = data_process_loader(df_data.index.values, df_data.Label.values, df_data, **self.config)
		self.model.to(device)
		params = {'batch_size': self.config['batch_size'],
				'shuffle': False,
				'num_workers': self.config['num_workers'],
				'drop_last': False,
				'sampler':SequentialSampler(info)}

		if (self.drug_encoding == "MPNN"):
			params['collate_fn'] = mpnn_collate_func


		generator = data.DataLoader(info, **params)

		score = self.test_(generator, self.model, repurposing_mode = True)
		return score

	def save_model(self, path_dir):
		if not os.path.exists(path_dir):
			os.makedirs(path_dir)
		torch.save(self.model.state_dict(), path_dir + '/model.pt')
		save_dict(path_dir, self.config)

	def load_pretrained(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

		if self.device == 'cuda':
			state_dict = torch.load(path)
		else:
			state_dict = torch.load(path, map_location = torch.device('cpu'))
		# to support training from multi-gpus data-parallel:
        
		if next(iter(state_dict))[:7] == 'module.':
			# the pretrained model is from data-parallel module
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				name = k[7:] # remove `module.`
				new_state_dict[name] = v
			state_dict = new_state_dict

		self.model.load_state_dict(state_dict)

		self.binary = self.config['binary']
        
def generate_config_o(drug_encoding = None, target_encoding = None, 
					result_folder = "./result/",
					input_dim_drug = 1024, 
					input_dim_protein = 8420,
					hidden_dim_drug = 256, 
					hidden_dim_protein = 256,
					cls_hidden_dims = [1024, 1024, 512],
					mlp_hidden_dims_drug = [1024, 256, 64],
					mlp_hidden_dims_target = [1024, 256, 64],
					batch_size = 256,
					train_epoch = 10,
					test_every_X_epoch = 20,
					LR = 1e-4,
					decay = 0,
					transformer_emb_size_drug = 128,
					transformer_intermediate_size_drug = 512,
					transformer_num_attention_heads_drug = 8,
					transformer_n_layer_drug = 8,
					transformer_emb_size_target = 64,
					transformer_intermediate_size_target = 256,
					transformer_num_attention_heads_target = 4,
					transformer_n_layer_target = 2,
					transformer_dropout_rate = 0.1,
					transformer_attention_probs_dropout = 0.1,
					transformer_hidden_dropout_rate = 0.1,
					mpnn_hidden_size = 50,
					mpnn_depth = 3,
					cnn_drug_filters = [32,64,96],
					cnn_drug_kernels = [4,6,8],
					cnn_target_filters = [32,64,96],
					cnn_target_kernels = [4,8,12],
					rnn_Use_GRU_LSTM_drug = 'GRU',
					rnn_drug_hid_dim = 64,
					rnn_drug_n_layers = 2,
					rnn_drug_bidirectional = True,
					rnn_Use_GRU_LSTM_target = 'GRU',
					rnn_target_hid_dim = 64,
					rnn_target_n_layers = 2,
					rnn_target_bidirectional = True,
					num_workers = 0                    
					):

	base_config = {'input_dim_drug': input_dim_drug,
					'input_dim_protein': input_dim_protein,
					'hidden_dim_drug': hidden_dim_drug, # hidden dim of drug
					'hidden_dim_protein': hidden_dim_protein, # hidden dim of protein
					'cls_hidden_dims' : cls_hidden_dims, # decoder classifier dim 1
					'batch_size': batch_size,
					'train_epoch': train_epoch,
					'test_every_X_epoch': test_every_X_epoch, 
					'LR': LR,
					'drug_encoding': drug_encoding,
					'target_encoding': target_encoding, 
					'result_folder': result_folder,
					'binary': False,
					'num_workers': num_workers                  
	}
	if not os.path.exists(base_config['result_folder']):
		os.makedirs(base_config['result_folder'])
	
		base_config['mlp_hidden_dims_drug'] = mlp_hidden_dims_drug # MLP classifier dim 1				
		base_config['input_dim_drug'] = 881 #could be 2048 or 200 or 2586
		base_config['cnn_drug_filters'] = cnn_drug_filters
		base_config['cnn_drug_kernels'] = cnn_drug_kernels
		base_config['rnn_Use_GRU_LSTM_drug'] = rnn_Use_GRU_LSTM_drug
		base_config['rnn_drug_hid_dim'] = rnn_drug_hid_dim
		base_config['rnn_drug_n_layers'] = rnn_drug_n_layers
		base_config['rnn_drug_bidirectional'] = rnn_drug_bidirectional 
		base_config['transformer_emb_size_drug'] = transformer_emb_size_drug
		base_config['transformer_num_attention_heads_drug'] = transformer_num_attention_heads_drug
		base_config['transformer_intermediate_size_drug'] = transformer_intermediate_size_drug
		base_config['transformer_n_layer_drug'] = transformer_n_layer_drug
		base_config['transformer_dropout_rate'] = transformer_dropout_rate
		base_config['transformer_attention_probs_dropout'] = transformer_attention_probs_dropout
		base_config['transformer_hidden_dropout_rate'] = transformer_hidden_dropout_rate
		base_config['hidden_dim_drug'] = transformer_emb_size_drug #could also be hidden_dim_drug
		base_config['batch_size'] = batch_size 
		base_config['mpnn_hidden_size'] = mpnn_hidden_size
		base_config['mpnn_depth'] = mpnn_depth

		base_config['mlp_hidden_dims_target'] = mlp_hidden_dims_target # MLP classifier dim 1				
		base_config['input_dim_protein'] = 30 #could be 343 or 100 or 4114
		base_config['cnn_target_filters'] = cnn_target_filters
		base_config['cnn_target_kernels'] = cnn_target_kernels
		base_config['rnn_Use_GRU_LSTM_target'] = rnn_Use_GRU_LSTM_target
		base_config['rnn_target_hid_dim'] = rnn_target_hid_dim
		base_config['rnn_target_n_layers'] = rnn_target_n_layers
		base_config['rnn_target_bidirectional'] = rnn_target_bidirectional 
		base_config['cnn_target_filters'] = cnn_target_filters
		base_config['cnn_target_kernels'] = cnn_target_kernels
		base_config['transformer_emb_size_target'] = transformer_emb_size_target
		base_config['transformer_num_attention_heads_target'] = transformer_num_attention_heads_target
		base_config['transformer_intermediate_size_target'] = transformer_intermediate_size_target
		base_config['transformer_n_layer_target'] = transformer_n_layer_target	
		base_config['transformer_dropout_rate'] = transformer_dropout_rate
		base_config['transformer_attention_probs_dropout'] = transformer_attention_probs_dropout
		base_config['transformer_hidden_dropout_rate'] = transformer_hidden_dropout_rate
		base_config['hidden_dim_protein'] = transformer_emb_size_target

	return base_config