import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import os
import time

a = st.radio(
	'What do you want to do?',
	('Generate new IC50 prediction model','Load pretrained model'))

if a == 'Generate new IC50 prediction model':
	st.title('BindingDB input feature selection')

	ki = st.checkbox('Ki (nM)')
	kd = st.checkbox('Kd (nM)')
	ec50 = st.checkbox('EC50 (nM)')
	organism = st.checkbox('Target Source Organism') 
	ph = st.checkbox('pH')
	temp = st.checkbox('Temp (C)')
	
	interp = st.selectbox(
		'Method to impute missing values?',
	   ('Simple', 'Iterative','k nearest neighbors'))

	st.title('Input encodings')

	st.subheader('Drug encoding')
	morgan = st.checkbox('Morgan Extended-Connectivity Fingerprints')
	pubchem = st.checkbox('Pubchem Substructure-based Fingerprints')
	daylight = st.checkbox('Daylight-type fingerprints')
	rdk = st.checkbox('Normalized Descriptastorus')
	
	st.subheader('Target encoding')
	aac = st.checkbox('Amino acid composition up to 3-mers')
	pAAC = st.checkbox('Pseudo amino acid composition')
	conjoint = st.checkbox('Conjoint triad features')
	quasi = st.checkbox('Quasi-sequence order descriptor')

	st.title('Model selection')

	custom_example = "drug_encoding = drug_encoding, target_encoding = target_encoding, cls_hidden_dims = [1024,1024,512], train_epoch = 3, LR = 0.001, batch_size = 128,hidden_dim_drug = 128,mpnn_hidden_size = 128,mpnn_depth = 3"
		
	option = st.selectbox(
		'Which model architecture should be trained on this input?',
	   ('DeepIC50', 'MPNN', 'SVM', 'Ridge regressor', 'XGBoost', 'Custom architecture'))

	if option == 'Custom architecture':
		st.text_input('Paste model configuration here', value=custom_example)

	output = st.radio(
		'Which value should the model predict?',
	   ('IC50','pIC50','Order of magnitude of IC50'))

	if st.button('Train model'):
		'Training ', option, '...'

		# Add a placeholder
		latest_iteration = st.empty()
		bar = st.progress(0)

		for i in range(100):
		  # Update the progress bar with each iteration.
		  latest_iteration.text(f'Iteration {i+1}')
		  bar.progress(i + 1)
		  time.sleep(0.1)

		'Model complete'

		sav = st.button('Save trained model')

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

if a == 'Load pretrained model':
	filename = file_selector()
	st.write('You selected `%s`' % filename)

	'Loading...'

	# Add a placeholder
	latest_iteration = st.empty()
	bar = st.progress(0)

	for i in range(100):
	  # Update the progress bar with each iteration.
	  latest_iteration.text(f'{i+1} percent')
	  bar.progress(i + 1)
	  time.sleep(0.1)

	'Model loaded'

st.title('Model usage')

drug_example = "CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N"
target_example = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNPGILPIQPALTPRKRATVQPPPQAAGSSNQPGLLASVPQPKPQAPPSQPLPQTQAKQPQAPPTPQQTPSTQAQGLPAQAQATPQHQQQLFLKQQQQQQQPPPAQQQPAGTFYQQQQAQTQQFQAVHPATQKPAIAQFPVVSQGGSQQQLMQNFYQQQQQQQQQQQQQQLATALHQQQLMTQQAALQQKPTMAAGQQPQPQPAAAPQPAPAQEPAIQAPVRQQPKVQTTPPPAVQGQKVGSLTPPSSPKTQRAGHRRILSDVTHSAVFGVPASKSTQLLQAAAAEASLNKSKSATTTPSGSPRTSQQNVYNPSEGSTWNPFDDDNFSKLTAEELLNKDFAKLGEGKHPEKLGGSAESLIPGFQSTQGDAFATTSFSAGTAEKRKGGQTVDSGLPLLSVSDPFIPLQVPDAPEKLIEGLKSPDTSLLLPDLLPMTDPFGSTSDAVIEKADVAVESLIPGLEPPVPQRLPSQTESVTSNRTDSLTGEDSLLDCSLLSNPTTDLLEEFAPTAISAPVHKAAEDSNLISGFDVPEGSDKVAEDEFDPIPVLITKNPQGGHSRNSSGSSESSLPNLARSLLLVDQLIDL"

drug = st.text_input('Drug (SMILES format)', value=drug_example)
target = st.text_input('Target amino chain', value=target_example)

if ki:
	st.text_input('Ki (nM)')
if kd:
	st.text_input('Kd (nM)')
if ec50:
	st.text_input('EC50 (nM)')
if organism:
	st.text_input('Target Source Organism') 
if ph:
	st.text_input('pH')
if temp:
	st.checkbox('Temp (C)')

if st.button('Predict IC50'):
	'The predicted pIC50 is 7.395412921905518'

st.title('Model metrics')

from PIL import Image
image1 = Image.open('imgs/metric1.png')
image2 = Image.open('imgs/metric2.png')

st.image(image1, use_column_width=True)
st.image(image2, use_column_width=True)

st.title('EDA')

image3 = Image.open('imgs/eda1.png')
image4 = Image.open('imgs/eda2.png')

st.image(image3, use_column_width=True)
st.image(image4, use_column_width=True)

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
    
    # Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Data loaded')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Distribution of IC50 values')
hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)