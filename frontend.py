import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

st.title('Import data')

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.title('EDA')

st.line_chart(chart_data)

st.title('Feature selection')

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)
#Ligand SMILES.
#Standard InChI.
#Target Source Organism According to Curator or DataSource. Organism associated with the protein Target.
#Ki (nM)
#Kd (nM)
#EC50 (nM)
#kon (M-1-s-1)
#koff (s-1)
#pH.
#Temp (C)
#Morgan	Extended-Connectivity Fingerprints
#Pubchem	Pubchem Substructure-based Fingerprints
#Daylight	Daylight-type fingerprints
#rdkit_2d_normalized	Normalized Descriptastorus
#AAC	Amino acid composition up to 3-mers
#PseudoAAC	Pseudo amino acid composition
#Conjoint_triad	Conjoint triad features
#Quasi-seq	Quasi-sequence order descriptor

#do i want to provide the network layer encodings? what if i'm not using the NN?
#default is a multilayer perceptron

	
st.title('Model selection')
    
option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option

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
data_load_state.text('Loading data...done!')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)