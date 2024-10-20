import streamlit as st
import pandas as pd

# Title of the Streamlit app
st.title("CSV Data Viewer")

# GitHub raw file path to auto-load
default_file_path = (
    "https://raw.githubusercontent.com/iblend171/Trade_Lots_POC/main/OPEN_ind_tsx_350list.csv"
)

# Function to load the CSV from the URL
def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return None

# Load the CSV and display it
df = load_csv(default_file_path)
df = df.iloc[:,1:]

if df is not None:
    st.subheader("Tabular Data:")
    st.dataframe(df)

    st.subheader("Summary Statistics:")
    st.write(df.describe())

# print(df.head())
