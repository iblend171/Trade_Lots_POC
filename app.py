import streamlit as st
import pandas as pd
import os

# Title of the Streamlit app
st.title("CSV Data Viewer")

# Default file path to auto-load
default_file_path = r"https://github.com/iblend171/Trade_Lots_POC/blob/main/OPEN_ind_tsx_350list.csv"

# Function to load the CSV
def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return None

# Check if the default CSV exists and load it
if os.path.exists(default_file_path):
    st.info(f"Loaded file: {default_file_path}")
    df = load_csv(default_file_path)
else:
    st.error("Default CSV file not found. Please ensure it exists at the specified path.")
    df = None

# Display the DataFrame if it was loaded successfully
if df is not None:
    st.subheader("Tabular Data:")
    st.dataframe(df)

    st.subheader("Summary Statistics:")
    st.write(df.describe())
