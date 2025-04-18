import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Set Streamlit page config
st.set_page_config(page_title="Smart Data Analytics", layout="wide")
sns.set(style="whitegrid")

# Function to perform Exploratory Data Analysis (EDA)
def perform_eda(df):
    with st.expander("Data Processing & Missing Values Handling"):
        st.write("### Identifying Missing Values")
        missing_values = df.isnull().sum()
        st.write(missing_values)
        
        # Handling missing values
        df_filled = df.copy()
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df_filled[col].fillna(df[col].mean(), inplace=True)  # Replace with mean for numeric columns
            else:
                df_filled[col].fillna(df[col].mode()[0], inplace=True)  # Replace with mode for categorical columns
        
        st.write("### Missing Values After Handling")
        st.write(df_filled.isnull().sum())
    
    with st.expander("Statistical Measures"):
        st.write(df_filled.describe())
    
    numeric_df = df_filled.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        with st.expander("Correlation Matrix"):
            correlation_matrix = numeric_df.corr()
            st.write(correlation_matrix)
        with st.expander("Histograms"):
            for col in numeric_df.columns:
                plt.figure()
                sns.histplot(df_filled[col], bins=30, kde=True)
                st.pyplot(plt)
                plt.clf()
    else:
        st.write("No numeric columns available for EDA.")
    
    return df_filled  # Return the processed DataFrame

# Function to visualize data
def visualize_data(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        with st.expander("Pairplot"):
            sns.pairplot(numeric_df)
            st.pyplot(plt)
            plt.clf()
        with st.expander("Heatmap"):
            plt.figure(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt)
            plt.clf()
    else:
        st.write("No numeric columns available for visualization.")

# Function to detect anomalies
def detect_anomalies(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        contamination = st.slider("Select contamination level", 0.01, 0.2, 0.05)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df["anomaly"] = iso_forest.fit_predict(numeric_df)
        anomalies = df[df["anomaly"] == -1]
        st.write(f"### Detected {len(anomalies)} anomalies")
        st.write(anomalies)

        if len(numeric_df.columns) >= 2:
            plt.figure(figsize=(10, 5))
            sns.scatterplot(
                x=numeric_df.columns[0],
                y=numeric_df.columns[1],
                data=df,
                hue="anomaly",
                palette={1: "blue", -1: "red"}
            )
            st.pyplot(plt)
            plt.clf()
        else:
            st.write("Not enough numeric columns for anomaly visualization.")
    else:
        st.write("No numeric columns available for anomaly detection.")

# Main function to drive the app
def main():
    st.title("Smart Data Analytics")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head())

            # Sidebar navigation
            option = st.sidebar.radio("Choose an action:", ["EDA", "Visualize Data", "Detect Anomalies"])

            if option == "EDA":
                df = perform_eda(df)  # Ensure missing values are handled before proceeding
            elif option == "Visualize Data":
                visualize_data(df)
            elif option == "Detect Anomalies":
                detect_anomalies(df)

        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")

if  __name__ == "__main__":
    main()