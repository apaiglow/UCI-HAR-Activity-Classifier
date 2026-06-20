import streamlit as st
import pandas as pd

from src.predict import predict_csv

st.set_page_config(page_title="UCI HAR Activity Classifier", layout="centered")

st.title("🏃 Human Activity Recognition (UCI HAR)")
st.write("Upload a CSV file with 561 sensor features to predict activity.")

uploaded_file = st.file_uploader(
    "Upload your CSV file (561 features)",
    type=["csv"]
)

if uploaded_file is not None:

    st.success("File uploaded successfully!")

    st.write('Preview of uploaded data : ')

    df_preview = pd.read_csv(uploaded_file, header = None)
    st.dataframe(df_preview.head())

    if st.button('Predict Activity'):

        uploaded_file.seek(0)

        try:
            predictions, confidence = predict_csv(uploaded_file)
            st.success('Prediction completed!')
            st.write('Predicted Activities : ')
            result_df = pd.DataFrame({
                'Prediction' : predictions,
                'Confidence' : confidence
            })
            st.subheader('📊 Results')
            st.dataframe(result_df)
            st.subheader('📈 Summary')
            st.write(result_df['Prediction'].value_counts())
            csv = result_df.to_csv(index = False).encode('utf-8')
            st.download_button(label = '⬇ Download Predictions as CSV',
                               data = csv,
                               file_name = 'predictions.csv',
                               mime = 'text/csv')
        except Exception as e:
            st.error(f'Error during prediction : {e}')