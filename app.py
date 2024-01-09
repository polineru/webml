from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('model_2')


def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
    return predictions


def run():
    from PIL import Image
    image = Image.open('logo-color.png')
    image_1 = Image.open('dataset-cover.jpeg')

    st.image(image, use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))

    st.sidebar.info('This app is created to predict intensity fastly')
    st.sidebar.success('https://www.cidp.edu.cn/')

    st.sidebar.image(image_1)

    st.title("Intensity Prediction App")

    if add_selectbox == 'Online':

        mmi = st.number_input('mmi', min_value=1, max_value=10, value=5)
        mag = st.number_input('magnitude', min_value=5., max_value=10., value=6.)
        magtype = st.selectbox('magType', ['mww', 'mwc', 'mwb', 'mw', 'Mi'])
        dep = st.number_input('depth', min_value=0., max_value=70., value=35.)
        mon = st.number_input('Month', min_value=1, max_value=12, value=6)
        day = st.number_input('Day', min_value=1, max_value=31, value=15)
        hour = st.number_input('hour', min_value=0, max_value=24, value=12)
        lat = st.number_input('latitude', min_value=-90.0, max_value=90.0, value=0.0)
        lon =st.number_input('longitude', min_value=-180.0, max_value=180.0, value=0.0)
        if st.checkbox('tsunami'):
            tsunami = 1
        else:
            tsunami = 0

        output = ""

        input_dict = {'mmi': mmi, 'magnitude': mag, 'magType': magtype, 'depth': dep,
                      'latitude' : lat, 'longitude' : lon, 'tsunami' : tsunami,
                      'Day' : day, 'hour' : hour, 'Month' : mon}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)

        st.success('The prediction intensity is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)


if __name__ == '__main__':
    run()