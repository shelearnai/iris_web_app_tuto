import streamlit as st 
import joblib
import numpy as np

def load_model():
    model=joblib.load('iris_model.joblib')
    return model


species = ['setosa', 'versicolor', 'virginica']
image = ['setosa.png', 'versicolor.png', 'virginica.png']

def main():
    st.title('Iris Flower Predictor')
    st.sidebar.title("Inputs")
    sepal_length = st.sidebar.slider("sepal length (cm)",4.3,7.9,5.0)
    sepal_width = st.sidebar.slider("sepal width (cm)",2.0,4.4,3.6)
    petal_length = st.sidebar.slider("petal length (cm)",1.0,6.9,1.4)
    petal_width = st.sidebar.slider("petal width (cm)",0.1,2.5,0.2)

    inp = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    model=load_model()
    prediction = model.predict(inp)[0]
    print(prediction)

    # Main page
    st.title("Iris Flower Classification")
    st.write("This app correctly classifies iris flower among 3 possible species")

    result=species[prediction]
    st.write("**This flower belongs to " + result + " class**")
    st.image(image[prediction])

main()