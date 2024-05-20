import altair as alt
import pandas as pd
from altair.vegalite.v4.api import Chart
import streamlit as st 

from main import AdaBoostNeww

def classify(features):
    prediction = AdaBoostNeww.predict(features)
    if prediction==True:
       return 'Hazardous'
    else:
       return 'Non-Hazardous'


def main():
    st.title("Asteroid Hazard Classification")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Asteroid Hazard Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    # Add input widgets for asteroid features
    Asteroid_ID = st.number_input('Asteroid ID', min_value=2002340, max_value=3781897)
    Est_Diameter_Min = st.number_input('Estimated Diameter (Min)', min_value=0.006677, max_value=0.667659)
    Est_Diameter_Max = st.number_input('Estimated Diameter (Max)', min_value=0.014929, max_value=1.492932)
    Relative_Velocity = st.number_input('Relative Velocity', min_value=3739.155217, max_value=128299.337098)
    standardized_distance = st.number_input('Miss Distance', min_value=-1.9043900458903815, max_value=1.5783919092585006)
    Absolute_Magnitude = st.number_input('Absolute Magnitude', min_value=18.0, max_value=28.0)
    Orbit_ID = st.number_input('Orbit ID', min_value=1, max_value=89)
    Orbit_Uncertainty = st.number_input('Orbit Uncertainty', min_value=0, max_value=9)

    # Create a list of features
    features = [[Asteroid_ID, Est_Diameter_Min, Est_Diameter_Max, Relative_Velocity, standardized_distance, Absolute_Magnitude, Orbit_ID, Orbit_Uncertainty]]
    
    if st.button('Classify'):
        st.success(classify(features))

if __name__ == '__main__':
    main()
