# Setup
import streamlit as st
import tensorflow as tf
import keras
import pandas as pd
import numpy as np

# load model
model = keras.models.load_model('inference_model.keras')

# load data
df = pd.read_csv('cleaned_bank.csv')

# Create Page

# create title for the project
with st.container():

    # create two cols for image and title
    img_col, title_col = st.columns([0.5, 1.5], vertical_alignment='top')

    with img_col:
        st.image('Images/logo.png', width=100)
    
    with title_col:
        st.write('# :rainbow[The Subscription Seer]')

# create the about project section
with st.container():

    # create an expander for about project section
    with st.expander('**:red[WHAT IS THIS ?]**', expanded=True):
        st.write('**:rainbow[THE SUBSCRIPTION SEER]** is a supervised deep learning model, that use the features to predict whether the client subscribed or not to the bank term deposit.')
        st.write('An **:green[Artificial Neural Net]** is used to build this model.')

# create an output section with empty method
with st.container():
    output = st.empty() 

# create the input section
with st.container():
    # we have 20 input features
    # lets create four columns in each row five times
    age_col, job_col, marital_col, education_col = st.columns(4)
    default_col, housing_col, loan_col, contact_col = st.columns(4)
    month_col, day_of_week_col, campaign_col, pdays_col = st.columns(4)
    previous_col, poutcome_col, emp_var_rate_col, cons_price_idx_col = st.columns(4)
    cons_conf_idx_col, euribor3m_col, nr_employed_col, previously_contacted_col = st.columns(4)

    with age_col:
        age = st.text_input('**Age**', placeholder=f'{df["age"].min()} to {df["age"].max()}')
        if age:
            age = np.int64(age)

    with job_col:
        job = st.selectbox('**Job**', options=df['job'].unique())
    
    with marital_col:
        marital = st.selectbox('**Marital status**', options=df['marital'].unique())

    with education_col:
        education = st.selectbox('**Education**', options=df['education'].unique())

    with default_col:
        default = st.selectbox('**Default**', options=df['default'].unique())

    with housing_col:
        housing = st.selectbox('**Housing**', options=df['housing'].unique())

    with loan_col:
        loan = st.selectbox('**Loan**', options=df['loan'].unique())

    with contact_col:
        contact = st.selectbox('**Contact**', options=df['contact'].unique())

    with month_col:
        month = st.selectbox('**Month**', options=df['month'].unique())

    with day_of_week_col:
        day_of_week = st.selectbox('**Day of week**', options=df['day_of_week'].unique())

    with campaign_col:
        campaign = st.text_input('**Campaign**', placeholder=f'{df["campaign"].min()} to {df["campaign"].max()}')
        if campaign:
            campaign = np.int64(campaign)

    with pdays_col:
        pdays = st.text_input('**Passed days**', placeholder=f'{df["pdays"].min()} to {df["pdays"].max()}')
        if pdays:
            pdays = np.int64(pdays)

    with previous_col:
        previous = st.text_input('**Previous**', placeholder=f'{df["previous"].min()} to {df["previous"].max()}')
        if previous:
            previous = np.int64(previous)

    with poutcome_col:
        poutcome = st.selectbox('**Poutcome**', options=df['poutcome'].unique())

    with emp_var_rate_col:
        emp_var_rate = st.text_input('**Employment variation rate**', placeholder=f'{df["emp.var.rate"].min()} to {df["emp.var.rate"].max()}')
        if emp_var_rate:
            emp_var_rate = np.float64(emp_var_rate)

    with cons_price_idx_col:
        cons_price_idx = st.text_input('**Consumer price index**', placeholder=f'{df["cons.price.idx"].min()} to {df["cons.price.idx"].max()}')
        if cons_price_idx:
            cons_price_idx = np.float64(cons_price_idx)

    with cons_conf_idx_col:
        cons_conf_idx = st.text_input('**Consumer conf index**', placeholder=f'{df["cons.conf.idx"].min()} to {df["cons.conf.idx"].max()}')
        if cons_conf_idx:
            cons_conf_idx = np.float64(cons_conf_idx)

    with euribor3m_col:
        euribor3m = st.text_input('**Euribor 3 month rate**', placeholder=f'{df["euribor3m"].min()} to {df["euribor3m"].max()}')
        if euribor3m:
            euribor3m = np.float64(euribor3m)

    with nr_employed_col:
        nr_employed = st.text_input('**No of employees**', placeholder=f'{df["nr.employed"].min()} to {df["nr.employed"].max()}')
        if nr_employed:
            nr_employed = np.float64(nr_employed)

    with previously_contacted_col:
        previously_contacted = st.selectbox('**Previously contacted**', options=df['previously_contacted'].unique())

    # raw inputs
    if age and campaign and pdays and previous and emp_var_rate and cons_price_idx and cons_conf_idx and euribor3m and nr_employed:
        
        sample = {  'age':age, 'job':job, 'marital':marital, 'education':education, 'default':default, 'housing':housing, 'loan':loan,
                    'contact':contact, 'month':month, 'day_of_week':day_of_week, 'campaign': campaign, 'pdays':pdays, 'previous':previous,
                    'poutcome':poutcome, 'emp.var.rate':emp_var_rate, 'cons.price.idx':cons_price_idx, 'cons.conf.idx':cons_conf_idx,
                    'euribor3m':euribor3m, 'nr.employed':nr_employed, 'previously_contacted':previously_contacted
                }
        
        # converting dictionary to tensor
        input_dict = {
            name: keras.ops.convert_to_tensor([value]) for name, value in sample.items()
        }

        # model prediction
        predictions = model.predict(input_dict)

        with output.container():
            st.write(f'### ***:orange[Term deposit subscription probability is {100 * predictions[0][0]:.2f} %]***')
        
# Container for sharing contents
with st.container():
     # five more cols for linking app with other platforms
    youtube_col, hfspace_col, madee_col, repo_col, linkedIn_col = st.columns([1,1.2,1.08,1,1], gap='small')

    # Youtube link
    with youtube_col:
        st.link_button('**VIDEO**', icon=':material/slideshow:', url='https://youtu.be/IDHr9Z4Q4iY', help='YOUTUBE')
    
    # Hugging Face Space link
    with hfspace_col:
        st.link_button('**HF SPACE**', icon=':material/sentiment_satisfied:', url='https://huggingface.co/spaces/madhav-pani/Subscription_Seer/tree/main', help='HUGGING FACE SPACE')

    # Madee Link
    with madee_col:
        st.button('**MADEE**', icon=':material/flight:', disabled=True, help='MADEE')

    # Repository Link
    with repo_col:
        st.link_button('**REPO**', icon=':material/code_blocks:', url='https://github.com/madhavpani/The_Subscription_seer', help='GITHUB REPOSITORY')

    # LinkedIn link
    with linkedIn_col:
        st.link_button('**CONNECT**', icon=':material/connect_without_contact:', url='https://www.linkedin.com/in/madhavpani', help='LINKEDIN')
    