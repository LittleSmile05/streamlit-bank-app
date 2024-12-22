import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Bank Customer Analytics & Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache functions
@st.cache_data
def load_data():
    try:
        data_path = Path("final_dataset.csv")
        if not data_path.exists():
            st.error("Data file not found. Please ensure 'final_dataset.csv' is in the 'data' directory.")
            st.stop()
        df = pd.read_csv(data_path, encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

@st.cache_resource
def load_model():
    try:
        model_path = Path("models")
        inference_pipeline = joblib.load("inference_pipeline.pkl")
        model_info = joblib.load("model_info.pkl")
        return inference_pipeline, model_info
    except FileNotFoundError:
        st.error("Model files not found. Please ensure model files are in the 'models' directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load data
df = load_data()

# Main title with emoji and styling
st.title("🏦 Bank Customer Analytics & Prediction Hub")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Analytics Dashboard", "Purchase Prediction"])

if page == "Analytics Dashboard":
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Average Balance", f"₼ {df['current balance'].mean():,.2f}")
    with col3:
        st.metric("Total Transactions", f"{df['count of transactions'].sum():,.0f}")
    with col4:
        st.metric("Active Devices", f"{df['Bought'].nunique():,}")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Customer Demographics",
        "💰 Transaction Analysis",
        "💳 Balance Insights",
        "📱 Device Usage",
        "🔍 Advanced Analytics"
    ])

    with tab1:
        st.header("Customer Demographics")
        col1, col2 = st.columns(2)
        
        with col1:
            gender_dist = df['gender'].value_counts()
            fig_gender = px.pie(
                values=gender_dist.values,
                names=gender_dist.index,
                title='Gender Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        with col2:
            gender_balance = df.groupby('gender')['current balance'].mean().reset_index()
            fig_balance = px.bar(
                gender_balance,
                x='gender',
                y='current balance',
                title='Average Balance by Gender',
                color='gender',
                text_auto='.2s'
            )
            st.plotly_chart(fig_balance, use_container_width=True)

    with tab2:
        st.header("Transaction Patterns")
        col1, col2 = st.columns(2)

        with col1:
            tx_trend = df.groupby('gender')['count of transactions'].mean().reset_index()
            fig_tx = px.bar(
                tx_trend,
                x='gender',
                y='count of transactions',
                title='Average Transactions per Customer',
                color='gender',
                text_auto='.2f'
            )
            st.plotly_chart(fig_tx, use_container_width=True)

        with col2:
            atm_interval = df.groupby('gender')['average day interval between atm transactions'].mean().reset_index()
            fig_atm = px.bar(
                atm_interval,
                x='gender',
                y='average day interval between atm transactions',
                title='ATM Transaction Frequency (Days)',
                color='gender',
                text_auto='.2f'
            )
            st.plotly_chart(fig_atm, use_container_width=True)

    with tab3:
        st.header("Balance Distribution")
        
        fig_box = px.box(
            df,
            x='gender',
            y='current balance',
            title='Balance Distribution by Gender',
            color='gender'
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Balance segments
        df['balance_segment'] = pd.qcut(df['current balance'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        balance_segments = df.groupby('balance_segment').size().reset_index(name='count')
        
        fig_segments = px.pie(
            balance_segments,
            values='count',
            names='balance_segment',
            title='Customer Balance Segments',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_segments, use_container_width=True)

    with tab4:
        st.header("Device Usage Analytics")
        col1, col2 = st.columns(2)

        with col1:
            device_dist = df['Bought'].value_counts()
            fig_device = px.pie(
                values=device_dist.values,
                names=device_dist.index,
                title='Device Distribution',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_device, use_container_width=True)

        with col2:
            device_gender = df.groupby(['gender', 'Bought']).size().reset_index(name='count')
            fig_device_gender = px.bar(
                device_gender,
                x='gender',
                y='count',
                color='Bought',
                title='Device Usage by Gender',
                barmode='group'
            )
            st.plotly_chart(fig_device_gender, use_container_width=True)

    with tab5:
        st.header("Correlation Analysis")
        
        correlation_cols = [
            'count of transactions',
            'current balance',
            'average day interval between atm transactions',
            'median of cash transactions'
        ]
        
        corr_matrix = df[correlation_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect='auto',
            title='Correlation Heatmap',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

else:  # Purchase Prediction Page
    st.header("Customer Purchase Prediction")
    st.write("Enter customer information to predict purchase likelihood")

    col1, col2 = st.columns(2)

    with col1:
        transaction_count = st.number_input(
            'Number of Transactions',
            min_value=0,
            value=10,
            help="Enter the total number of customer transactions"
        )
        
        current_balance = st.number_input(
            'Current Balance (₼)',
            min_value=0.0,
            value=1000.0,
            help="Enter the customer's current balance"
        )
        
        median_cash = st.number_input(
            'Median Cash Transaction (₼)',
            min_value=0.0,
            value=100.0,
            help="Enter the median value of cash transactions"
        )

    with col2:
        gender = st.selectbox(
            'Gender',
            options=['M', 'F'],
            help="Select customer's gender"
        )
        
        phone = st.selectbox(
            'Phone Type',
            options=['ANDROID', 'IOS'],
            help="Select customer's phone type"
        )

    if st.button('Predict Purchase Likelihood', use_container_width=True):
        try:
            inference_pipeline, model_info = load_model()
            
            if inference_pipeline is not None and model_info is not None:
                input_data = {
                    'count of transactions': transaction_count,
                    'current balance': current_balance,
                    'median of cash transactions': median_cash,
                    'gender': gender,
                    'phone': phone
                }
                
                input_df = pd.DataFrame([input_data])
                
                # Make prediction using the inference pipeline
                prediction = inference_pipeline.predict(input_df)[0]
                
                # Check if the pipeline has predict_proba method
                if hasattr(inference_pipeline, 'predict_proba'):
                    probability = inference_pipeline.predict_proba(input_df)[0]
                    confidence = probability[1]
                else:
                    # If predict_proba is not available, use the binary prediction
                    confidence = 1.0 if prediction == 1 else 0.0

                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Prediction",
                        "Will Purchase" if prediction == 1 else "Won't Purchase",
                        delta="Positive" if prediction == 1 else "Negative"
                    )
                
               

                # Display model information
                st.write("### Model Information")
                st.write(f"Model Type: {model_info['model_name']}")
                st.write(f"Model Performance:")
                st.write(f"- Accuracy: {model_info['performance']['accuracy']:.2%}")
                st.write(f"- F1 Score: {model_info['performance']['f1_score']:.2%}")

                st.write("### Input Summary")
                st.dataframe(pd.DataFrame([input_data]))
                
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            st.info("Please check your model pipeline configuration. The error might be due to:")
            st.write("1. The model in the pipeline doesn't support probability predictions")
            st.write("2. The pipeline transformation steps might need to be updated")
            st.write("3. The model files might not be properly loaded")

# Footer
st.markdown("---")
st.markdown("*Dashboard created with Streamlit and Python*")
