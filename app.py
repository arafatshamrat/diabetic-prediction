"""
Streamlit Web Application for Diabetes Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .positive {
        background-color: #ff6b6b;
        color: white;
    }
    .negative {
        background-color: #51cf66;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model_path = 'models/diabetes_model.pkl'
        scaler_path = 'models/scaler.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None, "Model files not found. Please train the model first."
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load metadata if available
        metadata_path = 'models/model_metadata.txt'
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                for line in f:
                    key, value = line.strip().split(': ')
                    metadata[key] = value
        
        return model, scaler, metadata
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

def predict_diabetes(model, scaler, input_data):
    """Make prediction using the loaded model"""
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    return prediction, probability

def main():
    # Header
    st.markdown("<h1 style='text-align: center; color: white;'>🏥 Diabetes Prediction System</h1>", 
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>Predict diabetes risk using Machine Learning</h3>", 
                unsafe_allow_html=True)
    
    # Load model
    model, scaler, metadata = load_model()
    
    if model is None:
        st.error(metadata)
        st.info("Please run 'python src/train_model.py' to train the model first.")
        return
    
    # Display model info in sidebar
    st.sidebar.header("📊 Model Information")
    if isinstance(metadata, dict) and metadata:
        st.sidebar.success(f"**Model:** {metadata.get('model_name', 'N/A')}")
        st.sidebar.info(f"**Accuracy:** {float(metadata.get('accuracy', 0)):.2%}")
        st.sidebar.info(f"**ROC-AUC:** {float(metadata.get('roc_auc', 0)):.4f}")
    
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.info(
        "This application uses machine learning to predict the likelihood of diabetes "
        "based on various health metrics. Enter your health information below to get a prediction."
    )
    
    # Main content
    st.markdown("<div style='background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0;'>", 
                unsafe_allow_html=True)
    
    st.header("Enter Patient Information")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            help="Number of times pregnant"
        )
        
        glucose = st.slider(
            "Glucose Level (mg/dL)",
            min_value=0,
            max_value=200,
            value=120,
            help="Plasma glucose concentration"
        )
        
        blood_pressure = st.slider(
            "Blood Pressure (mm Hg)",
            min_value=0,
            max_value=140,
            value=70,
            help="Diastolic blood pressure"
        )
        
        skin_thickness = st.slider(
            "Skin Thickness (mm)",
            min_value=0,
            max_value=100,
            value=20,
            help="Triceps skin fold thickness"
        )
    
    with col2:
        insulin = st.slider(
            "Insulin Level (μU/mL)",
            min_value=0,
            max_value=900,
            value=80,
            help="2-Hour serum insulin"
        )
        
        bmi = st.slider(
            "BMI (Body Mass Index)",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            help="Body mass index (weight in kg/(height in m)^2)"
        )
        
        dpf = st.slider(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=2.5,
            value=0.5,
            step=0.01,
            help="Diabetes pedigree function (genetic factor)"
        )
        
        age = st.slider(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=30,
            help="Age in years"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Predict button
    if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })
        
        # Make prediction
        prediction, probability = predict_diabetes(model, scaler, input_data)
        
        # Display results
        st.markdown("---")
        st.header("Prediction Results")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction == 1:
                st.markdown(
                    f"<div class='prediction-box positive'>⚠️ HIGH RISK OF DIABETES</div>",
                    unsafe_allow_html=True
                )
                st.error(f"Probability: {probability[1]:.2%}")
                st.warning("⚕️ Please consult with a healthcare professional for proper diagnosis and treatment.")
            else:
                st.markdown(
                    f"<div class='prediction-box negative'>✅ LOW RISK OF DIABETES</div>",
                    unsafe_allow_html=True
                )
                st.success(f"Probability of No Diabetes: {probability[0]:.2%}")
                st.info("🎉 Keep maintaining a healthy lifestyle!")
        
        # Show probability breakdown
        st.markdown("---")
        st.subheader("Probability Breakdown")
        
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("No Diabetes", f"{probability[0]:.2%}")
        with prob_col2:
            st.metric("Diabetes", f"{probability[1]:.2%}")
        
        # Display input summary
        with st.expander("📋 View Input Summary"):
            st.dataframe(input_data, use_container_width=True)

if __name__ == "__main__":
    main()
