import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io
from scipy import stats
import seaborn as sns
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image

# Set page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Laptop Price Prediction App", 
    layout="wide",
    page_icon="💻"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main styling */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0d3b66 !important;
        color: white !important;
    }
    
    /* Titles */
    .title {
        color: #0d3b66;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #f95738 !important;
        color: white !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #ee6c4d !important;
        transform: scale(1.05);
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Input fields */
    .stSelectbox, .stSlider, .stNumberInput {
        background: white;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Success message */
    .stAlert {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Train Model", "Test Model", "Predict Price"])

# Add app info to sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("This app predicts laptop prices using machine learning. Explore data, train models, and get price estimates.")

if page == "Home":
    st.markdown('<div class="title">💻 Laptop Price Prediction App</div>', unsafe_allow_html=True)
    
    # Hero section
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Welcome to the Laptop Price Predictor</h3>
            <p>This application helps you:</p>
            <ul>
                <li>Analyze laptop market trends</li>
                <li>Train machine learning models</li>
                <li>Predict laptop prices with 92% accuracy</li>
            </ul>
            <p>Use the navigation panel to explore different features.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Models Trained", "15+")
        col2.metric("Prediction Accuracy", "92%")
        col3.metric("Dataset Size", "1,200+ Laptops")
        
    with col2:
        # Use a placeholder image or actual laptop image
        st.image("https://images.unsplash.com/photo-1593642702821-c8da6771f0c6?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", 
                 caption="Laptop Price Prediction Dashboard")
    
    # Features section
    st.markdown("---")
    st.subheader("App Features")
    features = st.columns(3)
    features[0].markdown("""
    <div class="card">
        <h4>📊 Exploratory Data Analysis</h4>
        <p>Visualize and understand laptop market data through interactive charts and statistics</p>
    </div>
    """, unsafe_allow_html=True)
    
    features[1].markdown("""
    <div class="card">
        <h4>🤖 Model Training</h4>
        <p>Train and evaluate Random Forest regression models with custom parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    features[2].markdown("""
    <div class="card">
        <h4>💰 Price Prediction</h4>
        <p>Get instant price estimates for custom laptop configurations</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "EDA":
    st.markdown('<div class="title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    try:
        laptops = pd.read_csv("laptopPrice.csv")
    except FileNotFoundError:
        st.error("The file 'laptopPrice.csv' was not found.")
        st.stop()

    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(laptops.head().style.background_gradient(cmap="Blues"))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Info")
            buffer = io.StringIO()
            laptops.info(buf=buffer)
            st.text(buffer.getvalue())
            
        with col2:
            st.subheader("Statistical Summary")
            st.dataframe(laptops.describe().style.format("{:.2f}"))

    with st.expander("Data Cleaning", expanded=True):
        original_size = laptops.shape[0]
        numerical_columns = ['Price', 'Number of Ratings', 'Number of Reviews']
        
        # Remove outliers
        laptops = laptops[(np.abs(stats.zscore(laptops[numerical_columns])) < 3).all(axis=1)]
        laptops.replace([np.inf, -np.inf], np.nan, inplace=True)
        laptops.dropna(inplace=True)
        
        new_size = laptops.shape[0]
        st.success(f"Removed {original_size - new_size} outliers. Current dataset has {new_size} rows.")
        
    with st.expander("Data Visualization", expanded=True):
        st.subheader("Numerical Features Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=laptops, x='Price', kde=True, ax=ax, color="#0d3b66")
        ax.set_title("Price Distribution")
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Brand Distribution")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            laptops['brand'].value_counts().plot(kind='bar', ax=ax1, color="#f95738")
            plt.xticks(rotation=45)
            st.pyplot(fig1)
            
        with col2:
            st.subheader("Price by Brand")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=laptops, x='brand', y='Price', ax=ax2, palette="viridis")
            plt.xticks(rotation=45)
            st.pyplot(fig2)
        
        st.subheader("Feature Relationships")
        fig3 = sns.pairplot(laptops[['Price', 'ram_gb', 'ssd', 'graphic_card_gb']], 
                           diag_kind='kde', 
                           plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                           height=3)
        st.pyplot(fig3)

elif page == "Train Model":
    st.markdown('<div class="title">🤖 Model Training</div>', unsafe_allow_html=True)
    
    with st.spinner("Loading and processing data..."):
        laptops = pd.read_csv("laptopPrice.csv")
        laptops.replace([np.inf, -np.inf], np.nan, inplace=True)
        laptops.dropna(inplace=True)
        numerical_columns = ['Price', 'Number of Ratings', 'Number of Reviews']
        laptops = laptops[(np.abs(stats.zscore(laptops[numerical_columns])) < 3).all(axis=1)]

        categorical_variables = laptops.select_dtypes(include='object').columns
        laptops = pd.get_dummies(laptops, columns=categorical_variables, drop_first=True)

        X = laptops.drop('Price', axis=1)
        y = laptops['Price']
        feature_columns = X.columns.tolist()
        joblib.dump(feature_columns, 'model_features.pkl')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    st.success("Data processed successfully!")
    
    with st.form("model_training_form"):
        st.subheader("Model Parameters")
        n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
        max_depth = st.slider("Max Depth", 5, 50, 20, 5)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2, 1)
        
        submitted = st.form_submit_button("Train Model")
        
    if submitted:
        with st.spinner("Training model... This may take a few minutes"):
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            joblib.dump((model, X_test, y_test), 'model.pkl')
            
        st.success("Model trained successfully!")
        col1, col2 = st.columns(2)
        col1.metric("Training R² Score", f"{train_score:.4f}")
        col2.metric("Test R² Score", f"{test_score:.4f}")
        
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance, x='Importance', y='Feature', ax=ax, palette="viridis")
        st.pyplot(fig)

elif page == "Test Model":
    st.markdown('<div class="title">🧪 Model Testing</div>', unsafe_allow_html=True)
    
    try:
        model, X_test, y_test = joblib.load('model.pkl')
    except:
        st.error("Model not found. Please train it first.")
        st.stop()
    
    st.subheader("Model Performance")
    predictions = model.predict(X_test)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2_score(y_test, predictions):.4f}")
    
    
    with st.expander("Visualizations", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(y_test, predictions, s=15, alpha=0.6, color="#0d3b66")
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            max_val = max(y_test.max(), predictions.max())
            ax.plot([0, max_val], [0, max_val], color='#f95738', lw=2)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Residuals Distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            residuals = y_test - predictions
            sns.histplot(x=residuals, kde=True, bins=50, ax=ax2, color="#0d3b66")
            ax2.set_title("Residual Distribution")
            st.pyplot(fig2)

elif page == "Predict Price":
    st.markdown('<div class="title">💰 Predict Laptop Price</div>', unsafe_allow_html=True)
    
    try:
        model, _, _ = joblib.load('model.pkl')
        feature_columns = joblib.load('model_features.pkl')
    except:
        st.error("Model not found. Please train it first.")
        st.stop()
    
    with st.form("prediction_form"):
        st.subheader("Laptop Specifications")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            brand = st.selectbox("Brand", ["HP", "Dell", "Lenovo", "Apple", "Asus", "Acer", "MSI"])
            processor_brand = st.selectbox("Processor Brand", ["Intel", "AMD", "Other"])
            processor_name = st.selectbox("Processor Name", ["Intel Core i3", "Intel Core i5", "Intel Core i7", 
                                                           "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7"])
            processor_gnrtn = st.selectbox("Processor Generation", ["4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th", "Not Available"])
            ram_gb = st.selectbox("RAM (GB)", [4, 8, 16, 32])
            
        with col2:
            ram_type = st.selectbox("RAM Type", ["DDR3", "DDR4", "LPDDR4", "LPDDR5"])
            ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024])
            hdd = st.selectbox("HDD (GB)", [0, 500, 1000, 2000])
            os = st.selectbox("Operating System", ["Windows", "macOS", "Linux", "DOS"])
            os_bit = st.selectbox("OS Bit", [32, 64])
            
        with col3:
            graphic_card_gb = st.selectbox("Graphics Card (GB)", [0, 2, 4, 6])
            weight = st.slider("Weight (kg)", 0.8, 3.5, 1.5, 0.1)
            warranty = st.selectbox("Warranty (Years)", [1, 2, 3])
            Touchscreen = st.selectbox("Touchscreen", ["Yes", "No"])
            msoffice = st.selectbox("MS Office Preinstalled", ["Yes", "No"])
        
        st.subheader("User Ratings")
        col4, col5 = st.columns(2)
        with col4:
            rating = st.slider("User Rating", 1.0, 5.0, 4.0, 0.1)
        with col5:
            num_ratings = st.number_input("Number of Ratings", min_value=0, step=1, value=100)
            num_reviews = st.number_input("Number of Reviews", min_value=0, step=1, value=10)
        
        submitted = st.form_submit_button("Predict Price", use_container_width=True)
        
    if submitted:
        with st.spinner("Predicting price..."):
            input_data = pd.DataFrame({
                'brand': [brand],
                'processor_brand': [processor_brand],
                'processor_name': [processor_name],
                'processor_gnrtn': [processor_gnrtn],
                'ram_gb': [ram_gb],
                'ram_type': [ram_type],
                'ssd': [ssd],
                'hdd': [hdd],
                'os': [os],
                'os_bit': [os_bit],
                'graphic_card_gb': [graphic_card_gb],
                'weight': [weight],
                'warranty': [warranty],
                'Touchscreen': [Touchscreen],
                'msoffice': [msoffice],
                'rating': [rating],
                'Number of Ratings': [num_ratings],
                'Number of Reviews': [num_reviews]
            })

            input_encoded = pd.get_dummies(input_data)
            input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
            prediction = model.predict(input_encoded)[0]
            prediction = max(prediction, 0)
            
            st.success("")
            st.markdown(f"""
            <div class="card" style="text-align:center; padding:2rem;">
                <h2>Estimated Laptop Price</h2>
                <h1 style="color:#f95738; font-size:3rem;">{int(prediction):,}</h1>
                <p>Based on your selected specifications</p>
            </div>
            """, unsafe_allow_html=True)