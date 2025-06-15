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
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

st.set_page_config(page_title="Laptop Price Prediction App", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Train Model", "Test Model", "Predict Price"])
if page == "Home":
    img_url = get_base64_image(r"C:\Users\PMLS\Desktop\ML-Project\Laptop\images\im.webp")
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url('{img_url}');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .title {{
            color: black;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }}
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="title">Laptop Price Prediction App</div>', unsafe_allow_html=True)
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    try:
        laptops = pd.read_csv("laptopPrice.csv")
    except FileNotFoundError:
        st.error("The file 'laptopPrice.csv' was not found.")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(laptops.head())

    st.subheader("Dataset Info")
    buffer = io.StringIO()
    laptops.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Statistical Summary")
    st.dataframe(laptops.describe())

    numerical_columns = ['Price', 'Number of Ratings', 'Number of Reviews']
    laptops = laptops[(np.abs(stats.zscore(laptops[numerical_columns])) < 3).all(axis=1)]
    laptops.replace([np.inf, -np.inf], np.nan, inplace=True)
    laptops.dropna(inplace=True)

    st.success(f"After removing outliers, dataset has {laptops.shape[0]} rows.")

    st.subheader("Pairplot of Numerical Features")
    fig = sns.pairplot(laptops[numerical_columns])
    st.pyplot(fig)

    st.subheader("Linear Relationship: Ratings vs Reviews")
    fig1 = sns.lmplot(x='Number of Ratings', y='Number of Reviews', data=laptops)
    st.pyplot(fig1)

    st.subheader("Histograms")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    laptops[numerical_columns].hist(ax=ax2)
    st.pyplot(fig2.figure)

    for feature in ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ssd', 'os', 'Touchscreen']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.countplot(ax=axes[0], x=laptops[feature])
        axes[0].set_title(f"Count by {feature}")
        sns.boxplot(ax=axes[1], x=laptops[feature], y=laptops['Price'])
        axes[1].set_title(f"Price vs {feature}")
        st.pyplot(fig)

elif page == "Train Model":
    st.title("Model Training")

    laptops = pd.read_csv("laptopPrice.csv")
    laptops.replace([np.inf, -np.inf], np.nan, inplace=True)
    laptops.dropna(inplace=True)
    numerical_columns = ['Price', 'Number of Ratings', 'Number of Reviews']
    laptops = laptops[(np.abs(stats.zscore(laptops[numerical_columns])) < 3).all(axis=1)]

    categorical_variables = laptops.select_dtypes(include='object').columns
    laptops = pd.get_dummies(laptops, columns=categorical_variables, drop_first=True)

    X = laptops.drop('Price', axis=1)
    y = laptops['Price']

    X_encoded = pd.get_dummies(X)
    feature_columns = X_encoded.columns.tolist()
    joblib.dump(feature_columns, 'model_features.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.success("Model trained successfully!")
    st.write(f"R² Score on Training Data: **{model.score(X_train, y_train):.4f}**")

    joblib.dump((model, X_test, y_test), 'model.pkl')

elif page == "Test Model":
    st.title("Model Testing")
    try:
        model, X_test, y_test = joblib.load('model.pkl')
    except:
        st.error("Model not found. Please train it first.")
        st.stop()

    predictions = model.predict(X_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, s=15)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.plot(y_test, y_test, color='red', lw=1)
    st.pyplot(fig)

    st.subheader("Residuals Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(x=(y_test - predictions), kde=True, bins=50, ax=ax2)
    st.pyplot(fig2)

    st.write(f"R² Score on Test Data: **{model.score(X_test, y_test):.4f}**")

elif page == "Predict Price":
    model, _, _ = joblib.load('model.pkl')
    feature_columns = joblib.load('model_features.pkl')

    st.title("💻 Laptop Price Prediction")

    brand = st.selectbox("Brand", ["HP", "Dell", "Lenovo", "Apple", "Asus", "Acer", "MSI"])
    processor_brand = st.selectbox("Processor Brand", ["Intel", "AMD", "Other"])
    processor_name = st.selectbox("Processor Name", ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7"])
    processor_gnrtn = st.selectbox("Processor Generation", ["4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th", "Not Available"])
    ram_gb = st.selectbox("RAM (GB)", [4, 8, 16, 32])
    ram_type = st.selectbox("RAM Type", ["DDR3", "DDR4", "LPDDR4", "LPDDR5"])
    ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024])
    hdd = st.selectbox("HDD (GB)", [0, 500, 1000, 2000])
    os = st.selectbox("Operating System", ["Windows", "macOS", "Linux", "DOS"])
    os_bit = st.selectbox("OS Bit", [32, 64])
    graphic_card_gb = st.selectbox("Graphics Card (GB)", [0, 2, 4, 6])
    weight = st.slider("Weight (kg)", 0.8, 3.5, 1.5)
    warranty = st.selectbox("Warranty (Years)", [1, 2, 3])
    Touchscreen = st.selectbox("Touchscreen", ["Yes", "No"])
    msoffice = st.selectbox("MS Office Preinstalled", ["Yes", "No"])
    rating = st.slider("User Rating", 1.0, 5.0, 4.0)
    num_ratings = st.number_input("Number of Ratings", min_value=0, step=1, value=100)
    num_reviews = st.number_input("Number of Reviews", min_value=0, step=1, value=10)

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

    if st.button("Predict Price"):
        prediction = model.predict(input_encoded)[0]
        prediction = max(prediction, 0) 
        st.success(f"💰 Estimated Laptop Price: ₹{int(prediction):,}")
