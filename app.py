import streamlit as st
import pandas as pd
import joblib

# Load model and expected feature columns
model = joblib.load("C:\\Users\\Sai Krishnan\\OneDrive\\Desktop\\ML Program\\Crop_production_prediction\\rf_model_1.pkl")
feature_columns = joblib.load("C:\\Users\\Sai Krishnan\\OneDrive\\Desktop\\ML Program\\Crop_production_prediction\\feature_columns.pkl")

df_cleaned = pd.read_csv("C:\\Users\Sai Krishnan\\OneDrive\\Desktop\\ML Program\\Crop_production_prediction\\df_cleaned.csv")

# Area Dropdown Setup
area_list = sorted(df_cleaned["Area"].unique())
area_code_map = df_cleaned.drop_duplicates(subset=["Area"])[["Area", "Area Code (M49)"]].set_index("Area").to_dict()["Area Code (M49)"]

# Item Dropdown Setup 
item_list = sorted(df_cleaned["Item"].unique())
item_code_map = df_cleaned.drop_duplicates(subset=["Item"])[["Item", "Item Code (CPC)"]].set_index("Item").to_dict()["Item Code (CPC)"]

# Streamlit UI
st.title("🌾 Crop Production Predictor")

st.markdown("Fill in the crop and region details below to predict production output.")

# Select Area 
selected_area = st.selectbox("Select Area", area_list)
area_code = area_code_map[selected_area]
st.write(f"📍 Area Code (M49): **{area_code}**")

# Select Item 
selected_item = st.selectbox("Select Item", item_list)
item_code = item_code_map[selected_item]
st.write(f"🌱 Item Code (CPC): **{item_code}**")

# Other Numeric Inputs
year = st.number_input("Year", min_value=1900, max_value=2100, value=2022)
area_harvested = st.number_input("Area harvested (ha)", min_value=0.0)
yield_val = st.number_input("Yield (kg/ha)", min_value=0.0)

#Build Input DataFrame
user_input = pd.DataFrame([{
    "Area Code (M49)": area_code,
    "Area": selected_area,
    "Item Code (CPC)": item_code,
    "Item": selected_item,
    "Year": year,
    "Area harvested": area_harvested,
    "Yield": yield_val
}])

#Preprocess Input
user_input_encoded = pd.get_dummies(user_input)
user_input_encoded = user_input_encoded.reindex(columns=feature_columns, fill_value=0)

# Prediction 
if st.button("Predict Production"):
    prediction = model.predict(user_input_encoded)[0]
    st.success(f"📈 Predicted Production: **{prediction:.2f}** units")

