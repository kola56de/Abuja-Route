import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from io import BytesIO
import pydeck as pdk

st.set_page_config(page_title="Abuja Route Speed Predictor (GIS)", layout="wide")

st.title("🗺 Abuja Route Speed Predictor with GIS Visualization")

# --- Step 1: Default Maitama Dataset ---
default_data = {
    "ROUTE": [
        "Banex - Hospital Junction", 
        "Banex - University Junction", 
        "Banex - Wuse Market Junction",
        "Banex - Head of Service Junction", 
        "Hospital Junction - University Junction",
        "Hospital Junction - Wuse Market Junction", 
        "Hospital Junction - Head of Service",
        "Wuse Market Junction - University Junction", 
        "Wuse Market Junction - Head of Service Junction",
        "University Junction - Head of Service"
    ],
    "LENGTH_km": [2.5, 3.9, 1.7, 7.0, 1.3, 2.5, 4.8, 3.6, 5.3, 1.0],
    "TIME_sec": [471, 364, 101, 408, 132, 227, 218, 185, 312, 149],
    "AVG_SPEED": [19, 29, 62, 61, 35, 39, 28, 37, 62, 25],
    "START_LAT": [9.084, 9.084, 9.084, 9.084, 9.080, 9.080, 9.080, 9.070, 9.070, 9.065],
    "START_LON": [7.489, 7.489, 7.489, 7.489, 7.500, 7.500, 7.500, 7.495, 7.495, 7.498],
    "END_LAT": [9.080, 9.072, 9.085, 9.060, 9.072, 9.085, 9.060, 9.072, 9.060, 9.060],
    "END_LON": [7.500, 7.510, 7.480, 7.470, 7.510, 7.480, 7.470, 7.510, 7.470, 7.470]
}
default_df = pd.DataFrame(default_data)

# --- Step 2: Upload Option ---
st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = default_df.copy()
else:
    st.info("Using default Maitama dataset.")
    df = default_df.copy()

# --- Step 3: Validate Required Columns ---
required_cols = {"LENGTH_km", "TIME_sec", "AVG_SPEED"}
if not required_cols.issubset(df.columns):
    st.error(f"Dataset must contain: {required_cols}")
else:
    st.subheader("📋 Dataset Preview")
    st.dataframe(df, use_container_width=True)

    # --- Step 4: Train Model ---
    X = df[["LENGTH_km", "TIME_sec"]]
    y = df["AVG_SPEED"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    df["PREDICTED_SPEED"] = model.predict(X)
    df["DIFFERENCE"] = abs(df["PREDICTED_SPEED"] - df["AVG_SPEED"])

    r2 = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))

    st.subheader("🧮 Model Performance")
    st.write(f"**R² Score:** {r2:.2f}")
    st.write(f"**Mean Absolute Error:** {mae:.2f}")

    # --- Step 5: Map Visualization ---
    st.subheader("🗺 Route Map Visualization")

    if {"START_LAT", "START_LON", "END_LAT", "END_LON"}.issubset(df.columns):
        # Create line data for pydeck
        line_data = []
        for _, row in df.iterrows():
            line_data.append({
                "from": [row["START_LON"], row["START_LAT"]],
                "to": [row["END_LON"], row["END_LAT"]],
                "speed": row["AVG_SPEED"],
                "route": row["ROUTE"]
            })

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=df["START_LAT"].mean(),
                longitude=df["START_LON"].mean(),
                zoom=12,
                pitch=45,
            ),
            layers=[
                pdk.Layer(
                    "LineLayer",
                    data=line_data,
                    get_source_position="from",
                    get_target_position="to",
                    get_color="[255 - speed*3, speed*3, 120]",
                    get_width=4,
                    pickable=True,
                    auto_highlight=True,
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=pd.concat([
                        df[["START_LAT", "START_LON"]].rename(columns={"START_LAT": "lat", "START_LON": "lon"}),
                        df[["END_LAT", "END_LON"]].rename(columns={"END_LAT": "lat", "END_LON": "lon"})
                    ]),
                    get_position='[lon, lat]',
                    get_color='[0, 100, 255]',
                    get_radius=60,
                )
            ],
            tooltip={"text": "{route}\nSpeed: {speed} km/h"}
        ))
    else:
        st.warning("No coordinate columns found (START_LAT, START_LON, END_LAT, END_LON). Map not displayed.")

    # --- Step 6: Visualization & Download ---
    st.subheader("📊 Predicted Speeds Comparison")
    st.dataframe(df, use_container_width=True)

    output = BytesIO()
    df.to_csv(output, index=False)
    st.download_button(
        label="📥 Download Predictions (CSV)",
        data=output.getvalue(),
        file_name="abuja_route_predictions.csv",
        mime="text/csv"
    )

# --- Step 7: App Summary ---
st.divider()
st.write("🧭 **App Summary:**")
st.markdown("""
- Upload or use sample Maitama District data  
- Model predicts average route speeds  
- See results on an **interactive map**  
- Download prediction results as CSV  
- Works with any Abuja district with optional GPS coordinates  
""")
