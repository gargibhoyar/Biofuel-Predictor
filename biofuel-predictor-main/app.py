import streamlit as st
import pandas as pd
import joblib
import json
import os
import numpy as np
import plotly.express as px

# st.set_page_config(page_title="🌱 Biofuel Predictor", layout="wide")

# st.title("🌱 Biofuel Energy Potential Predictor")
# st.write("This app predicts **biofuel energy (GJ)** from agricultural waste using a trained model.")
# st.set_page_config(
#     page_title="🌱 Biofuel Energy Potential • India",
#     layout="wide",
    
# )
# st.title("🌱 Biofuel Energy Potential Predictor")
st.set_page_config(
    page_title="🌱 Biofuel Energy Potential • India",
    layout="wide",
)


# --- Sidebar ---
st.sidebar.title("🌱 Project Info")

st.sidebar.markdown(
    """
    ### 📌 Project Summary  
    This app predicts **biofuel energy potential** from agricultural waste 
    using machine learning.  
    It supports single prediction, batch uploads, and visualization through 
    charts and an interactive India map.  

    

    ---
    <div style="text-align:left; font-size:15px; margin-top:10px;">
        🚀 <b>Skill4Future 2025</b>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- Load Model ----------------
MODEL_PATH = "saved_models/biofuel_model.joblib"
META_PATH = "saved_models/metadata.json"
DATASET_PATH = "crop_production.csv"
GEOJSON_PATH = "india_states.geojson"
import joblib
import json
import pandas as pd
import os

# Load model
pipe = joblib.load(MODEL_PATH)

# Load metadata
with open(META_PATH, "r") as f:
    meta = json.load(f)

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Load GeoJSON
import json
with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    geojson = json.load(f)

# -------------------------
# Helpers
# -------------------------
@st.cache_resource
def _load_joblib_from_path(path: str):
    return joblib.load(path)

@st.cache_data
def _load_json_from_path(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def _read_csv(path_or_file) -> pd.DataFrame:
    return pd.read_csv(path_or_file)


if not os.path.exists(MODEL_PATH):
    st.error("❌ No trained model found. Please run the Jupyter Notebook to train and save the model first.")
    st.stop()



# -------------------------
# Load dataset for dropdowns
# -------------------------
states_list, districts_map, crops_list, seasons_list = [], {}, [], []
df_ref = None

if os.path.exists(DATASET_PATH):
    try:
        df_ref = _read_csv(DATASET_PATH)
        df_ref.columns = [c.strip() for c in df_ref.columns]

        s_col = "State_Name"
        d_col = "District_Name"
        crop_col = "Crop"
        season_col = "Season"

        if s_col in df_ref:
            states_list = sorted(df_ref[s_col].dropna().astype(str).unique())
        if d_col in df_ref and s_col in df_ref:
            grouped = df_ref[[s_col, d_col]].dropna().astype(str).drop_duplicates().groupby(s_col)
            for stname, sub in grouped:
                districts_map[stname] = sorted(sub[d_col].unique())
        if crop_col in df_ref:
            crops_list = sorted(df_ref[crop_col].dropna().astype(str).unique())
        if season_col in df_ref:
            seasons_list = sorted(df_ref[season_col].dropna().astype(str).unique())
    except Exception as e:
        st.error(f"Error reading dataset: {e}")



# Fallbacks
if not states_list:
    states_list = ['Andaman and Nicobar Islands', 'Andhra Pradesh',
       'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh',
       'Chhattisgarh', 'Dadra and Nagar Haveli', 'Goa', 'Gujarat',
       'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir ', 'Jharkhand',
       'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
       'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry',
       'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana ',
       'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
if not crops_list:
    crops_list = ["Sugarcane","Rice","Wheat","Maize",'Banana', 'Cashewnut',
       'Coconut ', 'Dry ginger', 'Sugarcane', 'Sweet potato', 'Tapioca',
       'Black pepper', 'Dry chillies', 'other oilseeds', 'Turmeric','Cotton(lint)', 'Horse-gram',
       'Jowar', 'Korra', 'Ragi', 'Tobacco', 'Gram', 'Wheat', 'Masoor',
       'Sesamum', 'Linseed', 'Safflower', 'Onion', 'other misc. pulses',
       'Samai', 'Small millets','Barley']
if not seasons_list:
    seasons_list = ["Kharif","Rabi","Whole Year","Rabi","Autumn"]
if not districts_map:
    district_list = ['NICOBARS', 'NORTH AND MIDDLE ANDAMAN', 'SOUTH ANDAMANS',
       'ANANTAPUR', 'CHITTOOR', 'EAST GODAVARI', 'GUNTUR', 'KADAPA',
       'KRISHNA', 'KURNOOL', 'PRAKASAM', 'SPSR NELLORE', 'SRIKAKULAM',
       'VISAKHAPATANAM', 'VIZIANAGARAM', 'WEST GODAVARI', 'ANJAW',
       'CHANGLANG', 'DIBANG VALLEY', 'EAST KAMENG', 'EAST SIANG',
       'KURUNG KUMEY', 'LOHIT', 'LONGDING', 'LOWER DIBANG VALLEY',
       'LOWER SUBANSIRI', 'NAMSAI', 'PAPUM PARE', 'TAWANG', 'TIRAP',
       'UPPER SIANG', 'UPPER SUBANSIRI', 'WEST KAMENG', 'WEST SIANG',
       'BAKSA', 'BARPETA', 'BONGAIGAON', 'CACHAR', 'CHIRANG', 'DARRANG',
       'DHEMAJI', 'DHUBRI', 'DIBRUGARH', 'DIMA HASAO', 'GOALPARA',
       'GOLAGHAT', 'HAILAKANDI', 'JORHAT', 'KAMRUP', 'KAMRUP METRO',
       'KARBI ANGLONG', 'KARIMGANJ', 'KOKRAJHAR', 'LAKHIMPUR', 'MARIGAON',
       'NAGAON', 'NALBARI', 'SIVASAGAR', 'SONITPUR', 'TINSUKIA',
       'UDALGURI', 'ARARIA', 'ARWAL', 'AURANGABAD', 'BANKA', 'BEGUSARAI',
       'BHAGALPUR', 'BHOJPUR', 'BUXAR', 'DARBHANGA', 'GAYA', 'GOPALGANJ',
       'JAMUI', 'JEHANABAD', 'KAIMUR (BHABUA)', 'KATIHAR', 'KHAGARIA',
       'KISHANGANJ', 'LAKHISARAI', 'MADHEPURA', 'MADHUBANI', 'MUNGER',
       'MUZAFFARPUR', 'NALANDA', 'NAWADA', 'PASHCHIM CHAMPARAN', 'PATNA',
       'PURBI CHAMPARAN', 'PURNIA', 'ROHTAS', 'SAHARSA', 'SAMASTIPUR',
       'SARAN', 'SHEIKHPURA', 'SHEOHAR', 'SITAMARHI', 'SIWAN', 'SUPAUL',
       'VAISHALI', 'CHANDIGARH', 'BALOD', 'BALODA BAZAR', 'BALRAMPUR',
       'BASTAR', 'BEMETARA', 'BIJAPUR', 'BILASPUR', 'DANTEWADA',
       'DHAMTARI', 'DURG', 'GARIYABAND', 'JANJGIR-CHAMPA', 'JASHPUR',
       'KABIRDHAM', 'KANKER', 'KONDAGAON', 'KORBA', 'KOREA', 'MAHASAMUND',
       'MUNGELI', 'NARAYANPUR', 'RAIGARH', 'RAIPUR', 'RAJNANDGAON',
       'SUKMA', 'SURAJPUR', 'SURGUJA', 'DADRA AND NAGAR HAVELI',
       'NORTH GOA', 'SOUTH GOA', 'AHMADABAD', 'AMRELI', 'ANAND',
       'BANAS KANTHA', 'BHARUCH', 'BHAVNAGAR', 'DANG', 'DOHAD',
       'GANDHINAGAR', 'JAMNAGAR', 'JUNAGADH', 'KACHCHH', 'KHEDA',
       'MAHESANA', 'NARMADA', 'NAVSARI', 'PANCH MAHALS', 'PATAN',
       'PORBANDAR', 'RAJKOT', 'SABAR KANTHA', 'SURAT', 'SURENDRANAGAR',
       'TAPI', 'VADODARA', 'VALSAD', 'AMBALA', 'BHIWANI', 'FARIDABAD',
       'FATEHABAD', 'GURGAON', 'HISAR', 'JHAJJAR', 'JIND', 'KAITHAL',
       'KARNAL', 'KURUKSHETRA', 'MAHENDRAGARH', 'MEWAT', 'PALWAL',
       'PANCHKULA', 'PANIPAT', 'REWARI', 'ROHTAK', 'SIRSA', 'SONIPAT',
       'YAMUNANAGAR', 'CHAMBA', 'HAMIRPUR', 'KANGRA', 'KINNAUR', 'KULLU',
       'LAHUL AND SPITI', 'MANDI', 'SHIMLA', 'SIRMAUR', 'SOLAN', 'UNA',
       'ANANTNAG', 'BADGAM', 'BANDIPORA', 'BARAMULLA', 'DODA',
       'GANDERBAL', 'JAMMU', 'KARGIL', 'KATHUA', 'KISHTWAR', 'KULGAM',
       'KUPWARA', 'LEH LADAKH', 'POONCH', 'PULWAMA', 'RAJAURI', 'RAMBAN',
       'REASI', 'SAMBA', 'SHOPIAN', 'SRINAGAR', 'UDHAMPUR', 'BOKARO',
       'CHATRA', 'DEOGHAR', 'DHANBAD', 'DUMKA', 'EAST SINGHBUM', 'GARHWA',
       'GIRIDIH', 'GODDA', 'GUMLA', 'HAZARIBAGH', 'JAMTARA', 'KHUNTI',
       'KODERMA', 'LATEHAR', 'LOHARDAGA', 'PAKUR', 'PALAMU', 'RAMGARH',
       'RANCHI', 'SAHEBGANJ', 'SARAIKELA KHARSAWAN', 'SIMDEGA',
       'WEST SINGHBHUM', 'BAGALKOT', 'BANGALORE RURAL', 'BELGAUM',
       'BELLARY', 'BENGALURU URBAN', 'BIDAR', 'CHAMARAJANAGAR',
       'CHIKBALLAPUR', 'CHIKMAGALUR', 'CHITRADURGA', 'DAKSHIN KANNAD',
       'DAVANGERE', 'DHARWAD', 'GADAG', 'GULBARGA', 'HASSAN', 'HAVERI',
       'KODAGU', 'KOLAR', 'KOPPAL', 'MANDYA', 'MYSORE', 'RAICHUR',
       'RAMANAGARA', 'SHIMOGA', 'TUMKUR', 'UDUPI', 'UTTAR KANNAD',
       'YADGIR', 'ALAPPUZHA', 'ERNAKULAM', 'IDUKKI', 'KANNUR',
       'KASARAGOD', 'KOLLAM', 'KOTTAYAM', 'KOZHIKODE', 'MALAPPURAM',
       'PALAKKAD', 'PATHANAMTHITTA', 'THIRUVANANTHAPURAM', 'THRISSUR',
       'WAYANAD', 'AGAR MALWA', 'ALIRAJPUR', 'ANUPPUR', 'ASHOKNAGAR',
       'BALAGHAT', 'BARWANI', 'BETUL', 'BHIND', 'BHOPAL', 'BURHANPUR',
       'CHHATARPUR', 'CHHINDWARA', 'DAMOH', 'DATIA', 'DEWAS', 'DHAR',
       'DINDORI', 'GUNA', 'GWALIOR', 'HARDA', 'HOSHANGABAD', 'INDORE',
       'JABALPUR', 'JHABUA', 'KATNI', 'KHANDWA', 'KHARGONE', 'MANDLA',
       'MANDSAUR', 'MORENA', 'NARSINGHPUR', 'NEEMUCH', 'PANNA', 'RAISEN',
       'RAJGARH', 'RATLAM', 'REWA', 'SAGAR', 'SATNA', 'SEHORE', 'SEONI',
       'SHAHDOL', 'SHAJAPUR', 'SHEOPUR', 'SHIVPURI', 'SIDHI', 'SINGRAULI',
       'TIKAMGARH', 'UJJAIN', 'UMARIA', 'VIDISHA', 'AHMEDNAGAR', 'AKOLA',
       'AMRAVATI', 'BEED', 'BHANDARA', 'BULDHANA', 'CHANDRAPUR', 'DHULE',
       'GADCHIROLI', 'GONDIA', 'HINGOLI', 'JALGAON', 'JALNA', 'KOLHAPUR',
       'LATUR', 'MUMBAI', 'NAGPUR', 'NANDED', 'NANDURBAR', 'NASHIK',
       'OSMANABAD', 'PALGHAR', 'PARBHANI', 'PUNE', 'RAIGAD', 'RATNAGIRI',
       'SANGLI', 'SATARA', 'SINDHUDURG', 'SOLAPUR', 'THANE', 'WARDHA',
       'WASHIM', 'YAVATMAL', 'BISHNUPUR', 'CHANDEL', 'CHURACHANDPUR',
       'IMPHAL EAST', 'IMPHAL WEST', 'SENAPATI', 'TAMENGLONG', 'THOUBAL',
       'UKHRUL', 'EAST GARO HILLS', 'EAST JAINTIA HILLS',
       'EAST KHASI HILLS', 'NORTH GARO HILLS', 'RI BHOI',
       'SOUTH GARO HILLS', 'SOUTH WEST GARO HILLS',
       'SOUTH WEST KHASI HILLS', 'WEST GARO HILLS', 'WEST JAINTIA HILLS',
       'WEST KHASI HILLS', 'AIZAWL', 'CHAMPHAI', 'KOLASIB', 'LAWNGTLAI',
       'LUNGLEI', 'MAMIT', 'SAIHA', 'SERCHHIP', 'DIMAPUR', 'KIPHIRE',
       'KOHIMA', 'LONGLENG', 'MOKOKCHUNG', 'MON', 'PEREN', 'PHEK',
       'TUENSANG', 'WOKHA', 'ZUNHEBOTO', 'ANUGUL', 'BALANGIR',
       'BALESHWAR', 'BARGARH', 'BHADRAK', 'BOUDH', 'CUTTACK', 'DEOGARH',
       'DHENKANAL', 'GAJAPATI', 'GANJAM', 'JAGATSINGHAPUR', 'JAJAPUR',
       'JHARSUGUDA', 'KALAHANDI', 'KANDHAMAL', 'KENDRAPARA', 'KENDUJHAR',
       'KHORDHA', 'KORAPUT', 'MALKANGIRI', 'MAYURBHANJ', 'NABARANGPUR',
       'NAYAGARH', 'NUAPADA', 'PURI', 'RAYAGADA', 'SAMBALPUR', 'SONEPUR',
       'SUNDARGARH', 'KARAIKAL', 'MAHE', 'PONDICHERRY', 'YANAM',
       'AMRITSAR', 'BARNALA', 'BATHINDA', 'FARIDKOT', 'FATEHGARH SAHIB',
       'FAZILKA', 'FIROZEPUR', 'GURDASPUR', 'HOSHIARPUR', 'JALANDHAR',
       'KAPURTHALA', 'LUDHIANA', 'MANSA', 'MOGA', 'MUKTSAR', 'NAWANSHAHR',
       'PATHANKOT', 'PATIALA', 'RUPNAGAR', 'S.A.S NAGAR', 'SANGRUR',
       'TARN TARAN', 'AJMER', 'ALWAR', 'BANSWARA', 'BARAN', 'BARMER',
       'BHARATPUR', 'BHILWARA', 'BIKANER', 'BUNDI', 'CHITTORGARH',
       'CHURU', 'DAUSA', 'DHOLPUR', 'DUNGARPUR', 'GANGANAGAR',
       'HANUMANGARH', 'JAIPUR', 'JAISALMER', 'JALORE', 'JHALAWAR',
       'JHUNJHUNU', 'JODHPUR', 'KARAULI', 'KOTA', 'NAGAUR', 'PALI',
       'PRATAPGARH', 'RAJSAMAND', 'SAWAI MADHOPUR', 'SIKAR', 'SIROHI',
       'TONK', 'UDAIPUR', 'EAST DISTRICT', 'NORTH DISTRICT',
       'SOUTH DISTRICT', 'WEST DISTRICT', 'ARIYALUR', 'COIMBATORE',
       'CUDDALORE', 'DHARMAPURI', 'DINDIGUL', 'ERODE', 'KANCHIPURAM',
       'KANNIYAKUMARI', 'KARUR', 'KRISHNAGIRI', 'MADURAI', 'NAGAPATTINAM',
       'NAMAKKAL', 'PERAMBALUR', 'PUDUKKOTTAI', 'RAMANATHAPURAM', 'SALEM',
       'SIVAGANGA', 'THANJAVUR', 'THE NILGIRIS', 'THENI', 'THIRUVALLUR',
       'THIRUVARUR', 'TIRUCHIRAPPALLI', 'TIRUNELVELI', 'TIRUPPUR',
       'TIRUVANNAMALAI', 'TUTICORIN', 'VELLORE', 'VILLUPURAM',
       'VIRUDHUNAGAR', 'ADILABAD', 'HYDERABAD', 'KARIMNAGAR', 'KHAMMAM',
       'MAHBUBNAGAR', 'MEDAK', 'NALGONDA', 'NIZAMABAD', 'RANGAREDDI',
       'WARANGAL', 'DHALAI', 'GOMATI', 'KHOWAI', 'NORTH TRIPURA',
       'SEPAHIJALA', 'SOUTH TRIPURA', 'UNAKOTI', 'WEST TRIPURA', 'AGRA',
       'ALIGARH', 'ALLAHABAD', 'AMBEDKAR NAGAR', 'AMETHI', 'AMROHA',
       'AURAIYA', 'AZAMGARH', 'BAGHPAT', 'BAHRAICH', 'BALLIA', 'BANDA',
       'BARABANKI', 'BAREILLY', 'BASTI', 'BIJNOR', 'BUDAUN',
       'BULANDSHAHR', 'CHANDAULI', 'CHITRAKOOT', 'DEORIA', 'ETAH',
       'ETAWAH', 'FAIZABAD', 'FARRUKHABAD', 'FATEHPUR', 'FIROZABAD',
       'GAUTAM BUDDHA NAGAR', 'GHAZIABAD', 'GHAZIPUR', 'GONDA',
       'GORAKHPUR', 'HAPUR', 'HARDOI', 'HATHRAS', 'JALAUN', 'JAUNPUR',
       'JHANSI', 'KANNAUJ', 'KANPUR DEHAT', 'KANPUR NAGAR', 'KASGANJ',
       'KAUSHAMBI', 'KHERI', 'KUSHI NAGAR', 'LALITPUR', 'LUCKNOW',
       'MAHARAJGANJ', 'MAHOBA', 'MAINPURI', 'MATHURA', 'MAU', 'MEERUT',
       'MIRZAPUR', 'MORADABAD', 'MUZAFFARNAGAR', 'PILIBHIT', 'RAE BARELI',
       'RAMPUR', 'SAHARANPUR', 'SAMBHAL', 'SANT KABEER NAGAR',
       'SANT RAVIDAS NAGAR', 'SHAHJAHANPUR', 'SHAMLI', 'SHRAVASTI',
       'SIDDHARTH NAGAR', 'SITAPUR', 'SONBHADRA', 'SULTANPUR', 'UNNAO',
       'VARANASI', 'ALMORA', 'BAGESHWAR', 'CHAMOLI', 'CHAMPAWAT',
       'DEHRADUN', 'HARIDWAR', 'NAINITAL', 'PAURI GARHWAL', 'PITHORAGARH',
       'RUDRA PRAYAG', 'TEHRI GARHWAL', 'UDAM SINGH NAGAR', 'UTTAR KASHI',
       '24 PARAGANAS NORTH', '24 PARAGANAS SOUTH', 'BANKURA', 'BARDHAMAN',
       'BIRBHUM', 'COOCHBEHAR', 'DARJEELING', 'DINAJPUR DAKSHIN',
       'DINAJPUR UTTAR', 'HOOGHLY', 'HOWRAH', 'JALPAIGURI', 'MALDAH',
       'MEDINIPUR EAST', 'MEDINIPUR WEST', 'MURSHIDABAD', 'NADIA',
       'PURULIA']
    
# Populate districts_map with fallback for all states
if not districts_map and 'district_list' in locals():
    for state_name in states_list:
        districts_map[state_name] = district_list   




# # -------------------------
# # Load model + metadata
# # -------------------------
# model_obj, meta_obj = None, None
# if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
#     try:
#         model_obj = _load_joblib_from_path(MODEL_PATH)
#         meta_obj = _load_json_from_path(META_PATH)
#     except Exception as e:
#         st.error(f"Failed to load model/metadata: {e}")

# if model_obj is None or meta_obj is None:
#     st.stop()

# cat_cols = meta_obj.get("categorical_features", [])
# num_cols = meta_obj.get("numerical_features", [])

# Load model + metadata
# -------------------------
model_obj, meta_obj = None, None

try:
    if os.path.exists(MODEL_PATH):
        model_obj = joblib.load(MODEL_PATH)
    else:
        st.error(f"Model file not found at {MODEL_PATH}")

    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            meta_obj = json.load(f)
    else:
        st.error(f"Metadata file not found at {META_PATH}")

except Exception as e:
    st.error(f"Failed to load model/metadata: {e}")
    st.stop()

# Stop the app if either is None
if model_obj is None or meta_obj is None:
    st.stop()

# -------------------------
# Extract column info
# -------------------------
cat_cols = meta_obj.get("categorical_features", [])
num_cols = meta_obj.get("numerical_features", [])
target_col = meta_obj.get("target", "Energy_GJ")  # fallback to default target

# st.success(f"Model and metadata loaded successfully!")
# st.write(f"Categorical columns: {cat_cols}")
# st.write(f"Numerical columns: {num_cols}")

# State centroids for bubble map
# -------------------------
STATE_CENTROIDS = {
    "Maharashtra": (19.7515, 75.7139),
    "Punjab": (31.1471, 75.3412),
    "Kerala": (10.8505, 76.2711),
    "Tamil Nadu": (11.1271, 78.6569),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Gujarat": (22.2587, 71.1924),
    "West Bengal": (22.9868, 87.8550),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Rajasthan": (27.0238, 74.2179),
    "Bihar": (25.0961, 85.3131),
    # ... add more if needed
}

# -------------------------
# UI
# -------------------------
st.title("🌱 Biofuel Energy Potential Predictor — India")




# # Load model + metadata
# model = joblib.load(MODEL_PATH)
# with open(META_PATH, "r") as f:
#     meta = json.load(f)

# cat_cols = meta["categorical_features"]
# num_cols = meta["numerical_features"]

# # ---------------- Single Prediction ----------------
# st.header("🔹 Single Prediction")

# col1, col2, col3 = st.columns(3)

# with col1:
#     state = st.text_input("State Name", "Maharashtra")
#     district = st.text_input("District Name", "Pune")

# with col2:
#     season = st.text_input("Season", "Kharif")
#     crop = st.text_input("Crop", "Sugarcane")

# with col3:
#     area = st.number_input("Area (hectares)", min_value=0.0, value=1000.0)
#     production = st.number_input("Production (tonnes)", min_value=0.0, value=5000.0)

# if st.button("Predict Biofuel Energy"):
#     inp = pd.DataFrame([{
#         "State_Name": state,
#         "District_Name": district,
#         "Season": season,
#         "Crop": crop,
#         "Area": area,
#         "Production": production
#     }])
#     pred = model.predict(inp)[0]
#     st.success(f"Estimated Biofuel Energy: **{pred:,.2f} GJ**")

    
#     # 2. Ensure columns match training
#     inp = inp[cat_cols + num_cols]  # use metadata columns
    
#     # 3. Convert categorical columns to category type if needed
#     for col in cat_cols:
#         inp[col] = inp[col].astype(str)
    
#     # 4. Predict
#     try:
#         pred = model.predict(inp)[0]
#         st.success(f"Estimated Biofuel Energy: **{pred:,.2f} GJ**")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")


# -------------------------
# UI
# -------------------------
# st.title("🌱 Biofuel Energy Potential Predictor — India")

# --- Single Prediction ---
st.subheader("🔹 Single Prediction")

c1, c2, c3 = st.columns([1,1,1])

with c1:
    state = st.selectbox(
        "State",
        options=states_list,
        index=(states_list.index("Maharashtra") if "Maharashtra" in states_list else 0)
    )
    district_opts = districts_map.get(state, [])
    if district_opts:
        district = st.selectbox("District", options=district_opts)
    else:
        district = st.text_input("District", value="Unknown")

with c2:
    season = st.selectbox("Season", seasons_list, index=0)
    crop = st.selectbox("Crop", crops_list, index=0)

with c3:
    area = st.number_input("Area (hectares)", min_value=0.0, value=1000.0, step=10.0)
    production = st.number_input("Production (tonnes)", min_value=0.0, value=5000.0, step=10.0)

if st.button("🔮 Predict Biofuel Energy"):
    try:
        # Build input DataFrame
        inp = pd.DataFrame([{
            "State_Name": state,
            "District_Name": district,
            "Season": season,
            "Crop": crop,
            "Area": area,
            "Production": production
        }])

        # Pass the full DataFrame to the pipeline
        pred = float(model_obj.predict(inp)[0])

        # Display results
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Estimated Biofuel Energy (GJ)", f"{pred:,.2f}")
        st.write(f"Energy per tonne: **{pred/(production or 1):.3f} GJ/t**")
        st.write(f"Energy per hectare: **{pred/(area or 1):.3f} GJ/ha**")
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# # ---------------- Batch Prediction ----------------
# st.header("📂 Batch Prediction from CSV")

# st.write("Upload a CSV file with columns: `State_Name, District_Name, Season, Crop, Area, Production`")

# uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     try:
#         preds = model.predict(df[cat_cols + num_cols])
#         df["Predicted_Energy_GJ"] = preds
#         st.write("✅ Predictions complete. Preview below:")
#         st.dataframe(df.head(20), use_container_width=True)

#         # Download button
#         csv_out = df.to_csv(index=False).encode("utf-8")
#         st.download_button("Download Predictions CSV", data=csv_out, file_name="predictions.csv", mime="text/csv")

#     except Exception as e:
#         st.error(f"Error: {e}")
# ---------------- Batch Prediction ----------------
st.header("📑 Biofuel Potential Analysis (CSV Upload)")
st.markdown("""
📌 **Upload a CSV file with these columns:**  
- `State_Name` → e.g., Maharashtra, Punjab  
- `Season` → e.g., Kharif, Rabi, Whole Year  
- `Crop` → e.g., Sugarcane, Rice, Wheat  
- `Area` → Cultivated area in hectares  
- `Production` → Crop production in tonnes  

👉 If you don’t have a file, use the provided **`crop_production.csv`** to see sample outputs.
""")
# # st.write("Upload a CSV file with columns: `State_Name, District_Name, Season, Crop, Area, Production`")

# uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     try:
#         # Pass the full DataFrame to the pipeline
#         preds = model_obj.predict(df)

#         # Add predictions
#         df["Predicted_Energy_GJ"] = preds
#         st.write("✅ Batch predictions complete")
#         # st.dataframe(df.head(20), use_container_width=True)

#         # # Download button
#         # csv_out = df.to_csv(index=False).encode("utf-8")
#         # st.download_button("Download Predictions CSV", data=csv_out, file_name="predictions.csv", mime="text/csv")

#     except Exception as e:
#         st.error(f"Error: {e}")


# ---------------- Batch Prediction & Map View ----------------

up = st.file_uploader("Choose a CSV file", type=["csv"])
batch_df = None
# if uploaded_file is not None:
#     batch_df = pd.read_csv(uploaded_file)
#     try:
#         # Predict using the full pipeline
#         batch_df["Predicted_Energy_GJ"] = model_obj.predict(batch_df)

#         st.success("✅ Predictions complete. Preview below:")
#         # st.dataframe(batch_df.head(20), use_container_width=True)

#     #    # Download button
#     #     csv_out = batch_df.to_csv(index=False).encode("utf-8")
#     #     st.download_button(
#     #         "Down load Predictions CSV",
#     #         data=csv_out,
#     #         file_name="predictions.csv",
#     #         mime="text/csv"
if up is not None:
     try:
        batch_df = pd.read_csv(up)

         # Fill NaNs
        for col in num_cols:
            if col in batch_df:
                 batch_df[col] = batch_df[col].fillna(batch_df[col].median())
        for col in cat_cols:
            if col in batch_df:
                 batch_df[col] = batch_df[col].fillna("Unknown")

        st.success("✅ Batch predictions complete")
             # st.dataframe(batch_df.head())

     except Exception as e:
         st.error(f"Batch prediction failed: {e}")

  

# --- CO₂ Reduction & Real-World Impact ---
if batch_df is not None and "State_Name" in batch_df:

    st.subheader("🌍 Real-World Impact of Dataset")

    # Choose column to use for calculation (Production if exists, else count rows)
    if "Production" in batch_df.columns:
        total_potential = batch_df["Production"].sum()
        unit_label = "tonnes of production"
    else:
        total_potential = len(batch_df)  # just count rows
        unit_label = "records"

    # Conversion factors (example: 1 unit ≈ 0.056 tonnes CO₂ avoided)
    co2_saved = total_potential * 0.056
    households_powered = total_potential / 50_000  # example factor

    # Styled highlight box
    st.markdown(
        f"""
        <div style="
            background-color:#ffffff10;
            border: 1px solid #4CAF50;
            padding: 18px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            font-size:16px;
            line-height:1.8;
            color:white;
        ">
        ♻️ Sustainability Outcomes:<br><br>
        ⚡ <b style="color:#1E90FF;">{total_potential:,.0f}</b> {unit_label} in dataset.<br>
        🌍 This would avoid <b style="color:#16a34a;">{co2_saved:,.0f} tonnes</b> of CO₂ emissions per year.<br>
        🏠 Enough to power <b style="color:#FF8C00;">{households_powered:,.0f} households</b> for one year.
        </div>
        """,
        unsafe_allow_html=True
    )



# --- Visualization & Insights ---
st.subheader("📊 Visualization & Insights")

if batch_df is not None and "State_Name" in batch_df:

    with st.expander("🔍 Filters", expanded=True):
        sel_crop = st.multiselect("Filter by Crop", options=sorted(batch_df["Crop"].unique()) if "Crop" in batch_df else [])
        sel_state = st.multiselect("Filter by State", options=sorted(batch_df["State_Name"].unique()))
        sel_season = st.multiselect("Filter by Season", options=sorted(batch_df["Season"].unique()) if "Season" in batch_df else [])

    # Copy and filter dataframe
    viz_df = batch_df.copy()
    if sel_crop and "Crop" in viz_df: 
        viz_df = viz_df[viz_df["Crop"].isin(sel_crop)]
    if sel_state:
        viz_df = viz_df[viz_df["State_Name"].isin(sel_state)]
    if sel_season and "Season" in viz_df:
        viz_df = viz_df[viz_df["Season"].isin(sel_season)]

    # Determine column to use for visualization
    if "Production" in viz_df.columns:
        value_col = "Production"
    else:
        viz_df["_value"] = 1
        value_col = "_value"

    mode = st.radio("View Mode", ["By State", "By Crop", "By Year"], horizontal=True)

    if mode == "By State":
        agg = viz_df.groupby("State_Name", as_index=False)[value_col].sum()
        fig = px.pie(agg, names="State_Name", values=value_col,
                     title="Share of Biofuel by State")
        st.plotly_chart(fig, use_container_width=True)

    elif mode == "By Crop" and "Crop" in viz_df:
        agg = viz_df.groupby("Crop", as_index=False)[value_col].sum()
        fig = px.pie(agg, names="Crop", values=value_col,
                     title="Share of Biofuel by Crop")
        st.plotly_chart(fig, use_container_width=True)

    elif mode == "By Year" and "Crop_Year" in viz_df:
        agg = viz_df.groupby("Crop_Year", as_index=False)[value_col].sum()
        fig = px.pie(agg, names="Crop_Year", values=value_col,
                     title="Share of Biofuel by Year")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Year-wise trend not available (no 'Crop_Year' column in dataset).")

# # --- Leaderboard View ---
# st.subheader("📊 Top Contributors to Biofuel Potential")

# if batch_df is not None and "State_Name" in batch_df:

#     tab1, tab2, tab3 = st.tabs(["🌍 Top States", "🌾 Top Crops", "🌤️ Top Seasons"])

#     # Choose the column to aggregate (Production if exists, else count)
#     if "Production" in batch_df.columns:
#         value_col = "Production"
#     else:
#         value_col = None  # will count rows

#     with tab1:
#         if value_col:
#             agg = batch_df.groupby("State_Name", as_index=False)[value_col].sum()
#         else:
#             agg = batch_df.groupby("State_Name", as_index=False).size().reset_index(name="Count")
#             value_col = "Count"

#         top5 = agg.sort_values(value_col, ascending=False).head(5)
#         fig = px.bar(top5, x="State_Name", y=value_col,
#                      title="Top 5 States by Biofuel Potential",
#                      text=value_col,
#                      color=value_col, color_continuous_scale="Viridis")
#         fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
#         st.plotly_chart(fig, use_container_width=True)

#     with tab2:
#         if "Crop" in batch_df.columns:
#             if value_col == "Count":
#                 agg = batch_df.groupby("Crop", as_index=False).size().reset_index(name="Count")
#             else:
#                 agg = batch_df.groupby("Crop", as_index=False)[value_col].sum()
#             top5 = agg.sort_values(value_col, ascending=False).head(5)
#             fig = px.bar(top5, x="Crop", y=value_col,
#                          title="Top 5 Crops by Biofuel Potential",
#                          text=value_col,
#                          color=value_col, color_continuous_scale="Blues")
#             fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("No Crop column in dataset.")

#     with tab3:
#         if "Season" in batch_df.columns:
#             if value_col == "Count":
#                 agg = batch_df.groupby("Season", as_index=False).size().reset_index(name="Count")
#             else:
#                 agg = batch_df.groupby("Season", as_index=False)[value_col].sum()
#             top5 = agg.sort_values(value_col, ascending=False).head(5)
#             fig = px.bar(top5, x="Season", y=value_col,
#                          title="Top 5 Seasons by Biofuel Potential",
#                          text=value_col,
#                          color=value_col, color_continuous_scale="Greens")
#             fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("No Season column in dataset.")

# else:
#     st.info("📂 Upload a batch CSV to see leaderboard rankings.")

# --- Leaderboard View ---
st.subheader("🥇 Top Contributors to Biofuel Potential")

if batch_df is not None and "State_Name" in batch_df:

    tab1, tab2, tab3 = st.tabs(["🌍 Top States", "🌾 Top Crops", "🌤️ Top Seasons"])

    # Choose column to aggregate (Production if exists, else count)
    if "Production" in batch_df.columns:
        value_col = "Production"
        batch_df["_value"] = batch_df["Production"]
    else:
        value_col = "_value"
        batch_df["_value"] = 1  # just count rows

    with tab1:
        agg = batch_df.groupby("State_Name", as_index=False)[value_col].sum() if "Production" in batch_df.columns else batch_df.groupby("State_Name", as_index=False)["_value"].sum()
        top5 = agg.sort_values(value_col, ascending=False).head(5)
        fig = px.bar(top5, x="State_Name", y=value_col,
                     title="Top 5 States by Biofuel Potential",
                     text=value_col,
                     color=value_col, color_continuous_scale="Viridis")
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "Crop" in batch_df.columns:
            agg = batch_df.groupby("Crop", as_index=False)[value_col].sum()
            top5 = agg.sort_values(value_col, ascending=False).head(5)
            fig = px.bar(top5, x="Crop", y=value_col,
                         title="Top 5 Crops by Biofuel Potential",
                         text=value_col,
                         color=value_col, color_continuous_scale="Blues")
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Crop column in dataset.")

    with tab3:
        if "Season" in batch_df.columns:
            agg = batch_df.groupby("Season", as_index=False)[value_col].sum()
            top5 = agg.sort_values(value_col, ascending=False).head(5)
            fig = px.bar(top5, x="Season", y=value_col,
                         title="Top 5 Seasons by Biofuel Potential",
                         text=value_col,
                         color=value_col, color_continuous_scale="Greens")
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Season column in dataset.")

# else:
#     st.info("📂 Upload a batch CSV to see leaderboard rankings.")

# --- Time-Series Forecasting ---
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px

if batch_df is not None:
    if "Crop_Year" in batch_df.columns:
        st.subheader("📈 Forecast: Biofuel Potential till 2030")
        st.caption("Future biofuel potential from 2010 onwards (based on uploaded CSV).")

        # Determine column to aggregate (Production if exists, else count)
        if "Production" in batch_df.columns:
            yearly = batch_df.groupby("Crop_Year", as_index=False)["Production"].sum()
            value_col = "Production"
        else:
            yearly = batch_df.groupby("Crop_Year", as_index=False).size().reset_index(name="_value")
            value_col = "_value"

        if not yearly.empty:
            X = yearly["Crop_Year"].values.reshape(-1, 1)
            y = yearly[value_col].values

            # Fit regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predict from 2010 to 2030
            future_years = np.arange(2010, 2031).reshape(-1, 1)
            future_preds = model.predict(future_years)

            forecast_df = pd.DataFrame({
                "Year": future_years.flatten(),
                value_col: future_preds
            })

            # Plot with labels on markers
            fig = px.line(
                forecast_df, 
                x="Year", 
                y=value_col,
                markers=True,
                text=value_col
            )

            fig.update_traces(texttemplate="%{text:,.0f}", textposition="top center")
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Biofuel Potential",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                title_text=None
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No yearly data available to fit the forecast model.")
    else:
        st.warning("Required column 'Crop_Year' is missing in the uploaded CSV.")

# --- Smart Recommendations ---
st.subheader("💡 Smart Recommendations: States to Improve")

if batch_df is not None and "State_Name" in batch_df and "Crop" in batch_df:

    # Determine column to use (Production if exists, else count)
    if "Production" in batch_df.columns:
        value_col = "Production"
        batch_df["_value"] = batch_df["Production"]
    else:
        value_col = "_value"
        batch_df["_value"] = 1  # just count rows

    try:
        # Aggregate per state & crop
        rec_df = batch_df.groupby(["State_Name", "Crop"], as_index=False)[value_col].sum()

        # Find lowest crop per state and exclude Kerala
        low_recs = rec_df.loc[rec_df.groupby("State_Name")[value_col].idxmin()]
        low_recs = low_recs[low_recs["State_Name"] != "Kerala"]

        # Top 5 lowest potential states
        low5 = low_recs.sort_values(value_col).head(5)

        st.markdown("**Recommended Action: Focus on improving biofuel production in the following states**")
        for _, row in low5.iterrows():
            st.write(f"- {row['State_Name']}: low potential from {row['Crop']} ({row[value_col]:,.0f}). Consider optimizing crop selection or yield.")

    except Exception as e:
        st.error(f"Could not generate recommendations: {e}")

# else:
#     st.info("Upload data with 'State_Name' and 'Crop' columns to generate recommendations.")



# --- Map View with Labels ---
st.subheader("🗺️ India Overview: Biofuel Potential by State")

if batch_df is not None and "State_Name" in batch_df:

    if os.path.exists(GEOJSON_PATH):
        import json
        import plotly.express as px

        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        # Aggregate by state (sum of Production or count)
        if "Production" in batch_df.columns:
            agg = batch_df.groupby("State_Name", as_index=False)["Production"].sum()
            value_col = "Production"
        else:
            agg = batch_df.groupby("State_Name", as_index=False).size().reset_index(name="Count")
            value_col = "Count"

        agg["State_Name"] = agg["State_Name"].str.strip()

        # Fix mismatched state names
        STATE_NAME_MAP = {
            "Andaman and Nicobar Islands": "Andaman and Nicobar",
            "Jammu and Kashmir": "Jammu and Kashmir",
            "Odisha": "Orissa",
            "Puducherry": "Pondicherry",
            "Dadra and Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
        }
        agg["State_Name"] = agg["State_Name"].replace(STATE_NAME_MAP)

        # Choropleth map
        fig = px.choropleth(
            agg,
            geojson=geojson,
            locations="State_Name",
            featureidkey="properties.name",
            color=value_col,
            hover_name="State_Name",
            hover_data={value_col: ":,.0f"},
            color_continuous_scale="Viridis",
            title="🌱 Overview by State"
        )

        # --- Add state labels with centroids ---
        STATE_CENTROIDS = {
            "Andaman and Nicobar": [11.74, 92.65],
            "Andhra Pradesh": [15.91, 79.74],
            "Arunachal Pradesh": [28.21, 94.73],
            "Assam": [26.20, 92.94],
            "Bihar": [25.10, 85.31],
            "Chandigarh": [30.73, 76.78],
            "Chhattisgarh": [21.28, 81.87],
            "Dadra and Nagar Haveli and Daman and Diu": [20.39, 72.83],
            "Goa": [15.30, 74.12],
            "Gujarat": [22.26, 71.19],
            "Haryana": [29.06, 76.09],
            "Himachal Pradesh": [31.10, 77.17],
            "Jammu and Kashmir": [33.78, 76.58],
            "Jharkhand": [23.61, 85.28],
            "Karnataka": [15.32, 75.71],
            "Kerala": [10.85, 76.27],
            "Madhya Pradesh": [22.97, 78.66],
            "Maharashtra": [19.75, 75.71],
            "Manipur": [24.66, 93.91],
            "Meghalaya": [25.47, 91.37],
            "Mizoram": [23.16, 92.94],
            "Nagaland": [26.16, 94.56],
            "Orissa": [20.95, 85.10],
            "Pondicherry": [11.94, 79.81],
            "Punjab": [31.15, 75.34],
            "Rajasthan": [27.02, 74.22],
            "Sikkim": [27.53, 88.51],
            "Tamil Nadu": [11.13, 78.66],
            "Telangana": [18.11, 79.02],
            "Tripura": [23.94, 91.99],
            "Uttar Pradesh": [26.85, 80.95],
            "Uttarakhand": [30.07, 79.02],
            "West Bengal": [22.99, 87.86],
        }

        vmax = agg[value_col].max()
        for _, row in agg.iterrows():
            state = row["State_Name"]
            if state in STATE_CENTROIDS:
                lat, lon = STATE_CENTROIDS[state]
                value = row[value_col]
                text_color = "black" if value > (0.6 * vmax) else "white"
                fig.add_trace(dict(
                    type="scattergeo",
                    lon=[lon],
                    lat=[lat],
                    text=[f"{value/1e6:,.1f}M" if value>1e6 else f"{value:,.0f}"],
                    mode="text",
                    showlegend=False,
                    textfont=dict(color=text_color, size=11, family="Arial Bold")
                ))

        # Layout
        fig.update_layout(
            height=500,
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar=dict(
                title=value_col,
                tickformat=",.0f",
                lenmode="pixels", len=200
            ),
        )

        fig.update_geos(
            visible=False,
            showcoastlines=False,
            showland=True,
            landcolor="rgba(0,0,0,0.1)",
            lataxis_range=[6, 38],
            lonaxis_range=[68, 98],
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("❌ india_states.geojson not found. Please put it in the same folder as appone.py.")



# --- Footer ---
st.markdown(
    """
    <hr style="margin-top: 2rem; margin-bottom: 1rem;">
    <div style="text-align: center; color: grey; font-size: 14px;">
        🌱 Biofuel Energy Potential Predictor 
        (2025 Skill4Future Project)
    </div>
    """,
    unsafe_allow_html=True
)
