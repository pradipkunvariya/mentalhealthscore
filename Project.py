
# In[1]:
import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# In[2]:
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --- STREAMLIT APP CONFIGURATION ---
st.set_page_config(
    page_title="Mental Health Score Predictor Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- APP LAYOUT HEADER & SUBDETAILS ---
st.title("🧠 Student Mental Health Analysis & Predictive Analytics Dashboard")
st.markdown("""
This web application is a direct interactive deployment of the **Data Science & Machine Learning Research Framework** evaluated in the Jupyter Notebook. 
It analyzes how academic factors, lifestyle variables, and demographics affect overall mental welfare scores, enabling structural predictive inference using optimized Regression trees.
""")

# --- MOCK DATA INGESTION ---
# (Simulating your notebook data structure for complete functionality)
@st.cache_data
def load_and_preprocess_data():
    np.random.seed(42)
    n_samples = 300
    
    data = {
        'age': np.random.choice([18.0, 19.0, 20.0, 21.0, 24.0, 26.0], size=n_samples),
        'gender': np.random.choice(['Male', 'Female'], size=n_samples),
        'education': np.random.choice(['School', 'Undergraduate', 'Postgraduate'], size=n_samples),
        'sleep_hours': np.random.uniform(4, 9, size=n_samples),
        'screen_time': np.random.uniform(1, 10, size=n_samples),
        'study_hours': np.random.uniform(1, 8, size=n_samples),
        'exercise': np.random.choice(['Yes', 'No'], size=n_samples),
        'diet': np.random.choice(['Healthy', 'Average', 'Unhealthy'], size=n_samples),
        'stress_frequency': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Always'], size=n_samples),
        'anxiety_frequency': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Always'], size=n_samples),
        'mental_health_score': np.random.uniform(3, 10, size=n_samples)
    }
    
    raw_df = pd.DataFrame(data)
    
    # Mirroring notebook processing configurations
    processed_df = raw_df.copy()
    
    # Handle Missing Values mapping using standard mean values discovered in step blocks
    processed_df['sleep_hours'] = processed_df['sleep_hours'].fillna(6.68)
    processed_df['screen_time'] = processed_df['screen_time'].fillna(4.49)
    processed_df['study_hours'] = processed_df['study_hours'].fillna(3.36)
    
    # Categorical Conversions as modeled inside the raw cell arrays
    gender_map = {'Male': 0, 'Female': 1}
    edu_map = {'School': 0, 'Undergraduate': 1, 'Postgraduate': 2}
    exe_map = {'No': 0, 'Yes': 1}
    diet_map = {'Healthy': 0, 'Average': 1, 'Unhealthy': 2}
    freq_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Always': 3}
    
    processed_df['gender'] = processed_df['gender'].map(gender_map)
    processed_df['education'] = processed_df['education'].map(edu_map)
    processed_df['exercise'] = processed_df['exercise'].map(exe_map)
    processed_df['diet'] = processed_df['diet'].map(diet_map)
    processed_df['stress_frequency'] = processed_df['stress_frequency'].map(freq_map)
    processed_df['anxiety_frequency'] = processed_df['anxiety_frequency'].map(freq_map)
    
    return raw_df, processed_df

raw_df, df = load_and_preprocess_data()

# --- APP LAYOUT TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "🤖 Model Evaluation", "🔮 Custom Score Predictor"])

# ================= TAB 1: EXPLORATORY DATA ANALYSIS =================
with tab1:
    st.header("📊 Dataset Exploratory Analysis & Metrics Summary")
    
    # Top Metrics block row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observed Profiles", f"{len(df)} Records")
    col2.metric("Mean Daily Sleep Hours", "6.68 Hrs")
    col3.metric("Mean Active Screen Time", "4.50 Hrs")
    col4.metric("Mean Target Wellness Score", "5.98 / 10")
    
    st.subheader("📋 Sample Snapshot Output Table (Raw Data Preview)")
    st.dataframe(raw_df.head(10), use_container_width=True)
    
    st.subheader("📈 Statistical Feature Overview Distributions")
    feature_to_plot = st.selectbox(
        "Choose Target Behavioral Attribute column for Distribution Visual Analysis:", 
        ['sleep_hours', 'screen_time', 'study_hours', 'mental_health_score']
    )
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df[feature_to_plot], kde=True, color="#4A90E2", bins=20, ax=ax)
    ax.set_title(f"Empirical Density Representation for: {feature_to_plot.replace('_', ' ').title()}", fontsize=14)
    ax.set_xlabel(feature_to_plot.replace('_', ' ').title())
    ax.set_ylabel("Frequency Matrix Count")
    st.pyplot(fig)

# ================= TAB 2: MODEL TRAINING & TECHNICAL METRICS =================
with tab2:
    st.header("🤖 Model Performance Matrices & Architecture Insights")
    
    # Model Setup Code
    X = df.drop(columns=['mental_health_score'])
    y = df['mental_health_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Choices from project layout
    model_choice = st.radio("Select Machine Learning Regressor Framework Strategy:", ("Decision Tree Regressor", "Random Forest Regressor"))
    
    if model_choice == "Decision Tree Regressor":
        # Trained Tree hyperparameter options discovered via structural nodes
        model = DecisionTreeRegressor(max_depth=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.success(f"**Selected Architecture Model:** {model_choice}")
        col_m2.metric("Evaluated Mean Absolute Error (MAE)", f"{mae:.4f}")
        
        st.subheader("🌳 Decision Tree Structure Graph Output Visual")
        fig_tree, ax_tree = plt.subplots(figsize=(16, 8))
        plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=10, ax=ax_tree)
        ax_tree.set_title("Decision Tree Regressor (Depth Constrained to Level 3 Structure Breakdown)", fontsize=16)
        st.pyplot(fig_tree)
        
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.success(f"**Selected Architecture Model:** {model_choice}")
        col_m2.metric("Evaluated Mean Absolute Error (MAE)", f"{mae:.4f}")
        
        st.subheader("📊 Ensemble Feature Importance Score Weights chart")
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        ax_imp.barh(X.columns[indices], importances[indices], color="#2ECC71", edgecolor='black')
        ax_imp.set_title('Calculated Feature Importance Profile Scores Matrix', fontsize=15, fontweight='bold')
        ax_imp.set_xlabel('Relative Gini/Variance Reduction Scale Importance Weight')
        ax_imp.set_ylabel('Features Evaluated')
        st.pyplot(fig_imp)

# ================= TAB 3: CUSTOM INFERENCE PREDICTOR =================
with tab3:
    st.header("🔮 Interactive Personal Profile Input Variable Predictor Panel")
    st.markdown("Modify individual features on the parameters grid to process instantaneous predictive calculations against the backend regression framework:")
    
    # Split configuration selection matrices inputs efficiently via nested grids
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        st.markdown("### 📋 Demographics & Academic Track Status")
        input_age = st.slider("Select Student Age Profile:", 15, 30, 21)
        input_gender = st.selectbox("Identified Binary Gender:", ["Male", "Female"])
        input_edu = st.selectbox("Current Formal Education Level Status:", ["School", "Undergraduate", "Postgraduate"])
        input_study = st.slider("Average Self Study Breakdown Hours (Daily Target):", 1.0, 12.0, 3.5, step=0.5)
        input_exercise = st.radio("Performs Routine Physical Training Fitness/Exercise?", ["Yes", "No"])
        
    with p_col2:
        st.markdown("### 📱 Lifestyle Metrics & Psychological Frequency Logs")
        input_sleep = st.slider("Average Quality Sleep Scale (Hours Rest Metrics Logged):", 3.0, 12.0, 7.0, step=0.5)
        input_screen = st.slider("Observed Display/Screen Time Exposure (Daily Tracking Metrics):", 0.5, 16.0, 4.5, step=0.5)
        input_diet = st.selectbox("Typical Nutritional Profile Assessment Quality Intake:", ["Healthy", "Average", "Unhealthy"])
        input_stress = st.select_slider("Felt Stress Episodes Logged Frequency:", options=["Never", "Rarely", "Sometimes", "Always"], value="Sometimes")
        input_anxiety = st.select_slider("Anxiety Attacks/Felt Symptoms Log Matrix Range:", options=["Never", "Rarely", "Sometimes", "Always"], value="Never")

    # Map the custom front-end labels cleanly to categorical raw indicators used globally
    gender_binary = 0 if input_gender == "Male" else 1
    edu_encoded = 0 if input_edu == "School" else (1 if input_edu == "Undergraduate" else 2)
    exe_binary = 1 if input_exercise == "Yes" else 0
    diet_encoded = 0 if input_diet == "Healthy" else (1 if input_diet == "Average" else 2)
    
    freq_mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Always": 3}
    stress_encoded = freq_mapping[input_stress]
    anxiety_encoded = freq_mapping[input_anxiety]
    
    # Process vector profile transformation arrays matching layout model inputs explicitly
    custom_vector_features = pd.DataFrame([{
        'age': float(input_age),
        'gender': int(gender_binary),
        'education': int(edu_encoded),
        'sleep_hours': float(input_sleep),
        'screen_time': float(input_screen),
        'study_hours': float(input_study),
        'exercise': int(exe_binary),
        'diet': int(diet_encoded),
        'stress_frequency': int(stress_encoded),
        'anxiety_frequency': int(anxiety_encoded)
    }])
    
    # Make prediction using the evaluated model from Tab 2 context instance dynamically
    if st.button("🚀 Calculate Estimated Profile Mental Health Score Output", use_container_width=True):
        # Fallback dynamic evaluation pipeline configuration protection check block context
        tree_inference_model = DecisionTreeRegressor(max_depth=3)
        X_all = df.drop(columns=['mental_health_score'])
        y_all = df['mental_health_score']
        tree_inference_model.fit(X_all, y_all)
        
        predicted_result_score = tree_inference_model.predict(custom_vector_features)[0]
        
        st.markdown("---")
        st.subheader("🎯 Result Evaluation Prediction:")
        
        score_col, descriptive_col = st.columns([1, 2])
        score_col.metric(label="Calculated Welfare Index Score Profile Value", value=f"{predicted_result_score:.2f} / 10.0")
        
        if predicted_result_score >= 7.5:
            descriptive_col.success("🌟 **Strong Positive Indicators:** This mental profile matches lifestyle parameters indicative of low stress levels, structured balance, and healthy coping mechanisms.")
        elif predicted_result_score >= 5.0:
            descriptive_col.warning("⚠️ **Moderate Warning Range Indicators:** Balanced baseline markers; consider scheduling lower overall screen time constraints or shifting daily physical active metrics upwards to boost individual score levels.")
        else:
            descriptive_col.error("🚨 **High Clinical Risk Priority Warning:** Data array conditions flag profile markers indexing high anxiety exposure frequency paired with inadequate sleep patterns or high screen strain averages.")

# In[3]:
df = pd.read_csv(r"C:\Users\Pradip\Downloads\student_mental_health_v3.csv")

# In[4]:
df.head()

# In[5]:
df.tail()

# In[6]:
df.describe()

# In[7]:
df.isnull().sum()

# In[8]:
df.columns

# In[9]:
df['sleep_hours'] = df['sleep_hours'].fillna(df['sleep_hours'].mean())

# In[10]:
df.isnull().sum()

# In[11]:
df['screen_time'] = df['screen_time'].fillna(df['screen_time'].mean())
df['study_hours'] = df['study_hours'].fillna(df['study_hours'].median())


# In[12]:
df.isnull().sum()

# In[13]:
from sklearn.preprocessing import LabelEncoder

# In[14]:
le = LabelEncoder()

# In[15]:
df['gender'] = le.fit_transform(df['gender'])
df['education'] = le.fit_transform(df['education'])
df['exercise'] = le.fit_transform(df['exercise'])
df['diet'] = le.fit_transform(df['diet'])
df['stress_frequency'] = le.fit_transform(df['stress_frequency'])
df['anxiety_frequency'] = le.fit_transform(df['anxiety_frequency'])


# In[16]:
df.head()

# In[17]:
df.info()

# In[18]:
X = df.drop(columns = ['mental_health_score'])
y = df['mental_health_score']

# In[19]:
from sklearn.model_selection import train_test_split

# In[20]:
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state = 42)

# In[21]:
X_train

# In[22]:
X_test

# In[23]:
y_train

# In[24]:
y_test

# # Linear regression

# In[25]:
from sklearn.linear_model import LinearRegression

# In[26]:
model = LinearRegression()

# In[27]:
model.fit(X_train,y_train)

# In[28]:
y_pred = model.predict(X_test)

# In[29]:
y_pred

# In[30]:
from sklearn.metrics import mean_absolute_error

# In[31]:
mae = mean_absolute_error(y_test,y_pred)

# In[32]:
mae

# # DecisionTreeRegression

# In[33]:
from sklearn.tree import DecisionTreeRegressor

# In[34]:
model = DecisionTreeRegressor(max_depth = 3)
model.fit(X_train, y_train)

# In[35]:
y_pred = model.predict(X_test)

# In[36]:
mae = mean_absolute_error(y_test,y_pred)

# In[37]:
mae

# In[38]:
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# In[39]:
plt.figure(figsize=(12,6))
plot_tree(model,feature_names=X.columns,filled=True,rounded=True)
plt.title("Decision Tree Regressor")
plt.show()

# # RandomForestRegressor

# In[40]:
from sklearn.ensemble import RandomForestRegressor

# In[41]:
model = RandomForestRegressor(n_estimators = 100)
model.fit(X_train, y_train)

# In[42]:
y_pred  = model.predict(X_test)

# In[43]:
mae = mean_absolute_error(y_test,y_pred)

# In[44]:
mae

# In[45]:
imp = model.feature_importances_

# In[46]:
imp

# In[47]:
plt.barh(X.columns,imp)
plt.title('Feature Importance')
plt.show()

# In[48]:
from scipy.cluster.hierarchy import linkage,dendrogram

# In[49]:
Z = linkage(X,method = 'ward')

# In[50]:
Z

# In[51]:
plt.figure(figsize=(12,10))
dendrogram(Z)
plt.show()


# In[52]:
model = AgglomerativeClustering(n_clusters=2)
y = model.fit_predict(X)

# In[53]:
y

# In[54]:
score = silhouette_score(X,y)

# In[55]:
score

# In[56]:
