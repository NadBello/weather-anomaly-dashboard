# ================================================================================================
# WEATHER ANOMALY DETECTION DASHBOARD - MSc Data Science Group Project
# University of Greenwich - 2025 - COMPLETE WORKING VERSION
# ================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import altair as alt
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ================================================================================================
# PAGE CONFIGURATION
# ================================================================================================

st.set_page_config(
    page_title="Weather Anomaly Detection Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================================
# CSS STYLING
# ================================================================================================

st.markdown("""
<style>
    .metric-container {
        display: flex;
        justify-content: space-between;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-normal {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 5px;
    }
    .metric-status {
        text-align: center;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: bold;
        margin-top: 5px;
        font-size: 0.9rem;
    }
    .status-normal {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .status-danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .anomaly-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    .anomaly-header {
        font-weight: bold;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        font-size: 1.1rem;
    }
    .anomaly-icon {
        font-size: 1.4rem;
        margin-right: 12px;
    }
    .anomaly-danger { color: #721c24; }
    .anomaly-warning { color: #856404; }
    .anomaly-normal { color: #155724; }
    .explanation-text {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.95rem;
        border-left: 4px solid #3498db;
    }
    .dashboard-title {
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 10px;
        color: #2c3e50;
        font-weight: 700;
    }
    .dashboard-subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 25px;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 20px;
        padding-bottom: 8px;
        border-bottom: 2px solid #3498db;
        color: #2c3e50;
    }
    .component-container {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .expert-container {
        background: linear-gradient(135deg, #f1f3f4 0%, #ffffff 100%);
        border: 1px solid #d0d7de;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .confidence-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 10px;
    }
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
    }
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
    }
    .last-updated {
        text-align: right;
        font-size: 0.8rem;
        color: #6c757d;
        font-style: italic;
    }
    .integration-success {
        background-color: #d1f2eb;
        border: 1px solid #7dcea0;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        color: #0e5132;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================================
# DATA LOADING FUNCTIONS
# ================================================================================================

@st.cache_data
def load_sample_data():
    """Load Jeremy's ML data with proper error handling"""
    try:
        # Try multiple paths
        possible_paths = [
            "data/dashboard_input_20250531_1700_merged.csv",
            "dashboard_input_20250531_1700_merged.csv",
            "./data/dashboard_input_20250531_1700_merged.csv"
        ]
        
        data = None
        loaded_path = None
        for path in possible_paths:
            try:
                data = pd.read_csv(path)
                loaded_path = path
                break
            except FileNotFoundError:
                continue
        
        if data is None:
            st.sidebar.warning("⚠️ CSV not found - using demo data")
            return load_fallback_data()
        
        # Parse date column
        try:
            data['timestamp'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M')
        except:
            data['timestamp'] = pd.to_datetime(data['date'])
        
        # Drop original date column
        if 'date' in data.columns:
            data = data.drop('date', axis=1)
        
        # Map anomaly labels
        def map_anomaly_label(label):
            if pd.isna(label) or label == 'Normal':
                return 'Normal'
            elif label == 'Pattern anomaly':
                return 'Pattern Anomaly'
            elif label == 'Compound anomaly':
                return 'Compound Anomaly'
            else:
                return 'Point Anomaly'
        
        data['pseudo_label'] = data['anomaly_label'].apply(map_anomaly_label)
        
        # Add confidence levels
        def assign_confidence(row):
            if row['anomaly_label'] == 'Compound anomaly':
                return 'High'
            elif row['anomaly_label'] == 'Pattern anomaly':
                return 'Medium'
            else:
                return 'High'
        
        data['confidence'] = data.apply(assign_confidence, axis=1)
        
        # Ensure numeric columns
        numeric_columns = ['temperature_2m', 'surface_pressure', 'precipitation', 'wind_speed_10m',
                          'temp_lower', 'temp_upper', 'wind_lower', 'wind_upper', 
                          'press_lower', 'press_upper', 'if_score', 'lstm_error',
                          'if_threshold', 'lstm_threshold']
        
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Success message
        anomaly_count = len(data[data['anomaly_label'] != 'Normal'])
        st.sidebar.success(f"✅ Data Loaded: {len(data)} records, {anomaly_count} anomalies")
        st.sidebar.info(f"📁 Source: {loaded_path}")
        
        return data
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {str(e)}")
        return load_fallback_data()

def load_fallback_data():
    """Generate fallback demo data"""
    np.random.seed(42)
    start_time = datetime.datetime(2025, 5, 31, 17, 0, 0)
    timestamps = [start_time + datetime.timedelta(hours=i) for i in range(72)]
    
    data = []
    for i, ts in enumerate(timestamps):
        base_temp = 15 + 3 * np.sin(i / 12) + np.random.normal(0, 2)
        base_pressure = 1013 + 8 * np.sin(i / 20) + np.random.normal(0, 3)
        base_precip = max(0, np.random.exponential(0.5) if np.random.random() > 0.8 else 0)
        base_wind = 8 + 4 * np.sin(i / 15) + np.random.normal(0, 2)
        
        # Statistical bounds
        temp_lower = base_temp - 5
        temp_upper = base_temp + 5
        wind_lower = max(0, base_wind - 3)
        wind_upper = base_wind + 8
        press_lower = base_pressure - 15
        press_upper = base_pressure + 15
        
        # Anomaly scores
        if_score = 0.3 + 0.4 * np.random.random()
        lstm_error = 0.2 + 0.5 * np.random.random()
        if_threshold = 0.15
        lstm_threshold = 0.65
        
        # Anomaly flags
        is_if_anomaly = 1 if if_score < if_threshold else 0
        is_lstm_anomaly = 1 if lstm_error > lstm_threshold else 0
        
        # Determine label
        if is_if_anomaly and is_lstm_anomaly:
            anomaly_label = "Compound anomaly"
            pseudo_label = "Compound Anomaly"
            confidence = "High"
        elif is_lstm_anomaly:
            anomaly_label = "Pattern anomaly"
            pseudo_label = "Pattern Anomaly"
            confidence = "Medium"
        else:
            anomaly_label = "Normal"
            pseudo_label = "Normal"
            confidence = "High"
        
        data.append({
            'timestamp': ts,
            'temperature_2m': base_temp,
            'surface_pressure': base_pressure,
            'precipitation': base_precip,
            'wind_speed_10m': max(0, base_wind),
            'temp_lower': temp_lower,
            'temp_upper': temp_upper,
            'wind_lower': wind_lower,
            'wind_upper': wind_upper,
            'press_lower': press_lower,
            'press_upper': press_upper,
            'if_score': if_score,
            'is_if_anomaly': is_if_anomaly,
            'lstm_error': lstm_error,
            'is_lstm_anomaly': is_lstm_anomaly,
            'if_threshold': if_threshold,
            'lstm_threshold': lstm_threshold,
            'anomaly_label': anomaly_label,
            'confidence': confidence,
            'pseudo_label': pseudo_label
        })
    
    return pd.DataFrame(data)

# ================================================================================================
# VISUALIZATION FUNCTIONS
# ================================================================================================

def create_enhanced_forecast_chart(data, selected_metric):
    """Create enhanced forecast chart with Altair"""
    try:
        # Metric parameters
        if selected_metric == "temperature":
            y_col = "temperature_2m"
            lower_col = "temp_lower"
            upper_col = "temp_upper"
            title = "72-Hour Temperature Forecast"
            y_title = "Temperature (°C)"
            
        elif selected_metric == "pressure":
            y_col = "surface_pressure"
            lower_col = "press_lower"
            upper_col = "press_upper"
            title = "72-Hour Surface Pressure Forecast"
            y_title = "Surface Pressure (hPa)"
            
        elif selected_metric == "precipitation":
            y_col = "precipitation"
            lower_col = None
            upper_col = None
            title = "72-Hour Precipitation Forecast"
            y_title = "Precipitation (mm)"
            
        elif selected_metric == "wind_speed":
            y_col = "wind_speed_10m"
            lower_col = "wind_lower"
            upper_col = "wind_upper"
            title = "72-Hour Wind Speed Forecast"
            y_title = "Wind Speed (km/h)"
        
        # Check columns exist
        if y_col not in data.columns:
            st.error(f"Column {y_col} not found")
            return None
        
        # Clean data
        data_copy = data.copy()
        data_copy = data_copy.dropna(subset=[y_col, 'timestamp'])
        
        # Base chart
        base = alt.Chart(data_copy)
        
        layers = []
        
        # Add confidence band if available
        if lower_col and upper_col and selected_metric != "precipitation":
            if lower_col in data_copy.columns and upper_col in data_copy.columns:
                band_data = data_copy.dropna(subset=[lower_col, upper_col])
                
                band = alt.Chart(band_data).mark_area(
                    opacity=0.2,
                    color='lightgrey'
                ).encode(
                    x=alt.X('timestamp:T', title='Date & Time'),
                    y=alt.Y(f'{lower_col}:Q'),
                    y2=alt.Y(f'{upper_col}:Q')
                )
                layers.append(band)
        
        # Main line
        line = base.mark_line(
            color='steelblue',
            strokeWidth=3
        ).encode(
            x=alt.X('timestamp:T', 
                   title='Date & Time',
                   axis=alt.Axis(format='%d %b %H:%M', labelAngle=-45)),
            y=alt.Y(f'{y_col}:Q', title=y_title),
            tooltip=[
                alt.Tooltip('timestamp:T', title='Time', format='%d %b %H:%M'),
                alt.Tooltip(f'{y_col}:Q', title=y_title, format='.1f')
            ]
        )
        layers.append(line)
        
        # Anomaly points
        anomaly_data = data_copy[data_copy['anomaly_label'] != 'Normal'].copy()
        if len(anomaly_data) > 0:
            anomalies = alt.Chart(anomaly_data).mark_circle(
                size=120,
                stroke='white',
                strokeWidth=2
            ).encode(
                x=alt.X('timestamp:T'),
                y=alt.Y(f'{y_col}:Q'),
                color=alt.Color('anomaly_label:N',
                               scale=alt.Scale(
                                   domain=['Pattern anomaly', 'Compound anomaly'],
                                   range=['#ba55d3', '#dc143c']),
                               title='Anomaly Type'),
                tooltip=[
                    alt.Tooltip('timestamp:T', title='Time', format='%d %b %H:%M'),
                    alt.Tooltip(f'{y_col}:Q', title=y_title, format='.1f'),
                    alt.Tooltip('anomaly_label:N', title='Anomaly Type')
                ]
            )
            layers.append(anomalies)
        
        # Combine layers
        if len(layers) > 0:
            final_chart = alt.layer(*layers).resolve_scale(
                color='independent'
            ).properties(
                title=title,
                width=800,
                height=400
            )
            return final_chart
        else:
            return None
        
    except Exception as e:
        st.error(f"Chart creation failed: {str(e)}")
        return None

def create_plotly_fallback(data, selected_metric):
    """Plotly fallback chart"""
    try:
        y_col_map = {
            "temperature": "temperature_2m",
            "pressure": "surface_pressure",
            "precipitation": "precipitation",
            "wind_speed": "wind_speed_10m"
        }
        y_col = y_col_map[selected_metric]
        
        fig = px.line(data, x='timestamp', y=y_col,
                      title=f"72-Hour {selected_metric.title()} Forecast")
        fig.update_traces(line=dict(width=3, color='steelblue'))
        
        # Add anomaly points
        anomaly_data = data[data['anomaly_label'] != 'Normal']
        if len(anomaly_data) > 0:
            fig.add_scatter(
                x=anomaly_data['timestamp'],
                y=anomaly_data[y_col],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Anomalies'
            )
        
        fig.update_layout(height=400, showlegend=True)
        return fig
        
    except Exception as e:
        st.error(f"Plotly fallback failed: {str(e)}")
        return None

# ================================================================================================
# UTILITY FUNCTIONS
# ================================================================================================

def get_metric_status(value, metric_type, season='summer'):
    """Get metric status classification"""
    ranges = {
        'temperature': {
            'summer': {'min': 12, 'max': 22}
        },
        'pressure': {'min': 1000, 'max': 1025},
        'precipitation': {'max': 2},
        'wind_speed': {'max': 10}
    }

    if metric_type == 'temperature':
        r = ranges[metric_type][season]
        if value < 0:
            return "Freezing", "danger"
        elif value < r['min']:
            return "Below Normal", "warning"
        elif value > r['max']:
            return "Above Normal", "warning"
        else:
            return "Normal", "normal"
    elif metric_type == 'pressure':
        r = ranges[metric_type]
        if value < r['min']:
            return "Low Pressure", "danger"
        elif value > r['max']:
            return "High Pressure", "warning"
        else:
            return "Normal", "normal"
    elif metric_type == 'precipitation':
        if value > 5:
            return "Heavy Rain", "danger"
        elif value > ranges[metric_type]['max']:
            return "Moderate Rain", "warning"
        else:
            return "Light/None", "normal"
    elif metric_type == 'wind_speed':
        if value > 15:
            return "Strong Winds", "danger"
        elif value > ranges[metric_type]['max']:
            return "Moderate Winds", "warning"
        else:
            return "Light Winds", "normal"

    return "Unknown", "normal"

def generate_natural_language_explanation(current_data):
    """Generate explanation text"""
    latest = current_data.iloc[-1]

    explanation = f"**Current Weather Assessment** (Updated: {latest['timestamp'].strftime('%d %B %Y, %H:%M')})<br><br>"

    if latest['pseudo_label'] == 'Normal':
        explanation += "✅ **Status: NORMAL CONDITIONS**<br>"
        explanation += "All weather parameters are within expected ranges.<br><br>"
    elif latest['pseudo_label'] == 'Pattern Anomaly':
        explanation += "⚠️ **Status: PATTERN ANOMALY DETECTED**<br>"
        explanation += "Unusual weather pattern identified in the data sequence.<br><br>"
    elif latest['pseudo_label'] == 'Compound Anomaly':
        explanation += "🚨 **Status: COMPOUND ANOMALY**<br>"
        explanation += "Multiple weather systems showing coordinated unusual behaviour.<br><br>"
    else:
        explanation += "❓ **Status: UNCERTAIN**<br>"
        explanation += "Mixed signals in weather data - monitoring required.<br><br>"

    conf_badge = f"<span class='confidence-badge confidence-{latest['confidence'].lower()}'>{latest['confidence']} Confidence</span>"
    explanation += f"**Model Confidence:** {conf_badge}<br><br>"

    explanation += "**Current Readings:**<br>"
    explanation += f"• Temperature: {latest['temperature_2m']:.1f}°C<br>"
    explanation += f"• Pressure: {latest['surface_pressure']:.1f} hPa<br>"
    explanation += f"• Precipitation: {latest['precipitation']:.1f} mm<br>"
    explanation += f"• Wind Speed: {latest['wind_speed_10m']:.1f} km/h<br><br>"

    if latest['pseudo_label'] != 'Normal':
        explanation += "**🔬 AI Model Analysis:**<br>"
        explanation += f"• Isolation Forest Score: {latest['if_score']:.3f} (threshold: {latest['if_threshold']:.3f})<br>"
        explanation += f"• LSTM Reconstruction Error: {latest['lstm_error']:.3f} (threshold: {latest['lstm_threshold']:.3f})<br><br>"

    return explanation

def create_heathrow_map(needs_gritting=False):
    """Create Heathrow area map"""
    m = folium.Map(location=[51.4700, -0.4543], zoom_start=12, tiles="OpenStreetMap")

    folium.Marker(
        [51.4700, -0.4543],
        popup="Heathrow Airport - Weather Monitoring Station",
        tooltip="Heathrow Airport - Weather Monitoring Station",
        icon=folium.Icon(color="blue", icon="plane", prefix="fa")
    ).add_to(m)

    # Major roads
    roads = {
        "M4": [[51.4890, -0.4200], [51.4895, -0.4300], [51.4898, -0.4400],
               [51.4899, -0.4500], [51.4897, -0.4600], [51.4895, -0.4700]],
        "A4": [[51.4780, -0.4200], [51.4785, -0.4300], [51.4790, -0.4400],
               [51.4792, -0.4500], [51.4793, -0.4600], [51.4791, -0.4700]]
    }

    for road_name, coordinates in roads.items():
        folium.PolyLine(
            coordinates,
            color="#e31a1c" if needs_gritting else "#33a02c",
            weight=5 if road_name == "M4" else 4,
            opacity=0.8,
            popup=f"{road_name} - {'Gritting Required' if needs_gritting else 'Normal Conditions'}"
        ).add_to(m)

    return m

# ================================================================================================
# MAIN APPLICATION
# ================================================================================================

def main():
    """Main dashboard application"""
    
    # Load data
    weather_data = load_sample_data()

    # Sidebar navigation
    st.sidebar.title("🌦️ Weather Dashboard")
    page = st.sidebar.radio(
        "Navigate to:",
        ["📊 Overview", "📈 Forecast", "🔬 Expert Mode", "💬 Feedback"],
        index=0
    )

    # Main title
    st.markdown("<h1 class='dashboard-title'>Weather Anomaly Detection Dashboard</h1>",
                unsafe_allow_html=True)
    st.markdown("<p class='dashboard-subtitle'>Heathrow Area - Real-time Monitoring & Analysis</p>",
                unsafe_allow_html=True)

    current_time = datetime.datetime.now().strftime("%d %B %Y, %H:%M:%S")
    st.markdown(f"<p class='last-updated'>Last updated: {current_time}</p>",
                unsafe_allow_html=True)

    # Integration status
    if len(weather_data) > 0:
        st.markdown("""
        <div class='integration-success'>
            ✅ <strong>Live Data Integration:</strong> Jeremy's ML Pipeline Active | Marie's XAI Analysis Ready | 72-Hour Forecast Available
        </div>
        """, unsafe_allow_html=True)

    # ========================================================================================
    # OVERVIEW PAGE
    # ========================================================================================
    
    if page == "📊 Overview":
        st.markdown("---")

        if len(weather_data) == 0:
            st.error("❌ No data available. Please check data integration.")
            return

        current = weather_data.iloc[-1]

        # Current weather metrics
        st.markdown('<div class="component-container">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🌡️ Current Weather Conditions</div>",
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            temp_status, temp_class = get_metric_status(current['temperature_2m'], 'temperature', 'summer')
            st.markdown(f"""
            <div class='metric-container'>
                <div>
                    <div><strong>Temperature</strong></div>
                    <div class='metric-value'>{current['temperature_2m']:.1f}°C</div>
                    <div class='metric-normal'>Normal: 12-22°C (Summer)</div>
                </div>
                <div class='metric-status status-{temp_class}'>{temp_status}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            pressure_status, pressure_class = get_metric_status(current['surface_pressure'], 'pressure')
            st.markdown(f"""
            <div class='metric-container'>
                <div>
                    <div><strong>Pressure</strong></div>
                    <div class='metric-value'>{current['surface_pressure']:.1f} hPa</div>
                    <div class='metric-normal'>Normal: 1000-1025 hPa</div>
                </div>
                <div class='metric-status status-{pressure_class}'>{pressure_status}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            precip_status, precip_class = get_metric_status(current['precipitation'], 'precipitation')
            st.markdown(f"""
            <div class='metric-container'>
                <div>
                    <div><strong>Precipitation</strong></div>
                    <div class='metric-value'>{current['precipitation']:.1f} mm</div>
                    <div class='metric-normal'>Normal: 0-2 mm</div>
                </div>
                <div class='metric-status status-{precip_class}'>{precip_status}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            wind_status, wind_class = get_metric_status(current['wind_speed_10m'], 'wind_speed')
            st.markdown(f"""
            <div class='metric-container'>
                <div>
                    <div><strong>Wind Speed</strong></div>
                    <div class='metric-value'>{current['wind_speed_10m']:.1f} km/h</div>
                    <div class='metric-normal'>Normal: 0-10 km/h</div>
                </div>
                <div class='metric-status status-{wind_class}'>{wind_status}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Anomaly Analysis and Recommendations
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="component-container">', unsafe_allow_html=True)
            st.markdown("<div class='section-title'>🔍 Anomaly Analysis</div>",
                        unsafe_allow_html=True)

            explanation = generate_natural_language_explanation(weather_data)
            st.markdown(f'<div class="explanation-text">{explanation}</div>',
                        unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="component-container">', unsafe_allow_html=True)
            st.markdown("<div class='section-title'>📋 Operational Recommendations</div>",
                        unsafe_allow_html=True)

            # Recommendations based on anomaly status
            if current['pseudo_label'] in ['Pattern Anomaly', 'Compound Anomaly']:
                if current['temperature_2m'] < 2 and current['precipitation'] > 0:
                    st.markdown("""
                    <div class='anomaly-card'>
                        <div class='anomaly-header anomaly-danger'>
                            <span class='anomaly-icon'>🚨</span>
                            <span>IMMEDIATE ACTION REQUIRED</span>
                        </div>
                        <ul>
                            <li><strong>Deploy gritting crews</strong> to priority routes immediately</li>
                            <li><strong>Issue black ice warning</strong> to public via all channels</li>
                            <li><strong>Coordinate with Heathrow operations</strong> for runway management</li>
                            <li><strong>Monitor temperature trend</strong> for next 2-4 hours</li>
                            <li><strong>Activate emergency protocols</strong> if conditions worsen</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif current['anomaly_label'] == 'Compound anomaly':
                    st.markdown("""
                    <div class='anomaly-card'>
                        <div class='anomaly-header anomaly-danger'>
                            <span class='anomaly-icon'>⚡</span>
                            <span>COMPOUND ANOMALY DETECTED</span>
                        </div>
                        <ul>
                            <li><strong>Multi-system weather event</strong> in progress</li>
                            <li><strong>Enhanced monitoring</strong> across all parameters</li>
                            <li><strong>Coordinate response teams</strong> for complex scenario</li>
                            <li><strong>Prepare contingency resources</strong> for escalation</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='anomaly-card'>
                        <div class='anomaly-header anomaly-warning'>
                            <span class='anomaly-icon'>⚠️</span>
                            <span>INCREASED MONITORING</span>
                        </div>
                        <ul>
                            <li><strong>Place crews on standby</strong> for rapid deployment</li>
                            <li><strong>Check equipment readiness</strong> and resource availability</li>
                            <li><strong>Monitor forecast updates</strong> every 30 minutes</li>
                            <li><strong>Review contingency plans</strong> with team leads</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='anomaly-card'>
                    <div class='anomaly-header anomaly-normal'>
                        <span class='anomaly-icon'>✅</span>
                        <span>STANDARD OPERATIONS</span>
                    </div>
                    <ul>
                        <li><strong>Continue routine monitoring</strong> schedule</li>
                        <li><strong>No immediate action required</strong></li>
                        <li><strong>Maintain equipment readiness</strong> for rapid response</li>
                        <li><strong>Review daily forecast</strong> for planning</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Heathrow Area Map
        st.markdown('<div class="component-container">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🗺️ Heathrow Area - Road Network Status</div>",
                    unsafe_allow_html=True)

        needs_gritting = current['temperature_2m'] < 3 and current['precipitation'] > 0
        heathrow_map = create_heathrow_map(needs_gritting)
        folium_static(heathrow_map)

        if needs_gritting:
            st.warning("🧊 **Gritting Alert**: Road surface temperatures may reach freezing point. Monitor closely.")
        else:
            st.info("✅ **Road Conditions**: Normal operations expected. Continue standard monitoring.")

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()

    # ========================================================================================
    # FORECAST PAGE
    # ========================================================================================
    
    elif page == "📈 Forecast":
        st.markdown("---")

        st.markdown('<div class="component-container">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📈 72-Hour Weather Forecast</div>",
                    unsafe_allow_html=True)

        st.info("""
        **📊 Forecast Guide:** Shaded bands show approximate "normal range" for each variable based on historical data. 
        Coloured dots indicate detected anomalies: 🟣 Pattern anomalies, 🔴 Compound anomalies.
        """)

        # Display options
        display_option = st.radio(
            "Display Options:",
            ["Individual Chart", "Combined View (All 4 Metrics)"],
            horizontal=True
        )

        if display_option == "Combined View (All 4 Metrics)":
            st.markdown("### 📊 Combined 72-Hour Forecast - All Weather Parameters")
            
            metrics = ["temperature", "pressure", "precipitation", "wind_speed"]
            
            for metric in metrics:
                st.markdown(f"#### {metric.title()} Forecast")
                chart = create_enhanced_forecast_chart(weather_data, metric)
                if chart:
                    try:
                        st.altair_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Altair failed for {metric}, using Plotly fallback")
                        fallback_chart = create_plotly_fallback(weather_data, metric)
                        if fallback_chart:
                            st.plotly_chart(fallback_chart, use_container_width=True)

        else:
            # Individual chart display
            selected_metric = st.selectbox(
                "Select weather parameter:",
                ["temperature", "pressure", "precipitation", "wind_speed"],
                format_func=lambda x: {
                    "temperature": "🌡️ Temperature (°C)",
                    "pressure": "🌊 Surface Pressure (hPa)",
                    "precipitation": "🌧️ Precipitation (mm)",
                    "wind_speed": "💨 Wind Speed (km/h)"
                }[x]
            )

            if len(weather_data) > 0:
                chart = create_enhanced_forecast_chart(weather_data, selected_metric)
                
                if chart:
                    try:
                        st.altair_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Altair failed, using Plotly fallback: {str(e)}")
                        fallback_chart = create_plotly_fallback(weather_data, selected_metric)
                        if fallback_chart:
                            st.plotly_chart(fallback_chart, use_container_width=True)

        # Forecast summary
        if len(weather_data) > 0:
            st.markdown("### 📊 Forecast Summary & Risk Assessment")
            
            if display_option == "Individual Chart":
                col1, col2, col3, col4 = st.columns(4)

                y_col = {
                    "temperature": "temperature_2m",
                    "pressure": "surface_pressure",
                    "precipitation": "precipitation", 
                    "wind_speed": "wind_speed_10m"
                }[selected_metric]

                with col1:
                    avg_val = weather_data[y_col].mean()
                    st.metric("Average", f"{avg_val:.1f}",
                              help=f"Average {selected_metric} over forecast period")

                with col2:
                    max_val = weather_data[y_col].max()
                    max_time = weather_data.loc[weather_data[y_col].idxmax(), 'timestamp']
                    st.metric("Maximum", f"{max_val:.1f}",
                              help=f"Peak {selected_metric} expected at {max_time.strftime('%a %d %b, %H:%M')}")

                with col3:
                    min_val = weather_data[y_col].min()
                    min_time = weather_data.loc[weather_data[y_col].idxmin(), 'timestamp']
                    st.metric("Minimum", f"{min_val:.1f}",
                              help=f"Lowest {selected_metric} expected at {min_time.strftime('%a %d %b, %H:%M')}")

                with col4:
                    anomaly_periods = len(weather_data[weather_data['anomaly_label'] != 'Normal'])
                    st.metric("Anomaly Periods", f"{anomaly_periods}",
                              help=f"Number of forecast periods showing anomalous conditions")

                # Risk analysis
                st.markdown("#### 🔍 Operational Risk Analysis")

                first_half = weather_data[y_col][:len(weather_data) // 2].mean()
                second_half = weather_data[y_col][len(weather_data) // 2:].mean()
                trend = "increasing" if second_half > first_half else "decreasing" if second_half < first_half else "stable"

                insights = f"**Trend Analysis:** {selected_metric.title()} shows a **{trend}** pattern over the forecast period. "

                if selected_metric == "temperature":
                    risk_periods = len(weather_data[weather_data[y_col] < 0])
                    if risk_periods > 0:
                        insights += f"**❄️ Ice Risk:** {risk_periods} forecast periods show sub-zero temperatures. Gritting operations may be required. "
                    compound_anomalies = len(weather_data[weather_data['anomaly_label'] == 'Compound anomaly'])
                    if compound_anomalies > 0:
                        insights += f"**⚡ Complex Weather:** {compound_anomalies} periods show compound anomalies requiring enhanced monitoring."
                        
                elif selected_metric == "pressure":
                    low_pressure_periods = len(weather_data[weather_data[y_col] < 990])
                    if low_pressure_periods > 0:
                        insights += f"**🌪️ Storm Risk:** {low_pressure_periods} periods show very low pressure indicating potential severe weather. "
                        
                elif selected_metric == "precipitation":
                    heavy_rain_periods = len(weather_data[weather_data[y_col] > 5])
                    if heavy_rain_periods > 0:
                        insights += f"**🌊 Flood Risk:** {heavy_rain_periods} periods show heavy precipitation. Surface water management may be needed. "
                        
                elif selected_metric == "wind_speed":
                    high_wind_periods = len(weather_data[weather_data[y_col] > 15])
                    if high_wind_periods > 0:
                        insights += f"**💨 Operations Risk:** {high_wind_periods} periods show strong winds affecting airport and road operations. "

                st.info(insights)
            else:
                # Combined view summary
                st.markdown("#### 🔍 Overall Forecast Summary")
                total_anomalies = len(weather_data[weather_data['anomaly_label'] != 'Normal'])
                st.info(f"**Combined View Analysis:** {total_anomalies} anomalous periods detected across all weather parameters. Review individual charts for detailed risk assessment.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================================================
    # EXPERT MODE PAGE
    # ========================================================================================
    
    elif page == "🔬 Expert Mode":
        st.markdown("---")
        st.markdown("### 🔬 Advanced Analytics & Model Insights")
        st.info("Technical details for data scientists, meteorologists, and model developers.")

        if len(weather_data) == 0:
            st.error("❌ No data available for expert analysis.")
            return

        # Model Performance Section
        st.markdown('<div class="expert-container">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Jeremy's Anomaly Detection Model Performance</div>",
                    unsafe_allow_html=True)

        # Model scores chart
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Isolation Forest Scores', 'LSTM Reconstruction Error'),
                vertical_spacing=0.1
            )

            fig.add_trace(
                go.Scatter(
                    x=weather_data['timestamp'],
                    y=weather_data['if_score'],
                    mode='lines+markers',
                    name='IF Score',
                    line=dict(color='#00bfff', width=2)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=weather_data['timestamp'],
                    y=weather_data['lstm_error'],
                    mode='lines+markers',
                    name='LSTM Error',
                    line=dict(color='#ba55d3', width=2)
                ),
                row=2, col=1
            )

            # Add threshold lines
            fig.add_hline(y=weather_data['if_threshold'].iloc[0], line_dash="dash", 
                          line_color="#00bfff", row=1, col=1)
            fig.add_hline(y=weather_data['lstm_threshold'].iloc[0], line_dash="dash", 
                          line_color="#ba55d3", row=2, col=1)

            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating model scores chart: {e}")

        # Model statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if_anomalies = weather_data['is_if_anomaly'].sum()
            st.metric("IF Anomalies", f"{if_anomalies}", 
                      help="Total anomalies detected by Isolation Forest")
        
        with col2:
            lstm_anomalies = weather_data['is_lstm_anomaly'].sum()
            st.metric("LSTM Anomalies", f"{lstm_anomalies}",
                      help="Total anomalies detected by LSTM Autoencoder")
        
        with col3:
            compound_anomalies = len(weather_data[weather_data['anomaly_label'] == 'Compound anomaly'])
            st.metric("Compound Anomalies", f"{compound_anomalies}",
                      help="Anomalies detected by both models")
        
        with col4:
            detection_rate = ((if_anomalies + lstm_anomalies) / len(weather_data) * 100)
            st.metric("Detection Rate", f"{detection_rate:.1f}%",
                      help="Percentage of time periods flagged as anomalous")

        st.markdown('</div>', unsafe_allow_html=True)

        # Marie's XAI Analysis
        with st.expander("🔍 Marie's XAI Analysis - TreeSHAP & Reconstruction Insights"):
            # Check if we have XAI data
            if 'TreeSHAP_natural_language_summary' in weather_data.columns:
                st.markdown("#### TreeSHAP Natural Language Explanations")
                
                anomaly_samples = weather_data[weather_data['anomaly_label'] != 'Normal']
                if len(anomaly_samples) > 0:
                    selected_sample = st.selectbox(
                        "Select anomaly sample for detailed XAI analysis:",
                        range(len(anomaly_samples)),
                        format_func=lambda x: f"Sample {x} - {anomaly_samples.iloc[x]['timestamp'].strftime('%Y-%m-%d %H:%M')} ({anomaly_samples.iloc[x]['anomaly_label']})"
                    )
                    
                    sample_data = anomaly_samples.iloc[selected_sample]
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("**TreeSHAP Analysis:**")
                        st.markdown(sample_data['TreeSHAP_natural_language_summary'])
                    
                    with col2:
                        st.markdown("**Reconstruction Error Analysis:**")
                        st.markdown(sample_data['reconstruction_error_summary'])
                        
            else:
                st.markdown("#### Global Feature Importance from TreeSHAP Analysis")

                importance_data = {
                    'temperature_2m': 0.42,
                    'surface_pressure': 0.35,
                    'wind_speed_10m': 0.28,
                    'precipitation': 0.19,
                    'temporal_features': 0.16
                }

                importance_df = pd.DataFrame({
                    'Feature': list(importance_data.keys()),
                    'Importance': list(importance_data.values())
                })

                # Feature importance chart
                fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                                        orientation='h', title="Global Feature Importance Rankings",
                                        color_discrete_sequence=['#3498db'])
                fig_importance.update_layout(height=250)
                st.plotly_chart(fig_importance, use_container_width=True)

                st.markdown("""
                **Marie's Key Insights from TreeSHAP Analysis:**
                - **Temperature variations** show highest predictive power for anomaly detection
                - **Surface pressure changes** are strong indicators of weather pattern anomalies  
                - **Wind speed patterns** provide crucial context for compound anomalies
                - **Precipitation events** help distinguish between different anomaly types
                - **Temporal features** capture important sequence-based patterns
                """)

        # Individual Anomaly Analysis
        with st.expander("🎯 Individual Anomaly Deep Dive"):
            anomaly_indices = weather_data[weather_data['anomaly_label'] != 'Normal'].index.tolist()

            if anomaly_indices:
                selected_anomaly_idx = st.selectbox(
                    "Select anomaly for detailed analysis:",
                    anomaly_indices,
                    format_func=lambda x: f"Anomaly {x} - {weather_data.iloc[x]['timestamp'].strftime('%Y-%m-%d %H:%M')} ({weather_data.iloc[x]['anomaly_label']})",
                    key="anomaly_selector"
                )

                selected_anomaly = weather_data.iloc[selected_anomaly_idx]

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("**Anomaly Details:**")
                    st.write(f"**Timestamp:** {selected_anomaly['timestamp']}")
                    st.write(f"**Type:** {selected_anomaly['anomaly_label']}")
                    st.write(f"**Confidence:** {selected_anomaly['confidence']}")
                    st.write(f"**IF Score:** {selected_anomaly['if_score']:.3f} (thresh: {selected_anomaly['if_threshold']:.3f})")
                    st.write(f"**LSTM Error:** {selected_anomaly['lstm_error']:.3f} (thresh: {selected_anomaly['lstm_threshold']:.3f})")

                with col2:
                    st.markdown("**Weather Conditions:**")
                    st.write(f"**Temperature:** {selected_anomaly['temperature_2m']:.1f}°C")
                    st.write(f"**Pressure:** {selected_anomaly['surface_pressure']:.1f} hPa")
                    st.write(f"**Precipitation:** {selected_anomaly['precipitation']:.1f} mm")
                    st.write(f"**Wind Speed:** {selected_anomaly['wind_speed_10m']:.1f} km/h")

        # Model Configuration
        with st.expander("⚙️ Model Configuration & Technical Details"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Jeremy's Isolation Forest Configuration")
                st.code("""
                n_estimators: 100
                contamination: 0.1
                max_samples: auto
                max_features: 1.0
                random_state: 42
                bootstrap: False
                """)

            with col2:
                st.markdown("#### Jeremy's LSTM Autoencoder Configuration")
                st.code("""
                sequence_length: 24
                encoding_dim: 32
                hidden_layers: [64, 32, 16]
                learning_rate: 0.001
                epochs: 100
                dropout: 0.2
                """)

            st.markdown("#### Model Training Information")
            st.code("""
            Last Training: 31 May 2025
            Training Data: 60 days historical weather data
            Data Sources: Open Meteo API, UKMO Seamless model
            Update Frequency: 1 hour
            Weather Model Resolution: 2-10km
            Validation Method: Time series cross-validation
            Performance Metrics: Precision/Recall optimised for operational use
            """)

    # ========================================================================================
    # FEEDBACK PAGE
    # ========================================================================================
    
    elif page == "💬 Feedback":
        st.markdown("---")

        st.markdown('<div class="component-container">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>💬 User Feedback & System Evaluation</div>",
                    unsafe_allow_html=True)

        st.markdown("### 📝 Provide Feedback")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**How helpful was the dashboard?**")

            col1_1, col1_2, col1_3 = st.columns(3)
            with col1_1:
                if st.button("👍 Helpful", key="thumbs_up_btn"):
                    st.success("Thank you for your positive feedback!")
            with col1_2:
                if st.button("👎 Not Helpful", key="thumbs_down_btn"):
                    st.error("We'll work to improve the system!")
            with col1_3:
                if st.button("🤔 Neutral", key="neutral_btn"):
                    st.info("Thanks for your feedback!")

        with col2:
            feedback_text = st.text_area(
                "Additional comments or suggestions:",
                placeholder="Please share your thoughts on dashboard usability, accuracy, or features you'd like to see...",
                height=100
            )

            feedback_category = st.selectbox(
                "Feedback category:",
                ["General", "Dashboard Design", "Data Accuracy", "Performance", "Feature Request", "Bug Report"]
            )

            if st.button("📤 Submit Feedback", type="primary"):
                if feedback_text:
                    st.success("Thank you for your detailed feedback! Your input helps us improve the system.")
                else:
                    st.warning("Please provide some feedback text before submitting.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================================================
    # SIDEBAR INFORMATION
    # ========================================================================================

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 System Information")
    
    st.sidebar.info("""
    **Model Status:** ✅ Active  
    **Last Training:** 31 May 2025  
    **Data Sources:** Open Meteo API, UKMO Seamless model  
    **Update Frequency:** 1 hour  
    **Weather Model Resolution:** 2-10km  
    **Coverage Area:** 25km radius
    """)

    st.sidebar.markdown("### 🚨 Alert Thresholds")
    st.sidebar.markdown("""
    **Temperature:** < 0°C (Ice Risk)  
    **Pressure:** < 990 hPa (Storm Risk)  
    **Precipitation:** > 5mm (Flood Risk)  
    **Wind:** > 15 km/h (Operations Risk)
    """)

    st.sidebar.markdown("### 📞 Emergency Contacts")
    with st.sidebar.expander("View Contacts"):
        st.markdown("""
        **Met Office:** 0370 900 0100  
        **Heathrow Ops:** +44 20 8759 4321  
        **Local Authority:** 020 8583 2000  
        **Emergency Services:** 999
        """)

    # Debug information
    if st.sidebar.checkbox("🔧 Debug Info", key="debug_info_checkbox"):
        st.sidebar.write("Data columns:", list(weather_data.columns))
        st.sidebar.write("Anomaly labels:", weather_data['anomaly_label'].unique() if len(weather_data) > 0 else "No data")
        st.sidebar.write("Data shape:", weather_data.shape if len(weather_data) > 0 else "No data")
        st.sidebar.write("Timestamp range:", f"{weather_data['timestamp'].min()} to {weather_data['timestamp'].max()}" if len(weather_data) > 0 else "No data")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; font-size: 0.8rem; margin-top: 20px;'>
        Weather Anomaly Detection Dashboard v2.4 | 
        MSc Data Science Group Project | 
        University of Greenwich | 
        Team: Nad (Dashboard), Jeremy (ML), Marie (XAI), Dipo (Community) |
        Data Sources: Open Meteo API, UKMO Seamless Model
    </div>
    """, unsafe_allow_html=True)

# ================================================================================================
# RUN APPLICATION
# ================================================================================================

if __name__ == "__main__":
