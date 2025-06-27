import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="Agricultural Commodity Price Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    div[data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    .st-emotion-cache-16txtl3 h1 {
        font-weight: 700;
        color: #2c3e50;
    }
    .card {
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .price-highlight {
        font-weight: bold;
        color: #2980b9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f1f1;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #2980b9;
        color: white;
    }
    div.block-container {
        padding-top: 2rem;
    }
    .metric-container {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 15px;
        text-align: center;
    }
    .metric-title {
        font-size: 0.8rem;
        color: #718096;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
    }
    .trend-up {
        color: #38a169;
    }
    .trend-down {
        color: #e53e3e;
    }
    .upload-section {
        border: 2px dashed #2980b9;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .download-button {
        display: inline-block;
        padding: 10px 15px;
        background-color: #2980b9;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
    }
    .download-button:hover {
        background-color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Add a custom title with icon
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <h1 style="margin: 0;">🌾 Agricultural Commodity Price Analysis</h1>
</div>
""", unsafe_allow_html=True)

# File upload section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📤 Upload Your Data")
st.markdown("Upload a CSV file with your agricultural commodity price data.")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
st.markdown('</div>', unsafe_allow_html=True)

# Function to load data from uploaded file
def load_uploaded_data(file):
    df = pd.read_csv(file)
    
    # Convert date columns to datetime if they exist
    date_columns = [col for col in df.columns if any(date_term in col.lower() for date_term in ['date', 'day', 'month', 'year', 'time'])]
    
    if date_columns:
        for col in date_columns:
            try:
                # Try different date formats
                for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']:
                    try:
                        df[col] = pd.to_datetime(df[col], format=fmt)
                        break
                    except:
                        continue
                # If specific formats fail, use the default parser
                if not pd.api.types.is_datetime64_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                st.warning(f"Could not convert {col} to date format. Using as-is.")
    
    return df

# Main dashboard layout
if uploaded_file is not None:
    # Load the data
    df = load_uploaded_data(uploaded_file)
    
    # Identify key columns
    st.markdown("### 🔑 Map Your Data Columns")
    st.markdown("Please identify the key columns in your dataset:")
    
    # Detect numeric columns for price data
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Detect potential date columns
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col])]
    if not date_cols:
        date_cols = [col for col in df.columns if any(date_term in col.lower() for date_term in ['date', 'day', 'month', 'year', 'time'])]
    
    # Detect potential location columns
    potential_location_cols = [col for col in df.columns if any(loc_term in col.lower() for loc_term in 
                                                         ['state', 'district', 'region', 'county', 'province', 
                                                          'city', 'town', 'village', 'market', 'location', 'area'])]
    
    # Detect potential commodity columns
    potential_commodity_cols = [col for col in df.columns if any(comm_term in col.lower() for comm_term in 
                                                           ['commodity', 'product', 'crop', 'item', 'vegetable', 
                                                            'fruit', 'grain', 'produce', 'name', 'variety'])]
    
    # Column selection
    col1, col2 = st.columns(2)
    
    with col1:
        # Date column
        date_column = st.selectbox(
            "Select date column:",
            options=['None'] + df.columns.tolist(),
            index=0 if not date_cols else df.columns.tolist().index(date_cols[0]) + 1
        )
        
        # Price columns
        price_columns = st.multiselect(
            "Select price columns (min, modal, max, etc.):",
            options=numeric_cols,
            default=[col for col in numeric_cols if any(price_term in col.lower() for price_term in ['price', 'cost', 'value', 'rate', 'min', 'max', 'modal', 'avg', 'average'])]
        )
        
    with col2:
        # Location columns
        location_columns = st.multiselect(
            "Select location columns (state, district, market, etc.):",
            options=df.columns.tolist(),
            default=potential_location_cols
        )
        
        # Commodity column
        commodity_column = st.selectbox(
            "Select commodity/product column:",
            options=['None'] + df.columns.tolist(),
            index=0 if not potential_commodity_cols else df.columns.tolist().index(potential_commodity_cols[0]) + 1
        )
    
    # Process data based on selected columns
    if date_column != 'None':
        if not pd.api.types.is_datetime64_dtype(df[date_column]):
            try:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            except:
                st.warning(f"Could not convert {date_column} to date format. Some time-based analyses may not work correctly.")
    
    # If we have valid selections, proceed with the dashboard
    if date_column != 'None' and len(price_columns) > 0:
        # Sidebar filters
        with st.sidebar:
            st.markdown("## 🔍 Filter Data")
            st.markdown("---")
            
            # Date range selector if date column is available
            if date_column != 'None':
                st.markdown("### 📅 Date Range")
                try:
                    date_range = st.date_input(
                        "Select date range",
                        [df[date_column].min(), df[date_column].max()],
                        df[date_column].min(),
                        df[date_column].max()
                    )
                    
                    # Convert to datetime for filtering
                    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                    
                    # Filter data by date
                    filtered_df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
                except:
                    st.warning("Date filtering may not be accurate due to date conversion issues.")
                    filtered_df = df.copy()
            else:
                filtered_df = df.copy()
                
            st.markdown("---")
            
            # Drop-down filters for location columns
            if location_columns:
                st.markdown("### 📊 Location Filters")
                
                for location_col in location_columns:
                    options = ['All'] + sorted(filtered_df[location_col].unique().tolist())
                    selected_location = st.selectbox(f'Select {location_col}', options)
                    
                    # Filter by selection
                    if selected_location != 'All':
                        filtered_df = filtered_df[filtered_df[location_col] == selected_location]
            
            # Commodity filter if commodity column is available
            if commodity_column != 'None':
                st.markdown("### 🌽 Commodity Filter")
                commodity_options = ['All'] + sorted(filtered_df[commodity_column].unique().tolist())
                selected_commodity = st.selectbox('Select Commodity', commodity_options)
                
                # Filter by commodity
                if selected_commodity != 'All':
                    filtered_df = filtered_df[filtered_df[commodity_column] == selected_commodity]
                    
            st.markdown("---")
            
            # Show data stats
            st.markdown("### 📈 Data Statistics")
            st.write(f"*Total Records:* {len(filtered_df)}")
            if date_column != 'None':
                try:
                    st.write(f"*Date Range:* {filtered_df[date_column].min().strftime('%Y-%m-%d')} to {filtered_df[date_column].max().strftime('%Y-%m-%d')}")
                except:
                    st.write("*Date Range:* Unable to determine")
            
            # Download option
            st.markdown("---")
            st.markdown("### 📥 Download Filtered Data")
            
            # Convert dataframe to CSV for download
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv" class="download-button">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

        # Main content area - tabbed layout
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Analysis", "🌍 Market Compare", "📈 Trend Analysis", "📋 Data View"])
        
        # Tab 1: Price Analysis
        with tab1:
            if price_columns:
                col1, col2, col3 = st.columns([1,1,1])
                
                # Key metrics (dynamic based on selected price columns)
                for i, col in enumerate(price_columns[:3]):  # Show up to 3 metrics
                    with [col1, col2, col3][i % 3]:
                        avg_price = filtered_df[col].mean()
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-title">Average {col}</div>
                            <div class="metric-value">{avg_price:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("### 📊 Price Distribution Analysis")
                
                # Box plot of price distribution
                fig_box = go.Figure()
                
                # Add box plots for price columns
                for col in price_columns:
                    fig_box.add_trace(go.Box(
                        y=filtered_df[col],
                        name=col,
                        boxmean=True
                    ))
                
                fig_box.update_layout(
                    title='Price Distribution (Box Plot)',
                    yaxis_title='Price',
                    template='plotly_white',
                    boxmode='group',
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Add violin plot of price distribution
                st.markdown("### 🎻 Price Distribution Density")
                
                # Violin plot for price distributions
                fig_violin = go.Figure()
                
                for col in price_columns:
                    fig_violin.add_trace(go.Violin(
                        y=filtered_df[col],
                        name=col,
                        box_visible=True,
                        meanline_visible=True,
                        opacity=0.6
                    ))
                
                fig_violin.update_layout(
                    title='Price Distribution Density (Violin Plot)',
                    yaxis_title='Price',
                    template='plotly_white',
                    violinmode='group',
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                st.plotly_chart(fig_violin, use_container_width=True)
                
                # Histogram of prices
                st.markdown("### 📊 Price Frequency Distribution")
                
                # Create histograms
                fig_hist = go.Figure()
                
                for col in price_columns:
                    fig_hist.add_trace(go.Histogram(
                        x=filtered_df[col],
                        name=col,
                        opacity=0.7,
                        nbinsx=30
                    ))
                
                fig_hist.update_layout(
                    title='Price Frequency Distribution',
                    xaxis_title='Price',
                    yaxis_title='Frequency',
                    template='plotly_white',
                    barmode='overlay',
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Price range analysis for each commodity (if commodity column is available)
                if commodity_column != 'None' and filtered_df[commodity_column].nunique() > 1:
                    st.markdown("### 💰 Commodity Price Range Analysis")
                    
                    # Group by commodity and calculate price statistics
                    agg_dict = {price_col: 'mean' for price_col in price_columns}
                    commodity_price_range = filtered_df.groupby(commodity_column).agg(agg_dict).reset_index()
                    
                    # Sort by first price column descending
                    commodity_price_range = commodity_price_range.sort_values(price_columns[0], ascending=False)
                    
                    # Horizontal bar chart for price ranges
                    fig_range = px.bar(
                        commodity_price_range,
                        y=commodity_column,
                        x=price_columns,
                        orientation='h',
                        barmode='group',
                        title='Commodity Price Analysis'
                    )
                    
                    fig_range.update_layout(
                        xaxis_title='Price',
                        template='plotly_white',
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    
                    st.plotly_chart(fig_range, use_container_width=True)
            else:
                st.warning("Please select at least one price column to see price analysis.")
        
        # Tab 2: Market Comparison
        with tab2:
            st.markdown("### 🌍 Market Comparison")
            
            # Determine what columns we can use for comparison
            comparison_options = []
            
            if location_columns:
                comparison_options.extend(location_columns)
            
            if commodity_column != 'None':
                comparison_options.append(commodity_column)
            
            if comparison_options:
                # Select what to compare
                compare_by = st.selectbox(
                    "Compare by:",
                    comparison_options,
                    index=0 if len(comparison_options) > 0 else 0
                )
                
                if len(price_columns) > 0:
                    # Group data by the selected comparison
                    agg_dict = {price_col: 'mean' for price_col in price_columns}
                    grouped = filtered_df.groupby(compare_by).agg(agg_dict).reset_index()
                    
                    # Sort by first price column
                    grouped = grouped.sort_values(price_columns[0], ascending=False)
                    
                    # Limit to top 15 if there are more than 15
                    if len(grouped) > 15:
                        grouped = grouped.head(15)
                    
                    # Create a grouped bar chart
                    fig_compare = go.Figure()
                    
                    for price_col in price_columns:
                        fig_compare.add_trace(go.Bar(
                            x=grouped[compare_by],
                            y=grouped[price_col],
                            name=price_col,
                            opacity=0.8
                        ))
                    
                    fig_compare.update_layout(
                        title=f'Average Prices by {compare_by}',
                        xaxis_title=compare_by,
                        yaxis_title='Price',
                        barmode='group',
                        template='plotly_white',
                        height=600,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    
                    st.plotly_chart(fig_compare, use_container_width=True)
                    
                    # Price spread analysis (if we have at least two price columns)
                    if len(price_columns) >= 2:
                        st.markdown("### 💹 Price Spread Analysis")
                        
                        # Calculate price spread between highest and lowest price columns
                        max_price_col = price_columns[0]
                        min_price_col = price_columns[0]
                        
                        # Find the columns with highest and lowest average values
                        for col in price_columns:
                            if filtered_df[col].mean() > filtered_df[max_price_col].mean():
                                max_price_col = col
                            if filtered_df[col].mean() < filtered_df[min_price_col].mean():
                                min_price_col = col
                        
                        # If they're different, calculate spread
                        if max_price_col != min_price_col:
                            grouped['Price_Spread'] = grouped[max_price_col] - grouped[min_price_col]
                            grouped['Spread_Percentage'] = (grouped['Price_Spread'] / grouped[min_price_col]) * 100
                            
                            # Create a horizontal bar chart for spread analysis
                            fig_spread = go.Figure()
                            
                            fig_spread.add_trace(go.Bar(
                                y=grouped[compare_by],
                                x=grouped['Price_Spread'],
                                orientation='h',
                                marker=dict(
                                    color=grouped['Spread_Percentage'],
                                    colorscale='RdYlGn_r',
                                    colorbar=dict(title='Spread %')
                                ),
                                text=grouped['Spread_Percentage'].round(1).astype(str) + '%',
                                textposition='auto',
                            ))
                            
                            fig_spread.update_layout(
                                title=f'Price Spread Analysis by {compare_by} ({max_price_col} - {min_price_col})',
                                xaxis_title='Price Spread',
                                yaxis_title=compare_by,
                                template='plotly_white',
                                height=600,
                                margin=dict(l=20, r=20, t=40, b=20),
                            )
                            
                            st.plotly_chart(fig_spread, use_container_width=True)
                    
                    # Bubble chart comparing price values
                    if len(price_columns) >= 2:
                        st.markdown("### 🔮 Price Relationship Analysis")
                        
                        # Select up to 3 price columns for the bubble chart
                        display_cols = price_columns[:min(3, len(price_columns))]
                        
                        if len(display_cols) == 2:
                            fig_bubble = px.scatter(
                                grouped,
                                x=display_cols[0],
                                y=display_cols[1],
                                size=grouped[display_cols[0]] + grouped[display_cols[1]],
                                hover_name=compare_by,
                                text=compare_by,
                                color=compare_by,
                                size_max=50,
                            )
                        else:  # 3 or more display columns
                            fig_bubble = px.scatter(
                                grouped,
                                x=display_cols[0],
                                y=display_cols[1],
                                size=display_cols[2],
                                hover_name=compare_by,
                                text=compare_by,
                                color=compare_by,
                                size_max=50,
                            )
                        
                        fig_bubble.update_layout(
                            title=f'Price Relationship Analysis by {compare_by}',
                            xaxis_title=display_cols[0],
                            yaxis_title=display_cols[1],
                            template='plotly_white',
                            height=600,
                            margin=dict(l=20, r=20, t=40, b=20),
                        )
                        
                        st.plotly_chart(fig_bubble, use_container_width=True)
                else:
                    st.warning("Please select at least one price column to see market comparison.")
            else:
                st.warning("Please select location or commodity columns to enable market comparison.")
            
        # Tab 3: Trend Analysis
        with tab3:
            if date_column != 'None' and price_columns:
                st.markdown("### 📈 Price Trend Analysis Over Time")
                
                # Group by date and calculate average prices
                agg_dict = {price_col: 'mean' for price_col in price_columns}
                time_trend = filtered_df.groupby(date_column).agg(agg_dict).reset_index()
                
                # Line chart for price trends over time
                fig_trend = go.Figure()
                
                for col in price_columns:
                    fig_trend.add_trace(go.Scatter(
                        x=time_trend[date_column],
                        y=time_trend[col],
                        mode='lines+markers',
                        name=col,
                        line=dict(width=2),
                        marker=dict(size=7)
                    ))
                
                fig_trend.update_layout(
                    title='Price Trends Over Time',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_white',
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Price volatility analysis
                st.markdown("### 📊 Price Volatility Analysis")
                
                # Calculate day-to-day price changes for the first price column
                main_price_col = price_columns[0]
                time_trend[f'{main_price_col}_Change'] = time_trend[main_price_col].diff()
                
                # Drop the first row with NaN change values
                time_trend_volatility = time_trend.dropna()
                
                # Create bar chart for price changes
                fig_volatility = go.Figure()
                
                fig_volatility.add_trace(go.Bar(
                    x=time_trend_volatility[date_column],
                    y=time_trend_volatility[f'{main_price_col}_Change'],
                    name=f'{main_price_col} Change',
                    marker_color=time_trend_volatility[f'{main_price_col}_Change'].apply(
                        lambda x: 'green' if x >= 0 else 'red'
                    )
                ))
                
                fig_volatility.update_layout(
                    title=f'Day-to-Day Price Volatility ({main_price_col})',
                    xaxis_title='Date',
                    yaxis_title='Price Change',
                    template='plotly_white',
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                st.plotly_chart(fig_volatility, use_container_width=True)
                
                # Commodity-specific trends if commodity column is available
                if commodity_column != 'None' and filtered_df[commodity_column].nunique() > 1:
                    st.markdown("### 🔍 Commodity-Specific Trends")
                    
                    # Select commodities to compare
                    all_commodities = filtered_df[commodity_column].unique().tolist()
                    if len(all_commodities) > 10:
                        default_commodities = all_commodities[:5]
                    else:
                        default_commodities = all_commodities
                        
                    selected_commodities = st.multiselect(
                        'Select items to compare:',
                        options=all_commodities,
                        default=default_commodities[:min(5, len(default_commodities))]
                    )
                    
                    if selected_commodities:
                        # Filter data for selected commodities
                        commodity_trend_data = filtered_df[filtered_df[commodity_column].isin(selected_commodities)]
                        
                        # Group by date and commodity for the main price column
                        main_price_col = price_columns[0]
                        commodity_trends = commodity_trend_data.groupby([date_column, commodity_column]).agg({
                            main_price_col: 'mean'
                        }).reset_index()
                        
                        # Create line chart for commodity price trends
                        fig_commodity_trends = px.line(
                            commodity_trends,
                            x=date_column,
                            y=main_price_col,
                            color=commodity_column,
                            title=f'{main_price_col} Trends by {commodity_column}',
                            labels={main_price_col: f'{main_price_col}', date_column: 'Date'},
                            template='plotly_white',
                        )
                        
                        fig_commodity_trends.update_layout(
                            height=600,
                            margin=dict(l=20, r=20, t=40, b=20),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_commodity_trends, use_container_width=True)
                
                # Heatmap of price correlations
                if len(price_columns) > 1:
                    st.markdown("### 🔥 Price Correlation Heatmap")
                    
                    # Calculate correlation matrix
                    corr_matrix = filtered_df[price_columns].corr()
                    
                    # Create heatmap
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='Viridis',
                        zmin=-1,
                        zmax=1,
                        text=corr_matrix.values.round(2),
                        texttemplate='%{text}',
                        textfont=dict(color='white'),
                    ))
                    
                    fig_heatmap.update_layout(
                        title='Price Correlation Heatmap',
                        template='plotly_white',
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                if date_column == 'None':
                    st.warning("Please select a date column to see trend analysis.")
                else:
                    st.warning("Please select at least one price column to see trend analysis.")
        
        # Tab 4: Data View
        with tab4:
            st.markdown("### 📋 Data Table View")
            
            # Column selection
            st.markdown("*Select columns to display:*")
            all_columns = filtered_df.columns.tolist()
            selected_columns = st.multiselect(
                "Choose columns",
                options=all_columns,
                default=all_columns[:min(len(all_columns), 10)]
            )
            
            if selected_columns:
                # Display data with selected columns
                st.dataframe(
                    filtered_df[selected_columns],
                    height=500,
                    use_container_width=True
                )
            
            # Summary statistics for numeric columns
             # Summary statistics for numeric columns
            st.markdown("### 📊 Data Summary Statistics")
            
            # Get numeric columns
            numeric_columns = filtered_df.select_dtypes(include=['number']).columns
            
            if len(numeric_columns) > 0:
                # Create summary statistics
                stats_df = filtered_df[numeric_columns].describe().transpose()
                stats_df = stats_df.round(2)
                
                # Add additional statistics
                stats_df['range'] = stats_df['max'] - stats_df['min']
                stats_df['cv'] = (stats_df['std'] / stats_df['mean']) * 100  # coefficient of variation
                
                # Display the statistics
                st.dataframe(
                    stats_df,
                    height=400,
                    use_container_width=True
                )
            else:
                st.warning("No numeric columns available for summary statistics.")
            
            # Data quality information
            st.markdown("### 🧹 Data Quality")
            
            # Check for missing values
            missing_values = filtered_df.isnull().sum()
            missing_percent = (missing_values / len(filtered_df)) * 100
            
            # Create a dataframe with missing value information
            missing_df = pd.DataFrame({
                'Missing Values': missing_values,
                'Percentage': missing_percent.round(2)
            })
            
            # Filter to show only columns with missing values
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            
            if not missing_df.empty:
                st.warning("The dataset contains missing values:")
                st.dataframe(
                    missing_df,
                    height=min(300, len(missing_df) * 35 + 38),
                    use_container_width=True
                )
            else:
                st.success("No missing values detected in the dataset.")
            
            # Data distribution for selected columns
            st.markdown("### 📊 Data Distribution")
            
            if numeric_columns.size > 0:
                # Select a column for distribution analysis
                dist_column = st.selectbox(
                    "Select a column to view distribution:",
                    options=numeric_columns
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    hist_fig = px.histogram(
                        filtered_df, 
                        x=dist_column,
                        nbins=30,
                        marginal="box",
                        title=f"Distribution of {dist_column}"
                    )
                    hist_fig.update_layout(
                        xaxis_title=dist_column,
                        yaxis_title="Count",
                        template="plotly_white"
                    )
                    st.plotly_chart(hist_fig, use_container_width=True)
                
                with col2:
                    # QQ Plot for normality check
                    from scipy import stats
                    
                    # Get values and remove NaN
                    values = filtered_df[dist_column].dropna()
                    
                    # Calculate theoretical quantiles
                    qq_fig = go.Figure()
                    
                    if len(values) > 1:  # Need at least 2 points for QQ plot
                        # Calculate QQ data
                        qq_data = stats.probplot(values, dist="norm")
                        
                        # Add scatter points
                        qq_fig.add_trace(go.Scatter(
                            x=qq_data[0][0],
                            y=qq_data[0][1],
                            mode='markers',
                            marker=dict(color='blue'),
                            name='Data'
                        ))
                        
                        # Add line
                        qq_fig.add_trace(go.Scatter(
                            x=qq_data[0][0],
                            y=qq_data[0][0] * qq_data[1][0] + qq_data[1][1],
                            mode='lines',
                            line=dict(color='red'),
                            name='Normal'
                        ))
                        
                        qq_fig.update_layout(
                            title=f"Q-Q Plot for {dist_column}",
                            xaxis_title="Theoretical Quantiles",
                            yaxis_title="Sample Quantiles",
                            template="plotly_white"
                        )
                        st.plotly_chart(qq_fig, use_container_width=True)
                    else:
                        st.warning("Not enough data for Q-Q plot.")
            else:
                st.warning("No numeric columns available for distribution analysis.")

    else:
        st.warning("Please select a date column and at least one price column to generate the dashboard.")
else:
    # Sample data display
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample data section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🔍 How to Use This Dashboard")
    
    st.markdown("""
    This interactive dashboard allows you to analyze agricultural commodity price data. 
    
    Follow these steps to get started:
    
    1. *Upload your data* using the file uploader at the top
    2. *Map your data columns* to date, price, location, and commodity fields
    3. *Use the filters* in the sidebar to focus on specific time periods, locations, or commodities
    4. *Explore the different analysis tabs* to gain insights into your price data
    
    ### Expected Data Format
    
    Your CSV file should contain:
    
    - A date/time column
    - One or more price columns (min, max, modal, etc.)
    - Location information (optional but recommended)
    - Commodity/product names (optional but recommended)
    
    ### Example Data Structure
    """)
    
    # Create example dataframe
    example_data = {
        'Date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
        'State': ['Karnataka', 'Maharashtra', 'Punjab', 'Karnataka', 'Maharashtra'],
        'Market': ['Bangalore', 'Mumbai', 'Amritsar', 'Bangalore', 'Mumbai'],
        'Commodity': ['Rice', 'Wheat', 'Rice', 'Rice', 'Wheat'],
        'Min_Price': [1500, 2000, 1800, 1520, 2050],
        'Modal_Price': [1650, 2200, 1950, 1680, 2150],
        'Max_Price': [1800, 2400, 2100, 1850, 2300]
    }
    example_df = pd.DataFrame(example_data)
    
    # Display example data
    st.dataframe(example_df, use_container_width=True)
    
    st.markdown("""
    ### Features Available
    
    - *Price Analysis*: Visualize price distributions with box plots, violin plots, and histograms
    - *Market Comparison*: Compare prices across different markets, states, or commodities
    - *Trend Analysis*: Track price trends over time and analyze volatility
    - *Data View*: Examine raw data and summary statistics
    
    ### Need Help?
    
    If you encounter any issues or have questions, please reach out to our support team.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample visualizations to showcase capabilities
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 📈 Dashboard Preview")
    
    # Create sample data for visualization
    dates = pd.date_range(start='2023-01-01', periods=30)
    
    sample_data = []
    commodities = ['Rice', 'Wheat', 'Maize']
    markets = ['Market A', 'Market B', 'Market C']
    
    np.random.seed(42)
    
    for date in dates:
        for commodity in commodities:
            for market in markets:
                base_price = 0
                if commodity == 'Rice':
                    base_price = 2000
                elif commodity == 'Wheat':
                    base_price = 1800
                else:
                    base_price = 1500
                
                # Add some random variations and trends
                noise = np.random.normal(0, 100)
                trend = (date - dates[0]).days * 5  # Upward trend
                
                min_price = base_price + noise + trend - 150
                modal_price = base_price + noise + trend
                max_price = base_price + noise + trend + 150
                
                sample_data.append({
                    'Date': date,
                    'Commodity': commodity,
                    'Market': market,
                    'Min_Price': min_price,
                    'Modal_Price': modal_price,
                    'Max_Price': max_price
                })
    
    sample_df = pd.DataFrame(sample_data)
    
    # Sample visualization 1: Price trends
    st.markdown("### Sample Price Trends")
    
    # Group by date and commodity
    sample_trend = sample_df.groupby(['Date', 'Commodity']).agg({'Modal_Price': 'mean'}).reset_index()
    
    fig_sample = px.line(
        sample_trend, 
        x='Date', 
        y='Modal_Price', 
        color='Commodity',
        title="Sample Price Trend Analysis",
        labels={'Modal_Price': 'Price', 'Date': 'Date'},
        template="plotly_white"
    )
    
    fig_sample.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig_sample, use_container_width=True)
    
    # Sample visualization 2: Market comparison
    st.markdown("### Sample Market Comparison")
    
    # Group by market and commodity
    sample_market = sample_df.groupby(['Market', 'Commodity']).agg({
        'Min_Price': 'mean',
        'Modal_Price': 'mean',
        'Max_Price': 'mean'
    }).reset_index()
    
    fig_market = px.bar(
        sample_market,
        x='Market',
        y='Modal_Price',
        color='Commodity',
        barmode='group',
        title="Sample Market Comparison",
        labels={'Modal_Price': 'Price', 'Market': 'Market'},
        template="plotly_white"
    )
    
    fig_market.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig_market, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid #e0e0e0;">
    <p>Developed for Agricultural Commodity Price Analysis | © 2025</p>
</div>
""", unsafe_allow_html=True)