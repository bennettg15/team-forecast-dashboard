import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Team Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Team Forecast Data Visualization Dashboard")
st.markdown("Upload your team forecast data and create interactive visualizations")

# Sidebar for controls
st.sidebar.header("📁 Data Upload & Controls")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload your team forecast data in CSV format"
)

# Main content area
if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        
        # Display basic info about the dataset
        st.success(f"✅ File uploaded successfully! Dataset contains {len(df)} rows and {len(df.columns)} columns.")
        
        # Data preview section
        st.header("🔍 Data Preview")
        
        # Show basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        with col4:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
        
        # Display the dataframe
        st.subheader("📋 Raw Data")
        
        # Add option to filter/search the dataframe
        search_term = st.text_input("🔍 Search in data (optional)")
        if search_term:
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            filtered_df = df[mask]
            st.write(f"Showing {len(filtered_df)} rows matching '{search_term}'")
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
        
        # Data types and info
        with st.expander("📊 Data Types & Info"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Column Data Types:**")
                st.dataframe(df.dtypes.to_frame('Data Type'))
            with col2:
                st.write("**Missing Values:**")
                missing_data = df.isnull().sum().to_frame('Missing Count')
                missing_data['Missing %'] = (missing_data['Missing Count'] / len(df) * 100).round(2)
                st.dataframe(missing_data)
        
        # Visualization section
        st.header("📈 Visualizations")
        
        # Get numeric and categorical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Try to identify potential date columns in object type
        potential_date_cols = []
        for col in categorical_columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(), errors='raise')
                    potential_date_cols.append(col)
                except:
                    pass
        
        if potential_date_cols:
            date_columns.extend(potential_date_cols)
            for col in potential_date_cols:
                categorical_columns.remove(col)
        
        # Sidebar controls for visualizations
        st.sidebar.subheader("🎨 Visualization Controls")
        
        if numeric_columns:
            # Chart type selection
            chart_types = ["Line Chart", "Bar Chart", "Scatter Plot", "Box Plot", "Histogram", "Heatmap"]
            selected_chart = st.sidebar.selectbox("Select Chart Type", chart_types)
            
            # Create visualizations based on selection
            if selected_chart == "Line Chart" and date_columns:
                st.subheader("📈 Line Chart")
                x_col = st.selectbox("Select X-axis (Date/Time)", date_columns)
                y_cols = st.multiselect("Select Y-axis (Numeric)", numeric_columns, default=numeric_columns[:3])
                
                if x_col and y_cols:
                    # Convert to datetime if needed
                    df_plot = df.copy()
                    if x_col in potential_date_cols:
                        df_plot[x_col] = pd.to_datetime(df_plot[x_col])
                    
                    fig = go.Figure()
                    for col in y_cols:
                        fig.add_trace(go.Scatter(
                            x=df_plot[x_col], 
                            y=df_plot[col], 
                            mode='lines+markers',
                            name=col
                        ))
                    fig.update_layout(
                        title=f"Forecast Trends Over Time",
                        xaxis_title=x_col,
                        yaxis_title="Values",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif selected_chart == "Bar Chart":
                st.subheader("📊 Bar Chart")
                if categorical_columns and numeric_columns:
                    x_col = st.selectbox("Select Category Column", categorical_columns)
                    y_col = st.selectbox("Select Numeric Column", numeric_columns)
                    
                    if x_col and y_col:
                        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif selected_chart == "Scatter Plot":
                st.subheader("🎯 Scatter Plot")
                if len(numeric_columns) >= 2:
                    x_col = st.selectbox("Select X-axis", numeric_columns)
                    y_col = st.selectbox("Select Y-axis", [col for col in numeric_columns if col != x_col])
                    color_col = st.selectbox("Color by (optional)", [None] + categorical_columns)
                    
                    if x_col and y_col:
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                                       title=f"{y_col} vs {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif selected_chart == "Box Plot":
                st.subheader("📦 Box Plot")
                if categorical_columns and numeric_columns:
                    cat_col = st.selectbox("Select Category Column", categorical_columns)
                    num_col = st.selectbox("Select Numeric Column", numeric_columns)
                    
                    if cat_col and num_col:
                        fig = px.box(df, x=cat_col, y=num_col, title=f"{num_col} Distribution by {cat_col}")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif selected_chart == "Histogram":
                st.subheader("📊 Histogram")
                num_col = st.selectbox("Select Numeric Column", numeric_columns)
                bins = st.slider("Number of Bins", 10, 100, 30)
                
                if num_col:
                    fig = px.histogram(df, x=num_col, nbins=bins, title=f"Distribution of {num_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif selected_chart == "Heatmap":
                st.subheader("🔥 Correlation Heatmap")
                if len(numeric_columns) >= 2:
                    corr_matrix = df[numeric_columns].corr()
                    fig = px.imshow(corr_matrix, 
                                  title="Correlation Matrix",
                                  color_continuous_scale='RdBu_r',
                                  aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        if numeric_columns:
            st.subheader("📊 Summary Statistics")
            st.dataframe(df[numeric_columns].describe(), use_container_width=True)
        
        # Download processed data
        st.subheader("💾 Download Data")
        
        # Option to download filtered data
        if search_term and 'filtered_df' in locals():
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name=f"filtered_forecast_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Full Data as CSV",
                data=csv,
                file_name=f"forecast_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.info("Please make sure your file is a valid CSV format.")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a CSV file to get started")
    
    # Show example of expected data format
    st.subheader("📋 Expected Data Format")
    st.write("Your CSV file should contain team forecast data. Here's an example structure:")
    
    # Create example dataframe
    example_data = {
        'Date': pd.date_range('2024-01-01', periods=10, freq='M'),
        'Team': ['Sales', 'Marketing', 'Engineering', 'Sales', 'Marketing', 'Engineering', 'Sales', 'Marketing', 'Engineering', 'Sales'],
        'Forecast_Revenue': np.random.randint(50000, 200000, 10),
        'Actual_Revenue': np.random.randint(45000, 190000, 10),
        'Target_Revenue': np.random.randint(60000, 210000, 10),
        'Confidence_Level': np.random.uniform(0.7, 0.95, 10).round(2)
    }
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df, use_container_width=True)
    
    # Tips for data preparation
    st.subheader("💡 Data Preparation Tips")
    st.markdown("""
    - Ensure your CSV has column headers in the first row
    - Date columns should be in a standard format (YYYY-MM-DD, MM/DD/YYYY, etc.)
    - Numeric data should not contain special characters (except decimal points)
    - Missing values are okay - they'll be identified in the analysis
    - Include team names, forecast values, actual values, and dates for best results
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Plotly")