import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Team Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Team Forecast Analysis Dashboard")
st.markdown("Upload your team forecast data to visualize performance across weeks, teams, and cost categories.")

# Function to generate sample CSV format
def generate_sample_csv():
    """Generate a sample CSV file in the expected format"""
    sample_data = {
        'Team Name': ['Team Alpha', 'Team Alpha', 'Team Alpha', 'Team Beta', 'Team Beta', 'Team Beta'],
        'Category': ['HC Related', 'Contracted Srvcs', 'Software Spend', 'HC Related', 'Contracted Srvcs', 'Software Spend'],
        'Budget': [25000000, 4000000, 150000, 22000000, 3500000, 140000],
        'Week 1 Q4 Forecast': [24419000, 3884405, 141137, 21500000, 3200000, 135000],
        'Week 2 Q4 Forecast': [22421082, 3566590, 129389, 20800000, 3100000, 128000],
        'Week 3 Q4 Forecast': [23309045, 3707841, 134721, 21200000, 3250000, 132000],
        'Week 4 Q4 Forecast': [24640991, 3919717, 142420, 22100000, 3400000, 138000]
    }
    return pd.DataFrame(sample_data)

# Sample data generation function for demo purposes
def generate_sample_data():
    """Generate sample data similar to the provided screenshot"""
    teams = ["Team A", "Team B", "Team C", "Team D"]
    categories = ["HC Related", "Contracted Srvcs", "Software Spend", 
                 "Discretionary", "Depr. & Other", "Centralized Allocations"]
    weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
    
    data = []
    for team in teams:
        for category in categories:
            base_amount = np.random.randint(100000, 25000000)
            budget = base_amount * np.random.uniform(0.95, 1.1)
            
            row = {"Team Name": team, "Category": category, "Budget": budget}
            
            # Generate weekly forecasts with some variation
            for i, week in enumerate(weeks):
                weekly_amount = base_amount * np.random.uniform(0.85, 1.15)
                row[f"{week} Q4 Forecast"] = weekly_amount
            
            data.append(row)
    
    return pd.DataFrame(data)

# Function to process uploaded data or use sample data
def load_data(uploaded_file):
    """Load and process data from uploaded file or generate sample data"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Convert numeric columns to proper data types
            numeric_columns = []
            
            # Find week forecast columns
            week_cols = [col for col in df.columns if 'Week' in col and 'Forecast' in col]
            numeric_columns.extend(week_cols)
            
            # Add other numeric columns
            if 'Budget' in df.columns:
                numeric_columns.append('Budget')
            
            # Convert to numeric, handling any non-numeric values
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce')
            
            # Check for any conversion issues
            if df[numeric_columns].isnull().any().any():
                st.warning("Some numeric values couldn't be converted. Please check your data format.")
                st.write("Columns with issues:", df[numeric_columns].columns[df[numeric_columns].isnull().any()].tolist())
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    else:
        st.info("No file uploaded. Using sample data for demonstration.")
        return generate_sample_data()

# Sidebar for file upload and filters
with st.sidebar:
    st.header("Data Upload")
    
    # Sample file download
    st.subheader("📥 Download Sample Format")
    sample_df = generate_sample_csv()
    sample_csv = sample_df.to_csv(index=False)
    
    st.download_button(
        label="📋 Download Sample CSV Format",
        data=sample_csv,
        file_name="sample_forecast_format.csv",
        mime="text/csv",
        help="Download a sample CSV file to see the expected format"
    )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload your forecast data (CSV)",
        type=['csv'],
        help="Upload a CSV file with team forecast data"
    )

# Load data
df = load_data(uploaded_file)

# Show data preview if file is uploaded
if uploaded_file is not None and df is not None:
    with st.sidebar:
        # Show data preview
        with st.expander("📋 Data Preview"):
            st.write("First 5 rows of your uploaded data:")
            st.dataframe(df.head())
            st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            st.write("Column types:")
            st.write(df.dtypes)

if df is not None:
    # Data preprocessing
    # Identify week columns (columns containing 'Week' and 'Forecast')
    week_cols = [col for col in df.columns if 'Week' in col and 'Forecast' in col]
    
    if not week_cols:
        st.error("No week forecast columns found. Please ensure your data has columns like 'Week 1 Q4 Forecast'")
        st.stop()
    
    # Add filters in sidebar
    with st.sidebar:
        st.header("Filters")
        
        # Team filter
        available_teams = df['Team Name'].unique() if 'Team Name' in df.columns else []
        selected_teams = st.multiselect(
            "Select Teams",
            available_teams,
            default=available_teams[:3] if len(available_teams) > 3 else available_teams
        )
        
        # Category filter
        available_categories = df['Category'].unique() if 'Category' in df.columns else []
        selected_categories = st.multiselect(
            "Select Cost Categories",
            available_categories,
            default=available_categories
        )
    
    # Filter data based on selections
    filtered_df = df[
        (df['Team Name'].isin(selected_teams)) & 
        (df['Category'].isin(selected_categories))
    ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your selection.")
        st.stop()
    
    # Calculate total expenses and over/under budget
    try:
        filtered_df['Total Forecast'] = filtered_df[week_cols].sum(axis=1)
        if 'Budget' in filtered_df.columns:
            filtered_df['Over/(Under)'] = filtered_df['Total Forecast'] - filtered_df['Budget']
            # Handle division by zero
            filtered_df['Budget Variance %'] = filtered_df.apply(
                lambda row: (row['Over/(Under)'] / row['Budget'] * 100) if row['Budget'] != 0 else 0, 
                axis=1
            )
    except Exception as e:
        st.error(f"Error calculating totals: {e}")
        st.write("Please ensure all forecast and budget columns contain numeric values.")
        st.stop()
    
    # Main dashboard layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            total_forecast = filtered_df['Total Forecast'].sum()
            st.metric("Total Forecast", f"${total_forecast:,.0f}")
        except Exception as e:
            st.metric("Total Forecast", "Error in calculation")
    
    with col2:
        if 'Budget' in filtered_df.columns:
            try:
                total_budget = filtered_df['Budget'].sum()
                st.metric("Total Budget", f"${total_budget:,.0f}")
            except Exception as e:
                st.metric("Total Budget", "Error in calculation")
    
    with col3:
        if 'Over/(Under)' in filtered_df.columns:
            try:
                total_variance = filtered_df['Over/(Under)'].sum()
                st.metric("Budget Variance", f"${total_variance:,.0f}")
            except Exception as e:
                st.metric("Budget Variance", "Error in calculation")
    
    with col4:
        if 'Budget Variance %' in filtered_df.columns:
            try:
                total_budget = filtered_df['Budget'].sum()
                avg_variance_pct = (total_variance / total_budget * 100) if total_budget != 0 else 0
                st.metric("Variance %", f"{avg_variance_pct:.1f}%")
            except Exception as e:
                st.metric("Variance %", "Error in calculation")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Weekly Trends", 
        "👥 Team Comparison", 
        "📊 Category Analysis", 
        "💰 Budget vs Forecast", 
        "📋 Data Table"
    ])
    
    with tab1:
        st.subheader("Weekly Forecast Trends")
        
        # Prepare data for weekly trends
        weekly_data = []
        for _, row in filtered_df.iterrows():
            for week_col in week_cols:
                week_num = week_col.split()[1]  # Extract week number
                weekly_data.append({
                    'Team': row['Team Name'],
                    'Category': row['Category'],
                    'Week': f"Week {week_num}",
                    'Forecast': row[week_col]
                })
        
        weekly_df = pd.DataFrame(weekly_data)
        
        # Weekly trends by team
        fig_weekly_team = px.line(
            weekly_df.groupby(['Team', 'Week'])['Forecast'].sum().reset_index(),
            x='Week', y='Forecast', color='Team',
            title="Weekly Forecast by Team",
            markers=True
        )
        fig_weekly_team.update_layout(height=400)
        st.plotly_chart(fig_weekly_team, use_container_width=True)
        
        # Weekly trends by category
        fig_weekly_cat = px.line(
            weekly_df.groupby(['Category', 'Week'])['Forecast'].sum().reset_index(),
            x='Week', y='Forecast', color='Category',
            title="Weekly Forecast by Category",
            markers=True
        )
        fig_weekly_cat.update_layout(height=400)
        st.plotly_chart(fig_weekly_cat, use_container_width=True)
    
    with tab2:
        st.subheader("Team Performance Comparison")
        
        # Team total forecast comparison
        team_totals = filtered_df.groupby('Team Name')['Total Forecast'].sum().reset_index()
        fig_team_bar = px.bar(
            team_totals, x='Team Name', y='Total Forecast',
            title="Total Forecast by Team",
            color='Total Forecast',
            color_continuous_scale='viridis'
        )
        fig_team_bar.update_layout(height=400)
        st.plotly_chart(fig_team_bar, use_container_width=True)
        
        # Team budget variance
        if 'Over/(Under)' in filtered_df.columns:
            team_variance = filtered_df.groupby('Team Name')['Over/(Under)'].sum().reset_index()
            fig_variance = px.bar(
                team_variance, x='Team Name', y='Over/(Under)',
                title="Budget Variance by Team",
                color='Over/(Under)',
                color_continuous_scale='RdYlGn_r'
            )
            fig_variance.update_layout(height=400)
            st.plotly_chart(fig_variance, use_container_width=True)
    
    with tab3:
        st.subheader("Cost Category Analysis")
        
        # Category breakdown
        category_totals = filtered_df.groupby('Category')['Total Forecast'].sum().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                category_totals, values='Total Forecast', names='Category',
                title="Forecast Distribution by Category"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_cat_bar = px.bar(
                category_totals, x='Category', y='Total Forecast',
                title="Total Forecast by Category",
                color='Total Forecast',
                color_continuous_scale='plasma'
            )
            fig_cat_bar.update_xaxis(tickangle=45)
            st.plotly_chart(fig_cat_bar, use_container_width=True)
        
        # Heatmap of team vs category
        pivot_data = filtered_df.pivot_table(
            values='Total Forecast', 
            index='Team Name', 
            columns='Category', 
            aggfunc='sum'
        ).fillna(0)
        
        fig_heatmap = px.imshow(
            pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            aspect="auto",
            title="Team vs Category Heatmap",
            color_continuous_scale='viridis'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab4:
        st.subheader("Budget vs Forecast Analysis")
        
        if 'Budget' in filtered_df.columns:
            # Budget vs Forecast comparison
            budget_comparison = filtered_df.groupby('Team Name').agg({
                'Budget': 'sum',
                'Total Forecast': 'sum'
            }).reset_index()
            
            fig_budget = go.Figure()
            fig_budget.add_trace(go.Bar(
                name='Budget',
                x=budget_comparison['Team Name'],
                y=budget_comparison['Budget'],
                marker_color='lightblue'
            ))
            fig_budget.add_trace(go.Bar(
                name='Forecast',
                x=budget_comparison['Team Name'],
                y=budget_comparison['Total Forecast'],
                marker_color='darkblue'
            ))
            fig_budget.update_layout(
                title='Budget vs Forecast by Team',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_budget, use_container_width=True)
            
            # Variance analysis
            variance_data = filtered_df.groupby('Category').agg({
                'Over/(Under)': 'sum',
                'Budget': 'sum'
            }).reset_index()
            variance_data['Variance %'] = variance_data.apply(
                lambda row: (row['Over/(Under)'] / row['Budget'] * 100) if row['Budget'] != 0 else 0, 
                axis=1
            )
            
            fig_variance_pct = px.bar(
                variance_data, x='Category', y='Variance %',
                title="Budget Variance % by Category",
                color='Variance %',
                color_continuous_scale='RdYlGn_r'
            )
            fig_variance_pct.update_xaxis(tickangle=45)
            fig_variance_pct.update_layout(height=400)
            st.plotly_chart(fig_variance_pct, use_container_width=True)
        else:
            st.info("Budget data not available for comparison.")
    
    with tab5:
        st.subheader("Detailed Data Table")
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            show_all_columns = st.checkbox("Show all columns", value=False)
        with col2:
            download_data = st.button("Download Filtered Data")
        
        if show_all_columns:
            display_df = filtered_df
        else:
            # Show key columns
            key_columns = ['Team Name', 'Category'] + week_cols
            if 'Budget' in filtered_df.columns:
                key_columns.extend(['Budget', 'Total Forecast', 'Over/(Under)'])
            display_df = filtered_df[key_columns]
        
        # Format numeric columns
        numeric_columns = display_df.select_dtypes(include=[np.number]).columns
        display_df_formatted = display_df.copy()
        for col in numeric_columns:
            display_df_formatted[col] = display_df_formatted[col].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_df_formatted, use_container_width=True)
        
        if download_data:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="filtered_forecast_data.csv",
                mime="text/csv"
            )
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    ### How to use this dashboard:
    1. **Download Sample**: Click "📋 Download Sample CSV Format" in the sidebar to get the correct file structure
    2. **Prepare Your Data**: Format your data to match the sample structure
    3. **Upload Data**: Use the sidebar to upload your CSV file with forecast data
    4. **Filter**: Select specific teams and categories to focus your analysis
    5. **Explore Tabs**: 
       - **Weekly Trends**: See how forecasts change over time
       - **Team Comparison**: Compare performance across teams
       - **Category Analysis**: Understand spending by category
       - **Budget vs Forecast**: Analyze variance from budget
       - **Data Table**: View and download the raw data
    
    **Required CSV Columns**: 
    - `Team Name` - Name of the team
    - `Category` - Cost categories (HC Related, Contracted Srvcs, etc.)
    - `Budget` - Budget amount for each category
    - `Week X Q4 Forecast` - Weekly forecast columns (Week 1, Week 2, etc.)
    
    💡 **Tip**: Download the sample format file first to see exactly how your data should be structured!
    """)

else:
    st.error("Unable to load data. Please check your file format.")