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
def load_data():
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
                if col in df.columns:
                    # Handle various formats: remove $, commas, and convert to float
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', '').str.replace('$', '').str.strip(), 
                        errors='coerce'
                    )
            
            # Check for any conversion issues
            if len(numeric_columns) > 0 and df[numeric_columns].isnull().any().any():
                st.warning("Some numeric values couldn't be converted. Please check your data format.")
                problematic_cols = df[numeric_columns].columns[df[numeric_columns].isnull().any()].tolist()
                st.write("Columns with issues:", problematic_cols)
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    else:
        st.info("No file uploaded. Using sample data for demonstration.")
        return generate_sample_data()

# Function to transform data into long format for week-by-week analysis
def transform_to_weekly_data(df, week_cols):
    """Transform wide format data to long format for weekly analysis"""
    weekly_data = []
    
    for _, row in df.iterrows():
        for week_col in week_cols:
            # Extract week number from column name (e.g., "Week 1 Q4 Forecast" -> "Week 1")
            week_parts = week_col.split()
            if len(week_parts) >= 2:
                week_name = f"{week_parts[0]} {week_parts[1]}"
            else:
                week_name = week_col
                
            weekly_data.append({
                'Team Name': row['Team Name'],
                'Category': row['Category'],
                'Budget': row['Budget'] if 'Budget' in row else 0,
                'Week': week_name,
                'Forecast': float(row[week_col]) if pd.notnull(row[week_col]) else 0,
                'Over/(Under)': (float(row[week_col]) if pd.notnull(row[week_col]) else 0) - (row['Budget'] if 'Budget' in row else 0),
                'Budget Variance %': ((float(row[week_col]) if pd.notnull(row[week_col]) else 0) - (row['Budget'] if 'Budget' in row else 0)) / (row['Budget'] if 'Budget' in row and row['Budget'] != 0 else 1) * 100
            })
    
    return pd.DataFrame(weekly_data)

# Load data
df = load_data()

# Show data preview if file uploaded
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    # Show data preview
    with st.expander("📋 Data Preview"):
        st.write("First 5 rows of your uploaded data:")
        if df is not None:
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
    
    # Transform data to weekly format
    weekly_df = transform_to_weekly_data(df, week_cols)
    
    # Extract available weeks for filtering
    available_weeks = sorted(weekly_df['Week'].unique())
    
    # Add filters in sidebar
    with st.sidebar:
        st.header("Filters")
        
        # Week filter - NEW!
        selected_weeks = st.multiselect(
            "Select Weeks",
            available_weeks,
            default=available_weeks,
            help="Select one or more weeks to analyze"
        )
        
        # Team filter
        available_teams = weekly_df['Team Name'].unique()
        selected_teams = st.multiselect(
            "Select Teams",
            available_teams,
            default=available_teams[:3] if len(available_teams) > 3 else available_teams
        )
        
        # Category filter
        available_categories = weekly_df['Category'].unique()
        selected_categories = st.multiselect(
            "Select Cost Categories",
            available_categories,
            default=available_categories
        )
    
    # Filter data based on selections
    if len(selected_weeks) > 0 and len(selected_teams) > 0 and len(selected_categories) > 0:
        filtered_df = weekly_df[
            (weekly_df['Week'].isin(selected_weeks)) & 
            (weekly_df['Team Name'].isin(selected_teams)) & 
            (weekly_df['Category'].isin(selected_categories))
        ].copy()
    else:
        filtered_df = weekly_df.copy()
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your selection.")
        st.stop()
    
    # Main dashboard layout - Updated metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            if len(selected_weeks) == 1:
                total_forecast = float(filtered_df['Forecast'].sum())
                st.metric("Total Forecast", f"${total_forecast:,.0f}", help=f"Total for {selected_weeks[0]}")
            else:
                avg_forecast = float(filtered_df.groupby('Week')['Forecast'].sum().mean())
                st.metric("Avg Weekly Forecast", f"${avg_forecast:,.0f}", help=f"Average across {len(selected_weeks)} weeks")
        except:
            st.metric("Total Forecast", "Error in calculation")
    
    with col2:
        try:
            if len(selected_weeks) == 1:
                total_budget = float(filtered_df['Budget'].sum())
                st.metric("Total Budget", f"${total_budget:,.0f}", help=f"Budget for {selected_weeks[0]}")
            else:
                avg_budget = float(filtered_df.groupby('Week')['Budget'].sum().mean())
                st.metric("Avg Weekly Budget", f"${avg_budget:,.0f}", help=f"Average across {len(selected_weeks)} weeks")
        except:
            st.metric("Total Budget", "Error in calculation")
    
    with col3:
        try:
            if len(selected_weeks) == 1:
                total_variance = float(filtered_df['Over/(Under)'].sum())
                st.metric("Budget Variance", f"${total_variance:,.0f}", help=f"Variance for {selected_weeks[0]}")
            else:
                avg_variance = float(filtered_df.groupby('Week')['Over/(Under)'].sum().mean())
                st.metric("Avg Weekly Variance", f"${avg_variance:,.0f}", help=f"Average across {len(selected_weeks)} weeks")
        except:
            st.metric("Budget Variance", "Error in calculation")
    
    with col4:
        try:
            if len(selected_weeks) == 1:
                week_data = filtered_df[filtered_df['Week'] == selected_weeks[0]]
                total_budget = float(week_data['Budget'].sum())
                total_variance = float(week_data['Over/(Under)'].sum())
                variance_pct = (total_variance / total_budget * 100) if total_budget != 0 else 0
                st.metric("Variance %", f"{variance_pct:.1f}%", help=f"Percentage for {selected_weeks[0]}")
            else:
                weekly_variance_pcts = []
                for week in selected_weeks:
                    week_data = filtered_df[filtered_df['Week'] == week]
                    week_budget = float(week_data['Budget'].sum())
                    week_variance = float(week_data['Over/(Under)'].sum())
                    if week_budget != 0:
                        weekly_variance_pcts.append(week_variance / week_budget * 100)
                avg_variance_pct = np.mean(weekly_variance_pcts) if weekly_variance_pcts else 0
                st.metric("Avg Variance %", f"{avg_variance_pct:.1f}%", help=f"Average across {len(selected_weeks)} weeks")
        except:
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
        
        if len(selected_weeks) > 1:
            # Show trends across multiple weeks
            # Team trends
            team_weekly = filtered_df.groupby(['Team Name', 'Week'])['Forecast'].sum().reset_index()
            fig_weekly_team = px.line(
                team_weekly,
                x='Week', y='Forecast', color='Team Name',
                title="Weekly Forecast Trends by Team",
                markers=True
            )
            fig_weekly_team.update_layout(height=400)
            st.plotly_chart(fig_weekly_team, use_container_width=True)
            
            # Category trends
            cat_weekly = filtered_df.groupby(['Category', 'Week'])['Forecast'].sum().reset_index()
            fig_weekly_cat = px.line(
                cat_weekly,
                x='Week', y='Forecast', color='Category',
                title="Weekly Forecast Trends by Category",
                markers=True
            )
            fig_weekly_cat.update_layout(height=400)
            st.plotly_chart(fig_weekly_cat, use_container_width=True)
        else:
            st.info(f"Select multiple weeks to see trends. Currently showing data for {selected_weeks[0] if selected_weeks else 'no weeks'} only.")
            
            # Show single week breakdown
            if selected_weeks:
                single_week_data = filtered_df[filtered_df['Week'] == selected_weeks[0]]
                
                # Team comparison for single week
                team_totals = single_week_data.groupby('Team Name')['Forecast'].sum().reset_index()
                fig_team_single = px.bar(
                    team_totals, x='Team Name', y='Forecast',
                    title=f"Team Forecast for {selected_weeks[0]}",
                    color='Forecast',
                    color_continuous_scale='viridis'
                )
                fig_team_single.update_layout(height=400)
                st.plotly_chart(fig_team_single, use_container_width=True)
    
    with tab2:
        st.subheader("Team Performance Comparison")
        
        if len(selected_weeks) > 1:
            # Multi-week comparison
            team_weekly_data = filtered_df.groupby(['Team Name', 'Week'])['Forecast'].sum().reset_index()
            fig_team_multi = px.bar(
                team_weekly_data, x='Team Name', y='Forecast', color='Week',
                title="Team Forecast Comparison Across Weeks",
                barmode='group'
            )
            fig_team_multi.update_layout(height=400)
            st.plotly_chart(fig_team_multi, use_container_width=True)
            
            # Variance by team and week
            variance_data = filtered_df.groupby(['Team Name', 'Week'])['Over/(Under)'].sum().reset_index()
            fig_variance_multi = px.bar(
                variance_data, x='Team Name', y='Over/(Under)', color='Week',
                title="Budget Variance by Team Across Weeks",
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_variance_multi.update_layout(height=400)
            st.plotly_chart(fig_variance_multi, use_container_width=True)
        else:
            # Single week analysis
            if selected_weeks:
                single_week_data = filtered_df[filtered_df['Week'] == selected_weeks[0]]
                
                # Team totals for single week
                team_totals = single_week_data.groupby('Team Name')['Forecast'].sum().reset_index()
                fig_team_bar = px.bar(
                    team_totals, x='Team Name', y='Forecast',
                    title=f"Team Forecast for {selected_weeks[0]}",
                    color='Forecast',
                    color_continuous_scale='viridis'
                )
                fig_team_bar.update_layout(height=400)
                st.plotly_chart(fig_team_bar, use_container_width=True)
                
                # Team variance for single week
                team_variance = single_week_data.groupby('Team Name')['Over/(Under)'].sum().reset_index()
                fig_variance = px.bar(
                    team_variance, x='Team Name', y='Over/(Under)',
                    title=f"Budget Variance by Team for {selected_weeks[0]}",
                    color='Over/(Under)',
                    color_continuous_scale='RdYlGn_r'
                )
                fig_variance.update_layout(height=400)
                st.plotly_chart(fig_variance, use_container_width=True)
    
    with tab3:
        st.subheader("Cost Category Analysis")
        
        col1, col2 = st.columns(2)
        
        if len(selected_weeks) == 1:
            # Single week analysis
            week_data = filtered_df[filtered_df['Week'] == selected_weeks[0]]
            category_totals = week_data.groupby('Category')['Forecast'].sum().reset_index()
            
            with col1:
                fig_pie = px.pie(
                    category_totals, values='Forecast', names='Category',
                    title=f"Forecast Distribution by Category - {selected_weeks[0]}"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_cat_bar = px.bar(
                    category_totals, x='Category', y='Forecast',
                    title=f"Forecast by Category - {selected_weeks[0]}",
                    color='Forecast',
                    color_continuous_scale='plasma'
                )
                fig_cat_bar.update_layout(xaxis={'tickangle': 45}, height=400)
                st.plotly_chart(fig_cat_bar, use_container_width=True)
        else:
            # Multi-week analysis
            with col1:
                # Average across weeks
                avg_category = filtered_df.groupby('Category')['Forecast'].mean().reset_index()
                fig_pie = px.pie(
                    avg_category, values='Forecast', names='Category',
                    title=f"Average Forecast Distribution by Category"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Week-by-week category comparison
                cat_week_data = filtered_df.groupby(['Category', 'Week'])['Forecast'].sum().reset_index()
                fig_cat_week = px.bar(
                    cat_week_data, x='Category', y='Forecast', color='Week',
                    title="Category Forecast Across Weeks",
                    barmode='group'
                )
                fig_cat_week.update_layout(xaxis={'tickangle': 45}, height=400)
                st.plotly_chart(fig_cat_week, use_container_width=True)
        
        # Heatmap - Team vs Category for selected weeks
        try:
            if len(selected_weeks) == 1:
                week_data = filtered_df[filtered_df['Week'] == selected_weeks[0]]
                pivot_data = week_data.pivot_table(
                    values='Forecast', 
                    index='Team Name', 
                    columns='Category', 
                    aggfunc='sum'
                ).fillna(0)
                title = f"Team vs Category Heatmap - {selected_weeks[0]}"
            else:
                # Average across selected weeks
                pivot_data = filtered_df.groupby(['Team Name', 'Category'])['Forecast'].mean().reset_index()
                pivot_data = pivot_data.pivot_table(
                    values='Forecast', 
                    index='Team Name', 
                    columns='Category', 
                    aggfunc='sum'
                ).fillna(0)
                title = f"Team vs Category Heatmap - Average Across Selected Weeks"
            
            if not pivot_data.empty:
                fig_heatmap = px.imshow(
                    pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    aspect="auto",
                    title=title,
                    color_continuous_scale='viridis'
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
        except Exception as e:
            st.write(f"Could not create heatmap: {e}")
    
    with tab4:
        st.subheader("Budget vs Forecast Analysis")
        
        if len(selected_weeks) == 1:
            # Single week budget vs forecast
            week_data = filtered_df[filtered_df['Week'] == selected_weeks[0]]
            budget_comparison = week_data.groupby('Team Name').agg({
                'Budget': 'sum',
                'Forecast': 'sum'
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
                y=budget_comparison['Forecast'],
                marker_color='darkblue'
            ))
            fig_budget.update_layout(
                title=f'Budget vs Forecast by Team - {selected_weeks[0]}',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_budget, use_container_width=True)
            
            # Variance by category for single week
            variance_data = week_data.groupby('Category').agg({
                'Over/(Under)': 'sum',
                'Budget': 'sum'
            }).reset_index()
            variance_data['Variance %'] = variance_data.apply(
                lambda row: (row['Over/(Under)'] / row['Budget']) * 100 if row['Budget'] != 0 else 0,
                axis=1
            )
            
            fig_variance_pct = px.bar(
                variance_data, x='Category', y='Variance %',
                title=f"Budget Variance % by Category - {selected_weeks[0]}",
                color='Variance %',
                color_continuous_scale='RdYlGn_r'
            )
            fig_variance_pct.update_layout(xaxis={'tickangle': 45}, height=400)
            st.plotly_chart(fig_variance_pct, use_container_width=True)
        else:
            # Multi-week budget vs forecast
            budget_weekly = filtered_df.groupby(['Team Name', 'Week']).agg({
                'Budget': 'sum',
                'Forecast': 'sum'
            }).reset_index()
            
            # Create subplot for budget vs forecast
            fig_budget_multi = make_subplots(rows=1, cols=2, 
                                           subplot_titles=('Weekly Budget by Team', 'Weekly Forecast by Team'))
            
            for team in budget_weekly['Team Name'].unique():
                team_data = budget_weekly[budget_weekly['Team Name'] == team]
                fig_budget_multi.add_trace(
                    go.Scatter(x=team_data['Week'], y=team_data['Budget'], 
                             mode='lines+markers', name=f'{team} Budget'),
                    row=1, col=1
                )
                fig_budget_multi.add_trace(
                    go.Scatter(x=team_data['Week'], y=team_data['Forecast'], 
                             mode='lines+markers', name=f'{team} Forecast'),
                    row=1, col=2
                )
            
            fig_budget_multi.update_layout(height=400, title_text="Budget vs Forecast Trends")
            st.plotly_chart(fig_budget_multi, use_container_width=True)
            
            # Weekly variance trends
            variance_weekly = filtered_df.groupby(['Team Name', 'Week'])['Over/(Under)'].sum().reset_index()
            fig_variance_trend = px.line(
                variance_weekly, x='Week', y='Over/(Under)', color='Team Name',
                title="Budget Variance Trends by Team",
                markers=True
            )
            fig_variance_trend.update_layout(height=400)
            st.plotly_chart(fig_variance_trend, use_container_width=True)
    
    with tab5:
        st.subheader("Detailed Data Table")
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            show_weekly_format = st.checkbox("Show weekly format (one row per team/category/week)", value=True)
        with col2:
            download_data = st.button("Download Filtered Data")
        
        if show_weekly_format:
            # Show weekly format data
            display_columns = ['Team Name', 'Category', 'Week', 'Budget', 'Forecast', 'Over/(Under)', 'Budget Variance %']
            display_df = filtered_df[display_columns].copy()
            
            # Format numeric columns
            numeric_columns = ['Budget', 'Forecast', 'Over/(Under)']
            for col in numeric_columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "")
            
            display_df['Budget Variance %'] = display_df['Budget Variance %'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
            
            st.dataframe(display_df, use_container_width=True)
        else:
            # Show summary by team and category (averaged across selected weeks)
            summary_df = filtered_df.groupby(['Team Name', 'Category']).agg({
                'Budget': 'first',  # Budget should be same across weeks
                'Forecast': 'mean',  # Average forecast across selected weeks
                'Over/(Under)': 'mean',  # Average variance
                'Budget Variance %': 'mean'  # Average percentage
            }).reset_index()
            
            # Format numeric columns
            numeric_columns = ['Budget', 'Forecast', 'Over/(Under)']
            for col in numeric_columns:
                summary_df[col] = summary_df[col].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "")
            
            summary_df['Budget Variance %'] = summary_df['Budget Variance %'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
            
            st.dataframe(summary_df, use_container_width=True)
            st.info(f"Showing average values across {len(selected_weeks)} selected weeks")
        
        if download_data:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"forecast_data_weeks_{'_'.join([w.replace(' ', '') for w in selected_weeks])}.csv",
                mime="text/csv"
            )
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    ### How to use this dashboard:
    1. **Download Sample**: Click "📋 Download Sample CSV Format" in the sidebar to get the correct file structure
    2. **Prepare Your Data**: Format your data to match the sample structure
    3. **Upload Data**: Use the sidebar to upload your CSV file with forecast data
    4. **Filter by Week**: Select specific weeks to analyze - compare week-to-week or focus on individual weeks
    5. **Filter by Team & Category**: Select specific teams and categories to focus your analysis
    6. **Explore Tabs**: 
       - **Weekly Trends**: See how forecasts change over time (requires multiple weeks selected)
       - **Team Comparison**: Compare performance across teams by week
       - **Category Analysis**: Understand spending by category and week
       - **Budget vs Forecast**: Analyze variance from budget by week
       - **Data Table**: View and download the week-by-week data
    
    **Key Feature**: Week-by-week analysis - data is never combined across weeks, allowing proper comparison and trend analysis.
    
    **Required CSV Columns**: 
    - `Team Name` - Name of the team
    - `Category` - Cost categories (HC Related, Contracted Srvcs, etc.)
    - `Budget` - Budget amount for each category
    - `Week X Q4 Forecast` - Weekly forecast columns (Week 1, Week 2, etc.)
    
    💡 **Tip**: Select multiple weeks to see trends, or single weeks for detailed analysis!
    """)

else:
    st.error("Unable to load data. Please check your file format.")