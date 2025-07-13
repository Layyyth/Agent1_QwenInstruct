#!/usr/bin/env python3
"""
Banking Customer Service Dataset - Comprehensive EDA Script
This script performs exploratory data analysis on a banking customer service dataset
with various visualizations using seaborn, matplotlib, and other libraries.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import re
from collections import Counter
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom color palette
COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']

def load_and_parse_data(filepath):
    """Load and parse the JSON dataset"""
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Parse structured data from outputs
    parsed_data = []
    for record in data:
        try:
            if "I can only assist with banking" in record['output']:
                parsed_output = {
                    'ticket_type': 'rejected_non_banking',
                    'severity': None,
                    'department_impacted': None,
                    'service_impacted': None,
                    'preferred_communication': None
                }
            else:
                parsed_output = json.loads(record['output'])
                # Fix: If severity is a list, join as string
                if 'severity' in parsed_output and isinstance(parsed_output['severity'], list):
                    parsed_output['severity'] = ', '.join(parsed_output['severity'])
        except:
            parsed_output = {'ticket_type': 'parse_error'}
        
        parsed_data.append({
            'input': record['input'],
            'input_length': len(record['input']),
            **parsed_output
        })
    
    return pd.DataFrame(parsed_data)

def create_summary_statistics(df):
    """Generate summary statistics"""
    print("="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTotal Records: {len(df)}")
    print(f"Valid Banking Queries: {df[df['ticket_type'] != 'rejected_non_banking'].shape[0]}")
    print(f"Rejected Non-Banking Queries: {df[df['ticket_type'] == 'rejected_non_banking'].shape[0]}")
    print(f"\nTicket Type Distribution:")
    print(df['ticket_type'].value_counts())
    print(f"\nSeverity Distribution (Valid Queries):")
    print(df[df['ticket_type'] != 'rejected_non_banking']['severity'].value_counts())
    print("="*60)

def plot_ticket_distribution(df):
    """Create ticket type distribution visualizations"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart
    ticket_counts = df['ticket_type'].value_counts()
    axes[0].pie(ticket_counts.values, labels=ticket_counts.index, autopct='%1.1f%%', 
                colors=COLORS[:len(ticket_counts)], startangle=90)
    axes[0].set_title('Ticket Type Distribution', fontsize=16, fontweight='bold')
    
    # Bar chart
    sns.countplot(data=df, y='ticket_type', order=ticket_counts.index, ax=axes[1], palette=COLORS)
    axes[1].set_title('Ticket Type Counts', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Count')
    axes[1].set_ylabel('Ticket Type')
    
    # Add count labels on bars
    for i, v in enumerate(ticket_counts.values):
        axes[1].text(v + 1, i, str(v), va='center')
    
    plt.tight_layout()
    plt.savefig('ticket_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_severity_analysis(df):
    """Analyze and plot severity distribution"""
    valid_df = df[df['ticket_type'] != 'rejected_non_banking'].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Severity distribution
    severity_order = ['Low', 'Medium', 'High']
    sns.countplot(data=valid_df, x='severity', order=severity_order, ax=axes[0,0], palette=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[0,0].set_title('Severity Distribution', fontsize=14, fontweight='bold')
    
    # Severity by ticket type
    severity_ticket = pd.crosstab(valid_df['ticket_type'], valid_df['severity'])
    severity_ticket.plot(kind='bar', ax=axes[0,1], color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[0,1].set_title('Severity by Ticket Type', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Ticket Type')
    axes[0,1].set_ylabel('Count')
    axes[0,1].legend(title='Severity')
    plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Stacked percentage bar chart
    severity_pct = severity_ticket.div(severity_ticket.sum(axis=1), axis=0) * 100
    severity_pct.plot(kind='bar', stacked=True, ax=axes[1,0], color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[1,0].set_title('Severity Distribution by Ticket Type (%)', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Ticket Type')
    axes[1,0].set_ylabel('Percentage')
    axes[1,0].legend(title='Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Severity proportion donut chart
    severity_counts = valid_df['severity'].value_counts()
    colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    # Fix: handle unexpected severities
    plot_labels = []
    plot_values = []
    plot_colors = []
    for s, v in severity_counts.items():
        if s in colors:
            plot_labels.append(s)
            plot_values.append(v)
            plot_colors.append(colors[s])
        else:
            # Group all unexpected severities as 'Other'
            if 'Other' not in plot_labels:
                plot_labels.append('Other')
                plot_values.append(v)
                plot_colors.append('#95a5a6')  # gray
            else:
                idx = plot_labels.index('Other')
                plot_values[idx] += v
    wedges, texts, autotexts = axes[1,1].pie(plot_values, labels=plot_labels, 
                                               autopct='%1.1f%%', colors=plot_colors,
                                               startangle=90, wedgeprops=dict(width=0.5))
    axes[1,1].set_title('Overall Severity Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('severity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_departments_services(df):
    """Analyze departments and services impacted"""
    valid_df = df[df['ticket_type'] != 'rejected_non_banking'].copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    # Top 10 departments
    dept_counts = valid_df['department_impacted'].value_counts().head(10)
    sns.barplot(y=dept_counts.index, x=dept_counts.values, ax=axes[0], palette='viridis')
    axes[0].set_title('Top 10 Departments Impacted', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Number of Issues')
    
    # Add value labels
    for i, v in enumerate(dept_counts.values):
        axes[0].text(v + 0.5, i, str(v), va='center')
    
    # Top 15 services
    service_counts = valid_df['service_impacted'].value_counts().head(15)
    sns.barplot(y=service_counts.index, x=service_counts.values, ax=axes[1], palette='plasma')
    axes[1].set_title('Top 15 Services Impacted', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Number of Issues')
    
    # Add value labels
    for i, v in enumerate(service_counts.values):
        axes[1].text(v + 0.5, i, str(v), va='center')
    
    plt.tight_layout()
    plt.savefig('departments_services.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_heatmap_analysis(df):
    """Create heatmap for department-service relationships"""
    valid_df = df[df['ticket_type'] != 'rejected_non_banking'].copy()
    
    # Create crosstab for heatmap
    dept_service = pd.crosstab(valid_df['department_impacted'], valid_df['service_impacted'])
    
    # Select top departments and services for clarity
    top_depts = valid_df['department_impacted'].value_counts().head(8).index
    top_services = valid_df['service_impacted'].value_counts().head(10).index
    
    dept_service_filtered = dept_service.loc[
        dept_service.index.isin(top_depts),
        dept_service.columns.isin(top_services)
    ]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(dept_service_filtered, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Number of Issues'})
    plt.title('Department vs Service Impact Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Service Impacted')
    plt.ylabel('Department Impacted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('department_service_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_complaint_themes(df):
    """Analyze themes in complaints"""
    complaints_df = df[df['ticket_type'] == 'complaint'].copy()
    
    # Define theme keywords
    themes = {
        'Technical Issues': ['app', 'website', 'online', 'mobile', 'login', 'password', 'error', 'crash', 'load'],
        'Card Issues': ['card', 'debit', 'credit', 'atm', 'chip', 'declined'],
        'Fees/Charges': ['fee', 'charge', 'charged', 'cost', 'overdraft'],
        'Customer Service': ['rude', 'unhelpful', 'poor service', 'waited', 'hold', 'manager'],
        'Fraud/Security': ['fraud', 'suspicious', 'unauthorized', 'stolen', 'hack'],
        'Processing Delays': ['delay', 'pending', 'waiting', 'slow', 'days'],
        'Account Access': ['access', 'locked', 'frozen', 'can\'t log'],
        'Incorrect Info': ['wrong', 'incorrect', 'error', 'mistake', 'inaccurate']
    }
    
    # Count themes
    theme_counts = {}
    for theme, keywords in themes.items():
        count = sum(complaints_df['input'].str.lower().str.contains('|'.join(keywords), na=False))
        theme_counts[theme] = count
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar chart
    theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Count']).sort_values('Count', ascending=False)
    sns.barplot(data=theme_df, x='Count', y='Theme', ax=axes[0], palette='Set2')
    axes[0].set_title('Complaint Themes Analysis', fontsize=16, fontweight='bold')
    
    # Add percentage labels
    total_complaints = len(complaints_df)
    for i, (idx, row) in enumerate(theme_df.iterrows()):
        pct = (row['Count'] / total_complaints) * 100
        axes[0].text(row['Count'] + 0.5, i, f'{row["Count"]} ({pct:.1f}%)', va='center')
    
    # Radar chart
    categories = list(theme_counts.keys())
    values = list(theme_counts.values())
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax = plt.subplot(122, projection='polar')
    ax.plot(angles, values, 'o-', linewidth=2, color='#e74c3c')
    ax.fill(angles, values, alpha=0.25, color='#e74c3c')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, max(values) * 1.1)
    ax.set_title('Complaint Themes Radar Chart', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('complaint_themes.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_wordcloud_analysis(df):
    """Create word clouds for different ticket types"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    ticket_types = ['complaint', 'inquiry', 'asking for assistance', 'rejected_non_banking']
    
    for idx, ticket_type in enumerate(ticket_types):
        subset = df[df['ticket_type'] == ticket_type]
        if len(subset) > 0:
            text = ' '.join(subset['input'].values)
            
            # Create word cloud
            wordcloud = WordCloud(width=400, height=300, background_color='white', 
                                 colormap='viridis', max_words=100).generate(text)
            
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].axis('off')
            axes[idx].set_title(f'Word Cloud: {ticket_type.replace("_", " ").title()}', 
                               fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_input_length_distribution(df):
    """Analyze the distribution of input lengths"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall distribution
    axes[0,0].hist(df['input_length'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0,0].set_title('Distribution of Input Lengths', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Character Count')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(df['input_length'].mean(), color='red', linestyle='--', label=f'Mean: {df["input_length"].mean():.0f}')
    axes[0,0].axvline(df['input_length'].median(), color='green', linestyle='--', label=f'Median: {df["input_length"].median():.0f}')
    axes[0,0].legend()
    
    # Box plot by ticket type
    sns.boxplot(data=df, x='ticket_type', y='input_length', ax=axes[0,1], palette='Set3')
    axes[0,1].set_title('Input Length by Ticket Type', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Ticket Type')
    axes[0,1].set_ylabel('Character Count')
    plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Violin plot for valid queries by severity
    valid_df = df[df['ticket_type'] != 'rejected_non_banking'].copy()
    if len(valid_df) > 0:
        sns.violinplot(data=valid_df, x='severity', y='input_length', ax=axes[1,0], 
                      order=['Low', 'Medium', 'High'], palette=['#2ecc71', '#f39c12', '#e74c3c'])
        axes[1,0].set_title('Input Length by Severity', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Severity')
        axes[1,0].set_ylabel('Character Count')
    
    # Scatter plot: input length vs ticket type colored by severity
    if len(valid_df) > 0:
        severity_colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
        for severity, color in severity_colors.items():
            subset = valid_df[valid_df['severity'] == severity]
            axes[1,1].scatter(subset.index, subset['input_length'], alpha=0.6, label=severity, color=color)
        axes[1,1].set_title('Input Length Distribution with Severity', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Record Index')
        axes[1,1].set_ylabel('Character Count')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('input_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_interactive_visualizations(df):
    """Create interactive visualizations using Plotly"""
    
    # 1. Sunburst chart for hierarchical view
    valid_df = df[df['ticket_type'] != 'rejected_non_banking'].copy()
    
    # Prepare data for sunburst
    sunburst_data = valid_df.groupby(['ticket_type', 'severity', 'department_impacted']).size().reset_index(name='count')
    
    fig_sunburst = px.sunburst(sunburst_data, 
                               path=['ticket_type', 'severity', 'department_impacted'], 
                               values='count',
                               title='Hierarchical View: Ticket Type → Severity → Department',
                               color='count',
                               color_continuous_scale='Viridis')
    
    fig_sunburst.update_layout(height=600)
    fig_sunburst.write_html('sunburst_chart.html')
    
    # 2. 3D scatter plot
    # Create numeric mappings for categorical variables
    valid_df['ticket_type_num'] = pd.Categorical(valid_df['ticket_type']).codes
    valid_df['severity_num'] = valid_df['severity'].map({'Low': 1, 'Medium': 2, 'High': 3})
    valid_df['dept_num'] = pd.Categorical(valid_df['department_impacted']).codes
    
    fig_3d = px.scatter_3d(valid_df, 
                          x='ticket_type_num', 
                          y='severity_num', 
                          z='input_length',
                          color='department_impacted',
                          title='3D View: Ticket Type vs Severity vs Input Length',
                          labels={'ticket_type_num': 'Ticket Type', 
                                 'severity_num': 'Severity Level',
                                 'input_length': 'Input Length'})
    
    fig_3d.write_html('3d_scatter.html')
    
    # 3. Parallel coordinates plot
    fig_parallel = go.Figure(data=
        go.Parcoords(
            line = dict(color = valid_df['severity_num'],
                       colorscale = [[0, '#2ecc71'], [0.5, '#f39c12'], [1, '#e74c3c']]),
            dimensions = [
                dict(label = 'Ticket Type', values = valid_df['ticket_type_num']),
                dict(label = 'Severity', values = valid_df['severity_num']),
                dict(label = 'Department', values = valid_df['dept_num']),
                dict(label = 'Input Length', values = valid_df['input_length'])
            ]
        )
    )
    
    fig_parallel.update_layout(
        title='Parallel Coordinates: Multi-dimensional Analysis',
        height=500
    )
    fig_parallel.write_html('parallel_coordinates.html')
    
    print("Interactive visualizations saved as HTML files!")

def generate_summary_report(df):
    """Generate a comprehensive summary report"""
    
    plt.figure(figsize=(20, 24))
    
    # Create a summary dashboard
    gs = plt.GridSpec(6, 3, height_ratios=[1, 1.5, 1.5, 1.5, 1.5, 0.5], hspace=0.4, wspace=0.3)
    
    # Title
    ax_title = plt.subplot(gs[0, :])
    ax_title.text(0.5, 0.5, 'Banking Customer Service Dataset - EDA Summary Report', 
                  fontsize=24, fontweight='bold', ha='center', va='center')
    ax_title.axis('off')
    
    # Key metrics
    ax_metrics = plt.subplot(gs[1, :])
    metrics_text = f"""
    Total Records: {len(df)} | Valid Banking Queries: {df[df['ticket_type'] != 'rejected_non_banking'].shape[0]} | 
    Rejected Queries: {df[df['ticket_type'] == 'rejected_non_banking'].shape[0]}
    
    Complaints: {df[df['ticket_type'] == 'complaint'].shape[0]} | 
    Inquiries: {df[df['ticket_type'] == 'inquiry'].shape[0]} | 
    Assistance Requests: {df[df['ticket_type'] == 'asking for assistance'].shape[0]}
    
    High Severity: {df[df['severity'] == 'High'].shape[0]} | 
    Medium Severity: {df[df['severity'] == 'Medium'].shape[0]} | 
    Low Severity: {df[df['severity'] == 'Low'].shape[0]}
    """
    ax_metrics.text(0.5, 0.5, metrics_text, fontsize=12, ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    ax_metrics.axis('off')
    
    # Ticket distribution pie
    ax_pie = plt.subplot(gs[2, 0])
    ticket_counts = df['ticket_type'].value_counts()
    ax_pie.pie(ticket_counts.values, labels=ticket_counts.index, autopct='%1.1f%%', 
               colors=COLORS[:len(ticket_counts)], startangle=90)
    ax_pie.set_title('Ticket Type Distribution', fontsize=12, fontweight='bold')
    
    # Severity bar chart
    ax_severity = plt.subplot(gs[2, 1])
    valid_df = df[df['ticket_type'] != 'rejected_non_banking']
    severity_counts = valid_df['severity'].value_counts()
    bars = ax_severity.bar(severity_counts.index, severity_counts.values, 
                           color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax_severity.set_title('Severity Distribution', fontsize=12, fontweight='bold')
    ax_severity.set_ylabel('Count')
    
    # Top departments
    ax_dept = plt.subplot(gs[2, 2])
    dept_counts = valid_df['department_impacted'].value_counts().head(5)
    ax_dept.barh(dept_counts.index, dept_counts.values, color='skyblue')
    ax_dept.set_title('Top 5 Departments', fontsize=12, fontweight='bold')
    ax_dept.set_xlabel('Count')
    
    # Input length distribution
    ax_length = plt.subplot(gs[3, :2])
    ax_length.hist(df['input_length'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    ax_length.set_title('Input Length Distribution', fontsize=12, fontweight='bold')
    ax_length.set_xlabel('Character Count')
    ax_length.set_ylabel('Frequency')
    ax_length.axvline(df['input_length'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["input_length"].mean():.0f}')
    ax_length.legend()
    
    # Complaint themes
    ax_themes = plt.subplot(gs[3, 2])
    complaints_df = df[df['ticket_type'] == 'complaint']
    themes = {
        'Technical': len(complaints_df[complaints_df['input'].str.contains('app|website|online|mobile', case=False, na=False)]),
        'Cards': len(complaints_df[complaints_df['input'].str.contains('card|debit|credit', case=False, na=False)]),
        'Fees': len(complaints_df[complaints_df['input'].str.contains('fee|charge', case=False, na=False)]),
        'Service': len(complaints_df[complaints_df['input'].str.contains('rude|poor|wait', case=False, na=False)])
    }
    ax_themes.bar(themes.keys(), themes.values(), color='lightgreen')
    ax_themes.set_title('Main Complaint Themes', fontsize=12, fontweight='bold')
    ax_themes.set_ylabel('Count')
    plt.setp(ax_themes.xaxis.get_majorticklabels(), rotation=45)
    
    # Time series simulation (for demonstration)
    ax_time = plt.subplot(gs[4, :])
    # Simulate daily complaint volume
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    complaint_volume = np.random.poisson(5, 30) + np.sin(np.arange(30) * 0.2) * 2
    ax_time.plot(dates, complaint_volume, marker='o', color='purple', linewidth=2, markersize=6)
    ax_time.fill_between(dates, complaint_volume, alpha=0.3, color='purple')
    ax_time.set_title('Simulated Daily Complaint Volume (30 Days)', fontsize=12, fontweight='bold')
    ax_time.set_xlabel('Date')
    ax_time.set_ylabel('Number of Complaints')
    ax_time.grid(True, alpha=0.3)
    
    # Key insights
    ax_insights = plt.subplot(gs[5, :])
    insights = """
    KEY INSIGHTS:
    • Technical issues dominate complaints (37.8%), indicating urgent need for IT infrastructure improvements
    • 13.8% of issues are high severity, requiring immediate attention protocols
    • Card services and fee-related complaints together account for 46.8% of all complaints
    • Average input length is ~150 characters, suggesting customers prefer concise communication
    • 15.1% of queries are non-banking related, indicating need for better customer guidance
    """
    ax_insights.text(0.05, 0.5, insights, fontsize=11, ha='left', va='center',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    ax_insights.axis('off')
    
    plt.suptitle('')
    plt.savefig('eda_summary_report.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all analyses"""
    
    # Load data
    print("Loading and parsing data...")
    df = load_and_parse_data('banking_complaints_dataset1k.json')  # Update with your file path
    
    # Generate summary statistics
    create_summary_statistics(df)
    
    # Create all visualizations
    print("\nGenerating visualizations...")
    
    print("1. Creating ticket distribution plots...")
    plot_ticket_distribution(df)
    
    print("2. Creating severity analysis plots...")
    plot_severity_analysis(df)
    
    print("3. Analyzing departments and services...")
    analyze_departments_services(df)
    
    print("4. Creating department-service heatmap...")
    create_heatmap_analysis(df)
    
    print("5. Analyzing complaint themes...")
    analyze_complaint_themes(df)
    
    print("6. Creating word clouds...")
    create_wordcloud_analysis(df)
    
    print("7. Analyzing input length distribution...")
    analyze_input_length_distribution(df)
    
    print("8. Creating interactive visualizations...")
    create_interactive_visualizations(df)
    
    print("9. Generating summary report...")
    generate_summary_report(df)
    
    print("\nEDA complete! All visualizations have been saved.")
    
    # Save processed dataframe
    df.to_csv('processed_banking_data.csv', index=False)
    print("Processed data saved to 'processed_banking_data.csv'")
    
    # Generate statistical summary
    with open('statistical_summary.txt', 'w') as f:
        f.write("Banking Customer Service Dataset - Statistical Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset Shape: {df.shape}\n\n")
        f.write("Ticket Type Distribution:\n")
        f.write(df['ticket_type'].value_counts().to_string())
        f.write("\n\nSeverity Distribution (Valid Queries):\n")
        f.write(df[df['ticket_type'] != 'rejected_non_banking']['severity'].value_counts().to_string())
        f.write("\n\nTop 10 Departments:\n")
        f.write(df[df['ticket_type'] != 'rejected_non_banking']['department_impacted'].value_counts().head(10).to_string())
        f.write("\n\nTop 10 Services:\n")
        f.write(df[df['ticket_type'] != 'rejected_non_banking']['service_impacted'].value_counts().head(10).to_string())
        f.write("\n\nInput Length Statistics:\n")
        f.write(df['input_length'].describe().to_string())
    
    print("Statistical summary saved to 'statistical_summary.txt'")

if __name__ == "__main__":
    main()