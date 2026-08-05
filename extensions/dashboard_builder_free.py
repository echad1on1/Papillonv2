# extensions/dashboard_builder_free.py
import sys
import os
import sqlite3
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = "./results/extensions_results.db"

def build_dashboard_free():
    """Build and run the free dashboard."""
    
    # Load data
    if not os.path.exists(DB_PATH):
        print(f"No database found at {DB_PATH}")
        print("Please run some attacks first:")
        print("  python extensions/run_all_attacks_free.py --mock")
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM attacks", conn)
        conn.close()
        
        # Convert timestamp
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        df = pd.DataFrame()
    
    # Create app
    app = dash.Dash(__name__, title="Free Attack Dashboard")
    
    # Layout
    app.layout = html.Div([
        html.H1("Papillon Free Attack Dashboard", style={'text-align': 'center'}),
        
        # Summary metrics
        html.Div([
            html.Div([
                html.H3("Summary Statistics"),
                html.Div([
                    html.Div([
                        html.H4(f"{len(df)}"),
                        html.P("Total Attacks")
                    ], className="metric-box"),
                    html.Div([
                        html.H4(f"{df['success'].mean():.2%}" if not df.empty else "0%"),
                        html.P("Overall ASR")
                    ], className="metric-box"),
                    html.Div([
                        html.H4(f"{df['attack_name'].nunique() if not df.empty else 0}"),
                        html.P("Attack Types")
                    ], className="metric-box"),
                ], className="metrics-grid")
            ], className="metrics-container")
        ]),
        
        # Charts
        html.Div([
            html.Div([
                dcc.Graph(
                    id='asr-chart',
                    figure=px.bar(
                        df.groupby('attack_name')['success'].mean().reset_index(),
                        x='attack_name', y='success',
                        title="ASR by Attack Strategy",
                        labels={'success': 'ASR', 'attack_name': 'Attack Strategy'}
                    ) if not df.empty else go.Figure()
                )
            ], className="chart-container"),
            
            html.Div([
                dcc.Graph(
                    id='score-distribution',
                    figure=px.histogram(
                        df, x='roberta_score', color='attack_name',
                        title="Score Distribution by Attack",
                        nbins=20
                    ) if not df.empty else go.Figure()
                )
            ], className="chart-container"),
        ], className="charts-row"),
        
        # Recent attacks table
        html.Div([
            html.H3("Recent Attacks"),
            dash_table.DataTable(
                id='recent-table',
                columns=[
                    {"name": "Attack", "id": "attack_name"},
                    {"name": "Question", "id": "question"},
                    {"name": "Success", "id": "success"},
                    {"name": "Score", "id": "roberta_score"},
                    {"name": "Time", "id": "timestamp"}
                ] if not df.empty else [],
                data=df.tail(10).to_dict('records') if not df.empty else [],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{success} = 1'},
                        'backgroundColor': 'lightgreen'
                    },
                    {
                        'if': {'filter_query': '{success} = 0'},
                        'backgroundColor': 'lightcoral'
                    }
                ]
            )
        ], className="table-container"),
        
        # CSS
        html.Style("""
            .metrics-container {
                margin: 20px;
                padding: 20px;
                background-color: #f5f5f5;
                border-radius: 10px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin-top: 20px;
            }
            .metric-box {
                text-align: center;
                padding: 15px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-box h4 {
                margin: 0;
                font-size: 24px;
                color: #2196F3;
            }
            .metric-box p {
                margin: 5px 0 0 0;
                color: #666;
            }
            .charts-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px;
            }
            .chart-container {
                background-color: white;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .table-container {
                margin: 20px;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            @media (max-width: 768px) {
                .charts-row {
                    grid-template-columns: 1fr;
                }
                .metrics-grid {
                    grid-template-columns: 1fr;
                }
            }
        """)
    ])
    
    print("\n" + "="*60)
    print("FREE DASHBOARD RUNNING")
    print("="*60)
    print("Access at: http://127.0.0.1:8050")
    print("Note: Results shown are from free API testing")
    print("They may not match the paper's ASR exactly")
    print("but should show relative attack effectiveness")
    print("="*60)
    
    app.run_server(debug=True, port=8050)

if __name__ == "__main__":
    build_dashboard_free()