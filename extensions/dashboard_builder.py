# extensions/dashboard_builder.py
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

class DashboardBuilder:
    def __init__(self):
        self.db_path = DB_PATH
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load data from the database."""
        if not os.path.exists(self.db_path):
            print(f"No database found at {self.db_path}")
            self.data = pd.DataFrame()
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            self.data = pd.read_sql_query("SELECT * FROM attacks", conn)
            conn.close()
            
            # Convert timestamp to datetime
            if not self.data.empty and 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                
            print(f"Loaded {len(self.data)} records from database")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = pd.DataFrame()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate metrics from the data."""
        if self.data.empty:
            return {
                'total_attacks': 0,
                'overall_asr': 0,
                'by_attack': {},
                'by_language': {}
            }
        
        # Overall metrics
        total_attacks = len(self.data)
        overall_asr = self.data['success'].mean() if 'success' in self.data.columns else 0
        
        # Metrics by attack name
        by_attack = {}
        if 'attack_name' in self.data.columns:
            for attack in self.data['attack_name'].unique():
                attack_data = self.data[self.data['attack_name'] == attack]
                by_attack[attack] = {
                    'total': len(attack_data),
                    'success': attack_data['success'].sum() if 'success' in attack_data.columns else 0,
                    'asr': attack_data['success'].mean() if 'success' in attack_data.columns else 0
                }
        
        # Metrics by language (for multilingual attack)
        by_language = {}
        if 'language' in self.data.columns:
            for lang in self.data['language'].unique():
                lang_data = self.data[self.data['language'] == lang]
                by_language[lang] = {
                    'total': len(lang_data),
                    'success': lang_data['success'].sum() if 'success' in lang_data.columns else 0,
                    'asr': lang_data['success'].mean() if 'success' in lang_data.columns else 0
                }
        
        return {
            'total_attacks': total_attacks,
            'overall_asr': overall_asr,
            'by_attack': by_attack,
            'by_language': by_language
        }
    
    def create_dashboard(self):
        """Create and run the Dash dashboard."""
        app = dash.Dash(__name__, title="Papillon Attack Dashboard")
        
        # Update data periodically
        @app.callback(
            Output('metrics-summary', 'children'),
            Output('asr-chart', 'figure'),
            Output('attack-table', 'data'),
            Output('language-heatmap', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_dashboard(n):
            self.load_data()
            metrics = self.get_metrics()
            
            # Summary metrics
            summary = html.Div([
                html.H3("Overall Metrics"),
                html.Div([
                    html.Div([
                        html.H4(f"{metrics['total_attacks']}"),
                        html.P("Total Attacks")
                    ], className="metric-box"),
                    html.Div([
                        html.H4(f"{metrics['overall_asr']:.2%}"),
                        html.P("Overall ASR")
                    ], className="metric-box"),
                    html.Div([
                        html.H4(f"{len(metrics['by_attack'])}"),
                        html.P("Attack Types")
                    ], className="metric-box"),
                ], className="metrics-grid")
            ])
            
            # ASR by attack chart
            if metrics['by_attack']:
                attacks = list(metrics['by_attack'].keys())
                asrs = [metrics['by_attack'][a]['asr'] for a in attacks]
                totals = [metrics['by_attack'][a]['total'] for a in attacks]
                
                asr_fig = go.Figure(data=[
                    go.Bar(
                        name='ASR',
                        x=attacks,
                        y=asrs,
                        text=[f"{a:.2%}" for a in asrs],
                        textposition='auto',
                        marker_color='skyblue'
                    )
                ])
                asr_fig.update_layout(
                    title="ASR by Attack Strategy",
                    xaxis_title="Attack Strategy",
                    yaxis_title="ASR",
                    yaxis_tickformat=".0%"
                )
            else:
                asr_fig = go.Figure()
                asr_fig.add_annotation(text="No data available", x=0.5, y=0.5)
            
            # Attack table
            if not self.data.empty:
                table_data = self.data[['attack_name', 'question', 'success', 'roberta_score', 'timestamp']].tail(10).to_dict('records')
            else:
                table_data = []
            
            # Language heatmap
            if metrics['by_language']:
                languages = list(metrics['by_language'].keys())
                asrs = [metrics['by_language'][l]['asr'] for l in languages]
                
                lang_fig = go.Figure(data=go.Heatmap(
                    z=[asrs],
                    x=languages,
                    y=['ASR'],
                    colorscale='RdYlGn',
                    zmin=0,
                    zmax=1,
                    text=[[f"{a:.2%}" for a in asrs]],
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ))
                lang_fig.update_layout(
                    title="Language Performance Heatmap",
                    xaxis_title="Language",
                    yaxis_title=""
                )
            else:
                lang_fig = go.Figure()
                lang_fig.add_annotation(text="No multilingual data available", x=0.5, y=0.5)
            
            return summary, asr_fig, table_data, lang_fig
        
        # Layout
        app.layout = html.Div([
            html.H1("Papillon Attack Dashboard", style={'text-align': 'center'}),
            
            dcc.Interval(
                id='interval-component',
                interval=60000,  # Update every 60 seconds
                n_intervals=0
            ),
            
            html.Div(id='metrics-summary', className='metrics-container'),
            
            html.Div([
                html.Div([
                    html.H3("ASR by Attack Strategy"),
                    dcc.Graph(id='asr-chart')
                ], className='chart-container'),
                
                html.Div([
                    html.H3("Language Performance"),
                    dcc.Graph(id='language-heatmap')
                ], className='chart-container'),
            ], className='charts-row'),
            
            html.Div([
                html.H3("Recent Attacks"),
                dash_table.DataTable(
                    id='attack-table',
                    columns=[
                        {"name": "Attack", "id": "attack_name"},
                        {"name": "Question", "id": "question"},
                        {"name": "Success", "id": "success"},
                        {"name": "RoBERTa Score", "id": "roberta_score"},
                        {"name": "Timestamp", "id": "timestamp"}
                    ],
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
            ], className='table-container'),
            
            # CSS styles
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
        
        return app

def build_dashboard():
    """Main function to build and run the dashboard."""
    dashboard = DashboardBuilder()
    app = dashboard.create_dashboard()
    
    print("\n=== Dashboard Running ===")
    print("Access the dashboard at: http://127.0.0.1:8050")
    print("The dashboard will auto-update every 60 seconds")
    print("Press Ctrl+C to stop the server")
    
    app.run_server(debug=True, port=8050)

if __name__ == "__main__":
    build_dashboard()