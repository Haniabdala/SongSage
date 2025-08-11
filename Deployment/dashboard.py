import dash
from dash import Dash, dcc, html, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import base64
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import numpy as np
from dash.dependencies import Input, Output

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server

df_global = pd.DataFrame()

sidebar = html.Div(
    [
        html.H2("Song Insights", className="display-6 fw-bold text-center mb-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="bi bi-house me-2"), "Home"], href="/", active="exact"),
                dbc.NavLink([html.I(className="bi bi-graph-up me-2"), "Lyrics Analysis"], href="/lyrics", active="exact"),
                dbc.NavLink([html.I(className="bi bi-music-note-list me-2"), "Audio Features"], href="/audio", active="exact"),
                dbc.NavLink([html.I(className="bi bi-card-list me-2"), "Song Metadata"], href="/metadata", active="exact"),
                dbc.NavLink([html.I(className="bi bi-upload me-2"), "Upload CSV"], href="/upload", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "18rem", "padding": "2rem 1rem", "backgroundColor": "#f8f9fa"},
)

content = html.Div(id="page-content", style={"marginLeft": "20rem", "padding": "2rem 1rem"})

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

upload_page = html.Div([
    html.H3("Upload CSV File"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ', html.A('Select Files')
        ]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center',
        },
        multiple=False
    ),
    html.Div(id='output-data-upload')
])

home_page = html.Div(
    style={
        'background': 'linear-gradient(to bottom, #f5f7fa, #c3cfe2)',
        'height': '100vh',
        'padding': '3rem',
        'textAlign': 'center',
        'fontFamily': 'Arial, sans-serif'
    },
    children=[
        html.Div(
            style={
                'maxWidth': '800px',
                'margin': '0 auto',
                'padding': '2rem',
                'backgroundColor': 'white',
                'borderRadius': '15px',
                'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'
            },
            children=[
                html.H1(
                    "Song Analytics Dashboard",
                    style={
                        'color': '#2c3e50',
                        'marginBottom': '1.5rem',
                        'fontWeight': '600',
                        'fontSize': '2.5rem'
                    }
                ),
                
                html.Div(
                    style={'borderTop': '2px solid #3498db', 'width': '100px', 'margin': '0 auto 2rem'}
                ),
                
                html.P(
                    "Advanced analytics platform for music data exploration and visualization",
                    style={
                        'color': '#7f8c8d',
                        'fontSize': '1.2rem',
                        'marginBottom': '2rem'
                    }
                ),
                
                html.Div(
                    style={
                        'display': 'flex',
                        'justifyContent': 'center',
                        'gap': '2rem',
                        'marginBottom': '2.5rem'
                    },
                    children=[
                        html.Div(
                            style={
                                'padding': '1.5rem',
                                'backgroundColor': '#f8f9fa',
                                'borderRadius': '10px',
                                'width': '200px',
                                'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'
                            },
                            children=[
                                html.I(className="fas fa-music fa-2x", style={'color': '#3498db', 'marginBottom': '1rem'}),
                                html.H3("Lyrics Analysis", style={'color': '#2c3e50', 'marginBottom': '0.5rem'}),
                                html.P("Word frequency, sentiment, and complexity metrics", style={'color': '#7f8c8d', 'fontSize': '0.9rem'})
                            ]
                        ),
                        html.Div(
                            style={
                                'padding': '1.5rem',
                                'backgroundColor': '#f8f9fa',
                                'borderRadius': '10px',
                                'width': '200px',
                                'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'
                            },
                            children=[
                                html.I(className="fas fa-wave-square fa-2x", style={'color': '#3498db', 'marginBottom': '1rem'}),
                                html.H3("Audio Features", style={'color': '#2c3e50', 'marginBottom': '0.5rem'}),
                                html.P("Tempo, key, loudness, and acoustic properties", style={'color': '#7f8c8d', 'fontSize': '0.9rem'})
                            ]
                        ),
                        html.Div(
                            style={
                                'padding': '1.5rem',
                                'backgroundColor': '#f8f9fa',
                                'borderRadius': '10px',
                                'width': '200px',
                                'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'
                            },
                            children=[
                                html.I(className="fas fa-chart-line fa-2x", style={'color': '#3498db', 'marginBottom': '1rem'}),
                                html.H3("Song Metadata", style={'color': '#2c3e50', 'marginBottom': '0.5rem'}),
                                html.P("Popularity, release trends, and genre insights", style={'color': '#7f8c8d', 'fontSize': '0.9rem'})
                            ]
                        )
                    ]
                ),
                
                html.Button(
                    "Get Started →",
                    id='start-button',
                    n_clicks=0, 
                    style={
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'border': 'none',
                        'padding': '12px 24px',
                        'borderRadius': '25px',
                        'fontSize': '1rem',
                        'cursor': 'pointer',
                        'transition': 'all 0.3s ease'
                    }
                ),
                
                html.Div(
                    style={'marginTop': '3rem', 'color': '#95a5a6', 'fontSize': '0.9rem'},
                    children=[
                        html.P("Upload your dataset or explore our sample data"),
                    ]
                )
            ]
        )
    ]
)


@app.callback(
    Output('url', 'pathname'),
    Input('start-button', 'n_clicks'),
    prevent_initial_call=True
)
def navigate_to_upload(n_clicks):
    if n_clicks:
        return '/upload'
    return dash.no_update

lyrics_page = html.Div([
    html.H3("Lyrics Analysis"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='lyrics-occured'), width=12),
        dbc.Col(dcc.Graph(id='lyrics-wordcount'), width=6),
        dbc.Col(dcc.Graph(id='lyrics-wordaveragesize'), width=6),
        dbc.Col(dcc.Graph(id='lyrics-explit'), width=6),
        
    ])
])

audio_page = html.Div([
    html.H3("Audio Features"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='audio-box'), width=6),
        dbc.Col(dcc.Graph(id='audio-corr'), width=6),
    ])
])

metadata_page = html.Div([
    html.H3("Song Metadata Analysis"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='meta-trend'), width=12),
        dbc.Col(dcc.Graph(id='meta-artistpop-pop'), width=6),
    ])
])

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page_content(pathname):
    if pathname == "/lyrics":
        return lyrics_page
    elif pathname == "/audio":
        return audio_page
    elif pathname == "/metadata":
        return metadata_page
    elif pathname == "/upload":
        return upload_page
    else:
        return home_page

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(contents, filename):
    global df_global
    if contents is not None:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df_global = df
            return html.Div([
                html.H5(filename),
                dash_table.DataTable(
                    data=df.head().to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'}
                )
            ])
        except Exception as e:
            return html.Div([f"There was an error processing this file: {e}"])

@app.callback(
    Output("lyrics-wordcount", "figure"), 
    Output("lyrics-wordaveragesize", "figure"),
    Output("lyrics-explit", "figure"), 
    Output("lyrics-occured", "figure"), 
    Input('url', 'pathname')
)
def update_lyrics_graphs(path):
    if df_global.empty:
        return dash.no_update, dash.no_update

    df_filtered = df_global[df_global['words_count'] > 10]
    fig3 = px.scatter(
        df_filtered,
        x='words_count',
        y='popularity',
        title='Words Count vs Popularity For All Songs',
        labels={'words_count': 'Number of Words in Lyrics', 'popularity': 'Popularity'},
        opacity=0.7,
        color_discrete_sequence=['blue']
    )

    df_wordaveragesize_vs_popularity_hits = df_global[(df_global['hit'] == 1) & (df_global['words_average_size'] > 0) & (df_global['words_count'] > 50)]
    
    fig4 = px.scatter(
        df_wordaveragesize_vs_popularity_hits,
        x='words_average_size',
        y='popularity',
        title='Word average size vs Popularity For Hit Songs',
        labels={'words_average_size': 'Words Average Size', 'popularity': 'Popularity'},
        opacity=0.7,
        color_discrete_sequence=['blue']
    )
    df_explicitness_vs_popularity_hits = df_global[(df_global['hit'] == 1)]
    
    non_explicit = df_explicitness_vs_popularity_hits[df_explicitness_vs_popularity_hits['Explicitness'] == 0]
    explicit = df_explicitness_vs_popularity_hits[df_explicitness_vs_popularity_hits['Explicitness'] == 1]
    
    fig5 = go.Figure()
    
    fig5.add_trace(go.Histogram(
        x=non_explicit['Explicitness'],
        name='Non-Explicit',
        marker_color='green',
        opacity=0.7
    ))
    
    fig5.add_trace(go.Histogram(
        x=explicit['Explicitness'],
        name='Explicit',
        marker_color='red',
        opacity=0.7
    ))
    
    fig5.update_layout(
        title='Distribution of Explicitness in Hit Songs',
        xaxis_title="Explicitness",
        yaxis_title="Number of Songs",
        xaxis=dict(tickvals=[0, 1], ticktext=['Non-Explicit', 'Explicit']),
        barmode='overlay',
        legend_title="Explicitness",
        legend=dict(x=0.8, y=0.9)
    )
    
    
    df_mostoccuringword_billboard = df_global[
        (df_global['words_count'] >= 50) & 
        (df_global['hit'] == 1) & 
        (df_global['1st_word'].str.len() >= 4) & 
        (df_global['1st_word'] != 'Not Found') & 
        (df_global['1st_word'] != 'feat')
    ]

    df_mostoccuringword_billboard['1st_word'] = df_mostoccuringword_billboard['1st_word'].str.lower()

    word_counts = df_mostoccuringword_billboard['1st_word'].value_counts().reset_index()
    word_counts.columns = ['word', 'count']

    top_words = word_counts.head(40)

    fig6 = px.bar(
        top_words, 
        x='word',  
        y='count',  
        title='Most Common First Words in Billboard Charting Songs',
        labels={'word': 'First Word in Lyrics', 'count': 'Number of Songs'},
        color='count',
        color_continuous_scale='Blues',
        text='count'
    )

    fig6.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        yaxis_gridcolor='lightgray',
        hovermode='x unified'
    )

    fig6.update_traces(
        texttemplate='%{text}',
        textposition='outside',
        marker_line_color='black',
        marker_line_width=1
    )

    return fig3, fig4, fig5, fig6

@app.callback(
    Output('audio-box', 'figure'),
    Output('audio-corr', 'figure'),
    Input('url', 'pathname')
)
def update_audio_graphs(path):
    if df_global.empty:
        return dash.no_update, dash.no_update
    features = ['danceability', 'energy', 'tempo']
    df_valid = df_global[[f for f in features if f in df_global]]
    
    hits = df_global[df_global['hit'] == 1]
    avg_audio_features = hits[["year", "danceability", "liveness", "energy", "acousticness", "duration_ms"]].groupby("year").mean().reset_index()

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=avg_audio_features["year"],
        y=avg_audio_features["danceability"],
        mode="lines+markers",
        name="Danceability",
        line=dict(color="blue", dash="solid"),
        marker=dict(symbol="circle")
    ))

    fig1.add_trace(go.Scatter(
        x=avg_audio_features["year"],
        y=avg_audio_features["liveness"],
        mode="lines+markers",
        name="Liveness",
        line=dict(color="red", dash="dash"),
        marker=dict(symbol="square")
    ))

    fig1.add_trace(go.Scatter(
        x=avg_audio_features["year"],
        y=avg_audio_features["energy"],
        mode="lines+markers",
        name="Energy",
        line=dict(color="green", dash="dashdot"),
        marker=dict(symbol="triangle-up")
    ))

    fig1.add_trace(go.Scatter(
        x=avg_audio_features["year"],
        y=avg_audio_features["acousticness"],
        mode="lines+markers",
        name="Acousticness",
        line=dict(color="purple", dash="dot"),
        marker=dict(symbol="diamond")
    ))

    fig1.update_layout(
        title=dict(text="Change of Audio Features Over Time", x=0.5, font=dict(size=16, family="Arial Black")),
        xaxis_title="Year",
        yaxis_title="Feature Value",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="Audio Feature",
        xaxis=dict(tickmode='array', tickvals=avg_audio_features["year"]),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )   
    
    
    features = ['danceability', 'energy', 'loudness', 'acousticness']
    colors = ['blue', 'red', 'green', 'purple']

    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[f'Distribution of {f}' for f in features])

    for i, (feature, color) in enumerate(zip(features, colors)):
        data = hits[feature].dropna()
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 200)
        y_vals = kde(x_vals)

        row = i // 2 + 1
        col = i % 2 + 1
        fig2.add_trace(
            go.Scatter(x=x_vals, y=y_vals, mode='lines', fill='tozeroy', line=dict(color=color), name=feature),
            row=row, col=col
        )

        fig2.update_xaxes(title_text='Value', row=row, col=col)
        fig2.update_yaxes(title_text='Density', row=row, col=col)

    fig2.update_layout(
        height=600,
        width=800,
        title_text="KDE Distributions of Features for Hit Songs",
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )    
    
    return fig1, fig2


@app.callback(
    Output('meta-artistpop-pop', 'figure'),
    Output('meta-trend', 'figure'),
    Input('url', 'pathname')
)
def update_metadata_graphs(path):
    if df_global.empty:
        return dash.no_update, dash.no_update

    fig1 = px.scatter(
        df_global,
        x='artist_popularity',
        y='popularity',
        title='Words Count vs Popularity For All Songs',
        labels={'words_count': 'Number of Words in Lyrics', 'popularity': 'Popularity'},
        opacity=0.7,
        color_discrete_sequence=['blue']
    )    

    hit_songs = df_global[df_global["hit"] == 1]
    genre_trends = hit_songs.groupby(["year", "genre_encoded"]).size().reset_index(name="num_hits")
    top_genres = hit_songs["genre_encoded"].value_counts().head(10).index
    
    
    genre_mapping = {
    161: "Pop",
    217: "reggaeton",
    170: "rap",
    174: "r&b",
    167: "melodic rap",
    103: "k-pop",
    100: "german hip hop",
    52: "edm",
    117: "soft-pop",
    75: "country"
}
    
    genre_trends_top10 = genre_trends[genre_trends["genre_encoded"].isin(top_genres)]
    genre_trends_top10["genre_name"] = genre_trends_top10["genre_encoded"].map(genre_mapping)
    genre_trends_top10 = genre_trends_top10.copy()
    genre_trends_top10["genre_name"] = genre_trends_top10["genre_encoded"].map(genre_mapping)

    fig2 = px.line(
        genre_trends_top10,
        x="year",
        y="num_hits",
        color="genre_name", 
        title="Top 10 Genres Over the Years (Based on Number of Hit Songs)",
        labels={"year": "Year", "num_hits": "Number of Hit Songs", "genre_name": "Genre"},
        markers=True,
        color_discrete_sequence=px.colors.qualitative.T10
    )

    fig2.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            title='Year'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            title='Number of Hit Songs'
        ),
        legend=dict(
            title='Genre',
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02 
        ),
        title=dict(
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        hovermode='x unified',
        margin=dict(r=150)
    )

    fig2.update_traces(
        line_width=2,
        marker=dict(
            size=8,
            line=dict(width=1, color='DarkSlateGrey')
    ))
    
    return fig1, fig2

if __name__ == '__main__':
    app.run(debug=True)
