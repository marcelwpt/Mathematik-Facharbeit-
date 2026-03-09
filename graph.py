import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context
from scipy.special import expi
import ast


# =============================================================================
# NULLSTELLEN AUS DATEI LADEN
# =============================================================================

def load_zeros_from_file(filename='zeros.txt'):
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
            zeros_list = ast.literal_eval(content)
            return np.array(zeros_list)
    except:
        return np.array([14.13472514, 21.02203964, 25.01085758, 30.42487613,
                         32.93506159, 37.58617816, 40.91871901, 43.32707328])


gammas_all = load_zeros_from_file('zeros.txt')


# =============================================================================
# MATHEMATISCHE FUNKTIONEN
# =============================================================================

def Li(x):
    return expi(np.log(x))


def J_approx(x_val, current_gammas):
    if x_val < 2:
        return 0
    res = Li(x_val)
    ln_x = np.log(x_val)
    sqrt_x = np.sqrt(x_val)
    osc = np.sum(2 * (sqrt_x / (current_gammas * ln_x)) * np.sin(current_gammas * ln_x))
    return res - osc - np.log(2)


def pi_riemann_inversion(x_array, current_gammas):
    pi_final = np.zeros_like(x_array)
    moebius = [(1, 1), (2, -1), (3, -1), (4, 0), (5, -1)]
    for i, x_val in enumerate(x_array):
        val_sum = 0
        for n, mu in moebius:
            if mu == 0:
                continue
            root_x = x_val ** (1.0 / n)
            if root_x < 2:
                break
            val_sum += (mu / n) * J_approx(root_x, current_gammas)
        pi_final[i] = val_sum
    return pi_final


def pi_exact_steps(n, x_grid):
    n = int(n)
    primes = []
    is_prime = [True] * (n + 1)
    for p in range(2, n + 1):
        if is_prime[p]:
            primes.append(p)
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
    return np.array([sum(1 for p in primes if p <= val) for val in x_grid])


# =============================================================================
# DASH APP
# =============================================================================

app = Dash(__name__)

input_style = {
    'width': '100%', 'padding': '10px', 'fontSize': '16px',
    'border': '2px solid #3498db', 'borderRadius': '6px',
    'textAlign': 'center', 'fontWeight': '500'
}

button_style = {
    'width': '100%', 'padding': '12px', 'fontSize': '16px',
    'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
    'borderRadius': '6px', 'cursor': 'pointer', 'fontWeight': 'bold',
    'marginTop': '10px'
}

app.layout = html.Div([
    # Header
    html.Div([
        html.H1('π(x) mit Riemann ζ-Nullstellen',
                style={'margin': 0, 'color': '#2c3e50'}),
        html.P(f'{len(gammas_all)} Nullstellen geladen',
               style={'margin': '5px 0 0 0', 'color': '#7f8c8d', 'fontSize': '14px'})
    ], style={
        'textAlign': 'center', 'padding': '20px',
        'backgroundColor': '#ecf0f1', 'borderBottom': '3px solid #3498db'
    }),

    html.Div([
        # ===== SIDEBAR =====
        html.Div([
            # Nullstellen
            html.Div([
                html.Label('Anzahl Nullstellen',
                           style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Input(
                    id='num-zeros-input',
                    type='number',
                    value=100,
                    min=1,
                    max=len(gammas_all),
                    style=input_style
                ),
            ], style={'marginBottom': '25px'}),

            # X-Bereich
            html.Div([
                html.Label('X-Minimum',
                           style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Input(
                    id='x-min-input',
                    type='number',
                    value=2.1,
                    min=2.1,
                    step=0.1,
                    style=input_style
                ),
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.Label('X-Maximum',
                           style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Input(
                    id='x-max-input',
                    type='number',
                    value=100,
                    min=10,
                    step=1,
                    style=input_style
                ),
            ], style={'marginBottom': '25px'}),

            # Auflösung
            html.Div([
                html.Label('Auflösung (Punkte)',
                           style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Input(
                    id='resolution-input',
                    type='number',
                    value=2000,
                    min=500,
                    max=5000,
                    step=500,
                    style=input_style
                ),
            ], style={'marginBottom': '25px'}),

            # Update Button
            html.Button('Aktualisieren', id='update-btn', style=button_style),

            # Schnellauswahl
            html.Div([
                html.Hr(style={'margin': '30px 0 20px 0', 'border': 'none', 'borderTop': '2px solid #bdc3c7'}),
                html.Label('Schnellauswahl',
                           style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                html.Button('Übersicht (10-100)', id='preset-1',
                            style={**button_style, 'backgroundColor': '#95a5a6', 'marginTop': '5px'}),
                html.Button('Detail (50-150)', id='preset-2',
                            style={**button_style, 'backgroundColor': '#95a5a6', 'marginTop': '5px'}),
                html.Button('Große Ansicht (10-1000)', id='preset-3',
                            style={**button_style, 'backgroundColor': '#95a5a6', 'marginTop': '5px'}),
            ]),

        ], style={
            'width': '280px', 'padding': '25px', 'backgroundColor': '#f8f9fa',
            'height': '100vh', 'overflowY': 'auto', 'boxShadow': '2px 0 8px rgba(0,0,0,0.1)'
        }),

        # ===== GRAPH =====
        html.Div([
            dcc.Graph(
                id='main-graph',
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'scrollZoom': True,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                },
                style={'height': '100vh'}
            )
        ], style={'flex': '1'})

    ], style={'display': 'flex'}),

], style={'margin': 0, 'padding': 0, 'fontFamily': 'system-ui, -apple-system, sans-serif'})


# =============================================================================
# CALLBACKS
# =============================================================================

# Presets
@app.callback(
    [Output('x-min-input', 'value'),
     Output('x-max-input', 'value'),
     Output('num-zeros-input', 'value')],
    [Input('preset-1', 'n_clicks'),
     Input('preset-2', 'n_clicks'),
     Input('preset-3', 'n_clicks')],
    prevent_initial_call=True
)
def update_presets(p1, p2, p3):
    from dash.exceptions import PreventUpdate
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'preset-1':
        return 10, 100, 100
    elif button_id == 'preset-2':
        return 50, 150, 500
    elif button_id == 'preset-3':
        return 10, 1000, 1000

    raise PreventUpdate


# Graph Update
@app.callback(
    Output('main-graph', 'figure'),
    Input('update-btn', 'n_clicks'),
    [State('num-zeros-input', 'value'),
     State('x-min-input', 'value'),
     State('x-max-input', 'value'),
     State('resolution-input', 'value')],
    prevent_initial_call=True
)
def update_graph(n_clicks, num_zeros, x_min, x_max, resolution):
    if x_min >= x_max:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="❌ Fehler: X-Minimum muss kleiner als X-Maximum sein",
            template='plotly_white',
            xaxis=dict(showgrid=True, zeroline=True, showline=True),
            yaxis=dict(showgrid=True, zeroline=True, showline=True)
        )
        return empty_fig

    try:
        # Berechnung
        gammas = gammas_all[:num_zeros]
        x = np.linspace(x_min, x_max, resolution)

        pi_corrected = pi_riemann_inversion(x, gammas)
        y_exact = pi_exact_steps(x_max, x)
        y_li = np.array([Li(v) for v in x])

        # Plot erstellen
        fig = go.Figure()

        # Exakte π(x)
        fig.add_trace(go.Scatter(
            x=x, y=y_exact,
            mode='lines',
            name='Exakte π(x)',
            line=dict(color='#e74c3c', width=2.5, shape='hv'),
            opacity=0.8
        ))

        # Riemann Approximation
        fig.add_trace(go.Scatter(
            x=x, y=pi_corrected,
            mode='lines',
            name=f'Riemann ({num_zeros} Nullstellen)',
            line=dict(color='#f39c12', width=3)
        ))

        # Li(x) Trend
        fig.add_trace(go.Scatter(
            x=x, y=y_li,
            mode='lines',
            name='Li(x)',
            line=dict(color='#27ae60', width=2, dash='dash'),
            opacity=0.5
        ))

        # Layout mit sichtbaren Achsen und Gitter
        fig.update_layout(
            title=dict(
                text=f'<b>π(x) Approximation</b><br><sub>Bereich: [{x_min:.1f}, {x_max:.1f}] • {num_zeros} Nullstellen • Mausrad zum Zoomen</sub>',
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            xaxis_title='x',
            yaxis_title='Anzahl der Primzahlen π(x)',
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#2c3e50',
                borderwidth=2
            ),
            margin=dict(l=70, r=30, t=100, b=60),
            dragmode='pan',
            # X-Achse mit Gitter und Nulllinie
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#d5d8dc',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='#7f8c8d',
                showline=True,
                linewidth=2,
                linecolor='#2c3e50',
                mirror=True,
                ticks='outside',
                tickwidth=2,
                tickcolor='#2c3e50'
            ),
            # Y-Achse mit Gitter und Nulllinie
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#d5d8dc',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='#7f8c8d',
                showline=True,
                linewidth=2,
                linecolor='#2c3e50',
                mirror=True,
                ticks='outside',
                tickwidth=2,
                tickcolor='#2c3e50'
            ),
            # Hintergrund
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=f"❌ Fehler bei der Berechnung: {str(e)}",
            template='plotly_white',
            xaxis=dict(showgrid=True, zeroline=True, showline=True),
            yaxis=dict(showgrid=True, zeroline=True, showline=True)
        )
        return empty_fig


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Riemann π(x) Visualisierung")
    print("=" * 60)
    print(f"📊 {len(gammas_all)} Nullstellen geladen")
    print("🌐 http://127.0.0.1:8050/")
    print("\n🖱️  ZOOM-FEATURES:")
    print("  • Mausrad im Graph: Zoomen")
    print("  • Maus ziehen: Verschieben (Pan)")
    print("  • Doppelklick: Zoom zurücksetzen")
    print("=" * 60 + "\n")

    app.run(debug=True)
