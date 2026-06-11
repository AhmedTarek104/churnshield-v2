"""ChurnShield v2 — 5-tab Dash dashboard."""
import base64
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

port = int(os.environ.get("PORT", 8050))

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
PROC_DIR    = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGS_DIR    = OUTPUTS_DIR / "figures"

# ── Theme ────────────────────────────────────────────────────────────────────
BG       = "#0D0D0D"
CARD_BG  = "#1A1A1A"
RED      = "#CC0000"
GREEN    = "#00C853"
AMBER    = "#FFB300"
TEXT     = "#F5F5F5"
SUBTEXT  = "#AAAAAA"

CARD_STYLE = {
    "backgroundColor": CARD_BG,
    "border": f"1px solid #333",
    "borderRadius": "8px",
    "padding": "16px",
    "color": TEXT,
}

# ── Data Loading ─────────────────────────────────────────────────────────────
def load_all():
    data = {}
    try:
        data["test_df"] = pd.read_csv(os.path.join(PROC_DIR, "test_with_predictions.csv"))
    except Exception:
        data["test_df"] = pd.DataFrame()

    try:
        with open(os.path.join(OUTPUTS_DIR, "metrics.json")) as f:
            data["metrics"] = json.load(f)
    except Exception:
        data["metrics"] = {}

    try:
        with open(os.path.join(MODELS_DIR, "thresholds.json")) as f:
            data["thresholds"] = json.load(f)
    except Exception:
        data["thresholds"] = {"f1_optimal": 0.5, "business_optimal": 0.5}

    try:
        data["comparison"] = pd.read_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"))
    except Exception:
        data["comparison"] = pd.DataFrame()

    try:
        data["fi"] = pd.read_csv(os.path.join(OUTPUTS_DIR, "feature_importance.csv"))
    except Exception:
        data["fi"] = pd.DataFrame()

    try:
        with open(os.path.join(PROC_DIR, "feature_names.json")) as f:
            data["feat_names"] = json.load(f)
    except Exception:
        data["feat_names"] = []

    try:
        with open(MODELS_DIR / "best_model.pkl", "rb") as f:
            data["model"] = pickle.load(f)
        print("Model loaded successfully")
    except FileNotFoundError:
        data["model"] = None
        print("Model file not found")

    try:
        with open(MODELS_DIR / "shap_explainer.pkl", "rb") as f:
            data["explainer_bundle"] = pickle.load(f)
        print("SHAP explainer loaded successfully")
    except FileNotFoundError:
        data["explainer_bundle"] = None
        print("SHAP explainer not found")

    return data


D = load_all()
test_df    = D["test_df"]
metrics    = D["metrics"]
thresholds = D["thresholds"]
comparison = D["comparison"]
fi_df      = D["fi"]
feat_names = D["feat_names"]
model      = D["model"]
expl_bundle= D["explainer_bundle"]

BIZ_THRESH = thresholds.get("business_optimal", 0.5)
F1_THRESH  = thresholds.get("f1_optimal", 0.5)

# Decode geography/gender back for display
GEO_REVERSE = {0: "France", 1: "Germany", 2: "Spain"}
GEN_REVERSE = {0: "Female", 1: "Male"}
AGE_GROUP_LABEL = {0: "Young (18-30)", 1: "Prime (31-45)", 2: "Senior (46-60)", 3: "Elderly (61+)"}
TENURE_LABEL = {0: "New (0-2yr)", 1: "Developing (3-5yr)", 2: "Loyal (6+yr)"}
CREDIT_LABEL = {0: "Poor (<580)", 1: "Fair (580-669)", 2: "Good (670-739)",
                3: "Very Good (740-799)", 4: "Exceptional (800+)"}

if not test_df.empty:
    if "Geography" in test_df.columns and test_df["Geography"].dtype in [np.int64, np.float64, int, float]:
        test_df["Geography_Label"] = test_df["Geography"].map(GEO_REVERSE)
    else:
        test_df["Geography_Label"] = test_df.get("Geography", "Unknown")

    if "Gender" in test_df.columns and test_df["Gender"].dtype in [np.int64, np.float64, int, float]:
        test_df["Gender_Label"] = test_df["Gender"].map(GEN_REVERSE)
    else:
        test_df["Gender_Label"] = test_df.get("Gender", "Unknown")

    test_df["Age_Group_Label"] = test_df.get("age_group", pd.Series(dtype=int)).map(AGE_GROUP_LABEL)
    test_df["Tenure_Label"] = test_df.get("tenure_stability", pd.Series(dtype=int)).map(TENURE_LABEL)
    test_df["Credit_Label"] = test_df.get("credit_score_band", pd.Series(dtype=int)).map(CREDIT_LABEL)

    if "churn_probability" in test_df.columns:
        test_df["risk_level"] = test_df["churn_probability"].apply(
            lambda p: "HIGH" if p >= 0.6 else ("MEDIUM" if p >= 0.35 else "LOW")
        )
    if "retention_value" not in test_df.columns and "clv_estimated" in test_df.columns:
        test_df["retention_value"] = (test_df["clv_estimated"] * test_df.get("churn_probability", 0) * 0.30).round(2)


# ── Helper: KPI card ─────────────────────────────────────────────────────────
def kpi_card(title, value, color=TEXT, subtitle=""):
    return html.Div([
        html.P(title, style={"color": SUBTEXT, "fontSize": "12px", "margin": "0"}),
        html.H3(value, style={"color": color, "margin": "4px 0", "fontSize": "1.6rem"}),
        html.P(subtitle, style={"color": SUBTEXT, "fontSize": "11px", "margin": "0"}),
    ], style={**CARD_STYLE, "textAlign": "center"})


# ── TAB 1: Executive Command Center ──────────────────────────────────────────
def build_tab1():
    if test_df.empty:
        return html.Div("No data", style={"color": RED})

    n_total = len(test_df)
    n_risk = int((test_df["churn_probability"] >= BIZ_THRESH).sum()) if "churn_probability" in test_df.columns else 0
    risk_pct = n_risk / n_total if n_total > 0 else 0
    risk_color = RED if risk_pct > 0.25 else (AMBER if risk_pct > 0.15 else GREEN)

    rev_at_risk = test_df.loc[test_df.get("risk_level", pd.Series("LOW", index=test_df.index)) == "HIGH", "retention_value"].sum() if "retention_value" in test_df.columns else 0
    auc = metrics.get("test_auc", 0)
    catch = metrics.get("catch_rate", 0)
    avg_clv = test_df["clv_estimated"].mean() if "clv_estimated" in test_df.columns else 0
    net_val = metrics.get("net_value_at_optimal_threshold", 0)

    # Churn risk distribution
    prob_col = test_df["churn_probability"] if "churn_probability" in test_df.columns else pd.Series([])
    fig_dist = go.Figure()
    if not prob_col.empty:
        fig_dist.add_trace(go.Histogram(
            x=prob_col, nbinsx=40,
            marker_color=[GREEN if p < 0.35 else (AMBER if p < 0.6 else RED) for p in np.linspace(0, 1, 40)],
            name="Churn Probability"
        ))
        fig_dist.add_vline(x=F1_THRESH, line_color=GREEN, line_dash="dash",
                           annotation_text=f"F1-opt ({F1_THRESH})", annotation_font_color=GREEN)
        fig_dist.add_vline(x=BIZ_THRESH, line_color=AMBER, line_dash="dash",
                           annotation_text=f"Biz-opt ({BIZ_THRESH})", annotation_font_color=AMBER)
    fig_dist.update_layout(
        title="Churn Risk Distribution",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font_color=TEXT, xaxis_title="Churn Probability", yaxis_title="Count",
        showlegend=False,
    )

    # Revenue at risk by segment
    if "Geography_Label" in test_df.columns and "age_group" in test_df.columns and "retention_value" in test_df.columns:
        seg_df = test_df.groupby(["Geography_Label", "Age_Group_Label"])["retention_value"].sum().reset_index()
        seg_df.columns = ["Geography", "Age Group", "Retention Value (€)"]
        fig_seg = px.bar(
            seg_df, x="Geography", y="Retention Value (€)", color="Age Group",
            title="Revenue at Risk by Segment", barmode="group",
            color_discrete_sequence=[RED, AMBER, GREEN, "#2196F3"],
        )
        fig_seg.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT)
    else:
        fig_seg = go.Figure()
        fig_seg.update_layout(title="Revenue at Risk by Segment (no data)",
                              paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT)

    # Top 10 highest retention value customers
    top10_cols = ["CustomerId", "Age", "Geography_Label", "Balance", "clv_estimated",
                  "churn_probability", "retention_value", "risk_level"]
    top10_cols = [c for c in top10_cols if c in test_df.columns]
    top10 = test_df.nlargest(10, "retention_value")[top10_cols] if "retention_value" in test_df.columns else test_df.head(10)

    table_header = [html.Tr([html.Th(c, style={"color": AMBER, "padding": "6px"}) for c in top10.columns])]
    table_rows = []
    for _, row in top10.iterrows():
        risk = str(row.get("risk_level", ""))
        row_color = "#3a0000" if risk == "HIGH" else ("#3a2a00" if risk == "MEDIUM" else CARD_BG)
        table_rows.append(html.Tr([
            html.Td(str(row[c])[:10] if c == "CustomerId" else
                    f"€{row[c]:,.0f}" if c in ("Balance", "clv_estimated", "retention_value") else
                    f"{row[c]:.2%}" if c == "churn_probability" else str(row[c]),
                    style={"padding": "6px", "color": TEXT, "fontSize": "12px"})
            for c in top10.columns
        ], style={"backgroundColor": row_color}))

    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card("At-Risk Customers", f"{n_risk:,}", color=risk_color,
                             subtitle=f"{risk_pct:.1%} of base"), width=2),
            dbc.Col(kpi_card("Revenue at Risk", f"€{rev_at_risk:,.0f}", color=RED), width=2),
            dbc.Col(kpi_card("Model AUC", f"{auc:.4f}", color=GREEN), width=2),
            dbc.Col(kpi_card("Catch Rate", f"{catch:.1%}", color=AMBER), width=2),
            dbc.Col(kpi_card("Avg CLV", f"€{avg_clv:,.0f}", color=TEXT), width=2),
            dbc.Col(kpi_card("Business Value", f"€{net_val:,.0f}", color=GREEN), width=2),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_dist), width=6),
            dbc.Col(dcc.Graph(figure=fig_seg), width=6),
        ], className="mb-4"),
        html.Div([
            html.H5("Top 10 Highest Retention Value Customers", style={"color": AMBER, "marginBottom": "10px"}),
            html.P("These are the customers to contact FIRST", style={"color": SUBTEXT, "fontSize": "12px"}),
            html.Table(
                table_header + table_rows,
                style={"width": "100%", "borderCollapse": "collapse", "backgroundColor": CARD_BG}
            )
        ], style={**CARD_STYLE}),
    ], style={"padding": "20px"})


# ── TAB 2: Retention Prioritization ──────────────────────────────────────────
def build_tab2():
    if test_df.empty:
        return html.Div("No data", style={"color": RED})

    max_rv = float(test_df["retention_value"].max()) if "retention_value" in test_df.columns else 500

    return html.Div([
        html.H4("Who should we call first?", style={"color": AMBER, "marginBottom": "16px"}),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label("Risk Level", style={"color": SUBTEXT, "fontSize": "12px"}),
                    dcc.Dropdown(
                        id="filter-risk",
                        options=[{"label": v, "value": v} for v in ["ALL", "HIGH", "MEDIUM", "LOW"]],
                        value="ALL", clearable=False,
                        style={"backgroundColor": CARD_BG, "color": "#000"},
                    ),
                    html.Label("Geography", style={"color": SUBTEXT, "fontSize": "12px", "marginTop": "10px"}),
                    dcc.Dropdown(
                        id="filter-geo",
                        options=[{"label": v, "value": v} for v in ["ALL", "France", "Germany", "Spain"]],
                        value="ALL", clearable=False,
                        style={"backgroundColor": CARD_BG, "color": "#000"},
                    ),
                    html.Label("Min Retention Value (€)", style={"color": SUBTEXT, "fontSize": "12px", "marginTop": "10px"}),
                    dcc.Slider(
                        id="filter-rv", min=0, max=int(max_rv), step=10, value=0,
                        marks={0: "€0", int(max_rv//2): f"€{int(max_rv//2)}", int(max_rv): f"€{int(max_rv)}"},
                        tooltip={"placement": "bottom"},
                    ),
                    html.Label("Active Status", style={"color": SUBTEXT, "fontSize": "12px", "marginTop": "10px"}),
                    dcc.Dropdown(
                        id="filter-active",
                        options=[{"label": v, "value": v} for v in ["ALL", "Active", "Inactive"]],
                        value="ALL", clearable=False,
                        style={"backgroundColor": CARD_BG, "color": "#000"},
                    ),
                    html.Label("Sort by", style={"color": SUBTEXT, "fontSize": "12px", "marginTop": "10px"}),
                    dcc.Dropdown(
                        id="filter-sort",
                        options=[
                            {"label": "Retention Value", "value": "retention_value"},
                            {"label": "Churn Probability", "value": "churn_probability"},
                            {"label": "CLV", "value": "clv_estimated"},
                        ],
                        value="retention_value", clearable=False,
                        style={"backgroundColor": CARD_BG, "color": "#000"},
                    ),
                    html.Button("Export CSV", id="btn-export", n_clicks=0,
                                style={"marginTop": "16px", "backgroundColor": RED,
                                       "color": TEXT, "border": "none",
                                       "padding": "8px 16px", "borderRadius": "4px",
                                       "cursor": "pointer", "width": "100%"}),
                    dcc.Download(id="download-csv"),
                ], style={**CARD_STYLE, "minHeight": "500px"}),
            ], width=2),
            dbc.Col([
                html.Div(id="tab2-table", style={**CARD_STYLE, "overflowX": "auto", "minHeight": "500px"}),
                html.Div(id="tab2-insight", style={**CARD_STYLE, "marginTop": "12px", "borderColor": AMBER}),
            ], width=10),
        ]),
    ], style={"padding": "20px"})


# ── TAB 3: Customer Deep Dive ─────────────────────────────────────────────────
def build_tab3():
    customer_options = []
    if not test_df.empty and "CustomerId" in test_df.columns:
        customer_options = [{"label": f"ID {row['CustomerId']} | Age {row['Age']} | {row.get('Geography_Label','?')} | {row.get('risk_level','?')}", "value": i}
                            for i, row in test_df.head(200).iterrows()]

    return html.Div([
        html.H4("Individual Customer Analysis", style={"color": AMBER, "marginBottom": "16px"}),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label("Select Customer from Test Set", style={"color": SUBTEXT, "fontSize": "12px"}),
                    dcc.Dropdown(
                        id="customer-select",
                        options=customer_options,
                        placeholder="Select a customer…",
                        style={"backgroundColor": CARD_BG, "color": "#000"},
                    ),
                    html.Hr(style={"borderColor": "#333"}),
                    html.Label("Or enter manually:", style={"color": SUBTEXT, "fontSize": "12px"}),
                    html.Div(id="manual-form", children=[
                        dbc.Row([
                            dbc.Col([
                                html.Label("CreditScore", style={"color": SUBTEXT, "fontSize": "11px"}),
                                dcc.Input(id="in-cs", type="number", value=650, min=300, max=850,
                                          style={"width": "100%", "backgroundColor": "#222", "color": TEXT,
                                                 "border": "1px solid #444", "borderRadius": "4px", "padding": "4px"}),
                            ], width=6),
                            dbc.Col([
                                html.Label("Age", style={"color": SUBTEXT, "fontSize": "11px"}),
                                dcc.Input(id="in-age", type="number", value=42, min=18, max=95,
                                          style={"width": "100%", "backgroundColor": "#222", "color": TEXT,
                                                 "border": "1px solid #444", "borderRadius": "4px", "padding": "4px"}),
                            ], width=6),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Geography", style={"color": SUBTEXT, "fontSize": "11px"}),
                                dcc.Dropdown(id="in-geo", options=[{"label": g, "value": g} for g in ["France", "Germany", "Spain"]],
                                             value="Germany", clearable=False,
                                             style={"backgroundColor": CARD_BG, "color": "#000"}),
                            ], width=6),
                            dbc.Col([
                                html.Label("Gender", style={"color": SUBTEXT, "fontSize": "11px"}),
                                dcc.Dropdown(id="in-gen", options=[{"label": g, "value": g} for g in ["Male", "Female"]],
                                             value="Female", clearable=False,
                                             style={"backgroundColor": CARD_BG, "color": "#000"}),
                            ], width=6),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Balance (€)", style={"color": SUBTEXT, "fontSize": "11px"}),
                                dcc.Input(id="in-bal", type="number", value=125000, min=0,
                                          style={"width": "100%", "backgroundColor": "#222", "color": TEXT,
                                                 "border": "1px solid #444", "borderRadius": "4px", "padding": "4px"}),
                            ], width=6),
                            dbc.Col([
                                html.Label("Salary (€)", style={"color": SUBTEXT, "fontSize": "11px"}),
                                dcc.Input(id="in-sal", type="number", value=70000, min=1,
                                          style={"width": "100%", "backgroundColor": "#222", "color": TEXT,
                                                 "border": "1px solid #444", "borderRadius": "4px", "padding": "4px"}),
                            ], width=6),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Tenure", style={"color": SUBTEXT, "fontSize": "11px"}),
                                dcc.Input(id="in-ten", type="number", value=2, min=0, max=10,
                                          style={"width": "100%", "backgroundColor": "#222", "color": TEXT,
                                                 "border": "1px solid #444", "borderRadius": "4px", "padding": "4px"}),
                            ], width=4),
                            dbc.Col([
                                html.Label("Products", style={"color": SUBTEXT, "fontSize": "11px"}),
                                dcc.Input(id="in-prod", type="number", value=1, min=1, max=4,
                                          style={"width": "100%", "backgroundColor": "#222", "color": TEXT,
                                                 "border": "1px solid #444", "borderRadius": "4px", "padding": "4px"}),
                            ], width=4),
                            dbc.Col([
                                html.Label("Active", style={"color": SUBTEXT, "fontSize": "11px"}),
                                dcc.Dropdown(id="in-active", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
                                             value=0, clearable=False,
                                             style={"backgroundColor": CARD_BG, "color": "#000"}),
                            ], width=4),
                        ], className="mb-2"),
                        html.Button("Analyze Customer", id="btn-analyze", n_clicks=0,
                                    style={"width": "100%", "backgroundColor": RED, "color": TEXT,
                                           "border": "none", "padding": "8px", "borderRadius": "4px",
                                           "cursor": "pointer", "marginTop": "8px"}),
                    ]),
                ], style={**CARD_STYLE}),
            ], width=3),
            dbc.Col([
                html.Div(id="customer-profile", style={**CARD_STYLE, "minHeight": "200px"}),
            ], width=3),
            dbc.Col([
                html.Div(id="churn-gauge", style={**CARD_STYLE, "minHeight": "200px"}),
            ], width=3),
            dbc.Col([
                html.Div(id="risk-factors", style={**CARD_STYLE, "minHeight": "200px", "borderColor": AMBER}),
            ], width=3),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Div(id="shap-waterfall", style={**CARD_STYLE}),
            ], width=12),
        ]),
    ], style={"padding": "20px"})


# ── TAB 4: Cohort Intelligence ────────────────────────────────────────────────
def build_tab4():
    if test_df.empty:
        return html.Div("No data", style={"color": RED})

    def churn_bar(group_col, label_col, title):
        if group_col not in test_df.columns or "Exited" not in test_df.columns:
            return go.Figure().update_layout(title=title, paper_bgcolor=CARD_BG,
                                             plot_bgcolor=CARD_BG, font_color=TEXT)
        grp = test_df.groupby(label_col)["Exited"].mean().reset_index()
        grp.columns = [label_col, "Churn Rate"]
        grp["Churn Rate %"] = (grp["Churn Rate"] * 100).round(1)
        fig = px.bar(grp, x=label_col, y="Churn Rate %", title=title,
                     color="Churn Rate %",
                     color_continuous_scale=[[0, GREEN], [0.5, AMBER], [1, RED]])
        fig.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT,
                          showlegend=False, coloraxis_showscale=False)
        return fig

    fig_geo = churn_bar("Geography", "Geography_Label", "Churn Rate by Geography")
    fig_age = churn_bar("age_group", "Age_Group_Label", "Churn Rate by Age Group")
    fig_ten = churn_bar("tenure_stability", "Tenure_Label", "Churn Rate by Tenure")
    fig_cred = churn_bar("credit_score_band", "Credit_Label", "Churn Rate by Credit Band")

    # Revenue by geography
    if "Geography_Label" in test_df.columns and "retention_value" in test_df.columns:
        rv_geo = test_df.groupby("Geography_Label")["retention_value"].sum().reset_index()
        rv_geo.columns = ["Geography", "Total Retention Value (€)"]
        fig_rv = px.bar(rv_geo, x="Geography", y="Total Retention Value (€)",
                        title="Avg Retention Value by Geography",
                        color="Total Retention Value (€)",
                        color_continuous_scale=[[0, "#1A1A1A"], [1, RED]])
        fig_rv.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT,
                             coloraxis_showscale=False)
    else:
        fig_rv = go.Figure().update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT)

    # Activity status impact
    if "IsActiveMember" in test_df.columns and "Exited" in test_df.columns:
        act_df = test_df.groupby("IsActiveMember")["Exited"].mean().reset_index()
        act_df["Status"] = act_df["IsActiveMember"].map({0: "Inactive", 1: "Active"})
        act_df["Churn Rate %"] = (act_df["Exited"] * 100).round(1)
        fig_act = px.bar(act_df, x="Status", y="Churn Rate %",
                         title="Churn Rate: Active vs Inactive",
                         color="Status",
                         color_discrete_map={"Active": GREEN, "Inactive": RED})
        fig_act.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT, showlegend=False)
    else:
        fig_act = go.Figure().update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT)

    # Auto insight
    insight = ""
    if "Geography_Label" in test_df.columns and "Exited" in test_df.columns:
        gr = test_df.groupby("Geography_Label")["Exited"].mean()
        worst_geo = gr.idxmax()
        worst_rate = gr.max()
        avg_rv = test_df[test_df["Geography_Label"] == worst_geo]["retention_value"].mean() if "retention_value" in test_df.columns else 0
        insight = (f"Highest-risk segment: {worst_geo} customers have a "
                   f"{worst_rate:.1%} churn rate with average retention value of €{avg_rv:.0f}.")

    return html.Div([
        html.H4("Which customer segments are most at risk?", style={"color": AMBER, "marginBottom": "16px"}),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_geo), width=6),
            dbc.Col(dcc.Graph(figure=fig_age), width=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_ten), width=6),
            dbc.Col(dcc.Graph(figure=fig_cred), width=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_rv), width=6),
            dbc.Col(dcc.Graph(figure=fig_act), width=6),
        ], className="mb-3"),
        html.Div([
            html.H6("Key Insight", style={"color": AMBER}),
            html.P(insight or "Insufficient data for insight.", style={"color": TEXT}),
        ], style={**CARD_STYLE, "borderColor": AMBER}),
    ], style={"padding": "20px"})


# ── TAB 5: Model Performance ──────────────────────────────────────────────────
def build_tab5():
    # Model comparison chart
    if not comparison.empty:
        fig_comp = go.Figure()
        colors = [RED if i == 0 else "#555" for i in range(len(comparison))]
        fig_comp.add_trace(go.Bar(
            name="CV AUC", x=comparison["Model"], y=comparison["CV AUC"],
            marker_color=[RED if row["Model"] == metrics.get("model_name", "") else "#444"
                          for _, row in comparison.iterrows()],
        ))
        fig_comp.add_trace(go.Bar(
            name="Val AUC", x=comparison["Model"], y=comparison["Val AUC"],
            marker_color=[AMBER if row["Model"] == metrics.get("model_name", "") else "#666"
                          for _, row in comparison.iterrows()],
        ))
        fig_comp.update_layout(
            title="Model Comparison — CV AUC vs Val AUC",
            barmode="group",
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT,
            yaxis=dict(range=[0.7, 1.0]),
        )
    else:
        fig_comp = go.Figure().update_layout(title="Model Comparison (no data)",
                                              paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT)

    # Threshold analysis — loaded from metrics.json (pre-computed by evaluate.py)
    thresh_fig = go.Figure()
    thresh_data = metrics.get("threshold_analysis", {})
    if thresh_data:
        t_vals = thresh_data["thresholds"]
        f1_vals = thresh_data["f1_scores"]
        net_vals = thresh_data["net_values"]
        thresh_fig.add_trace(go.Scatter(
            x=t_vals, y=f1_vals, name="F1 Score",
            line=dict(color=GREEN, width=2),
        ))
        thresh_fig.add_trace(go.Scatter(
            x=t_vals, y=net_vals, name="Net Value (€)",
            line=dict(color=AMBER, width=2), yaxis="y2",
        ))
        thresh_fig.add_vline(x=float(F1_THRESH), line_color=GREEN, line_dash="dash",
                             annotation_text=f"F1-opt ({F1_THRESH})",
                             annotation_font_color=GREEN)
        thresh_fig.add_vline(x=float(BIZ_THRESH), line_color=AMBER, line_dash="dash",
                             annotation_text=f"Biz-opt ({BIZ_THRESH})",
                             annotation_font_color=AMBER)
    thresh_fig.update_layout(
        title="Threshold Analysis: F1 vs Business Value",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT,
        yaxis=dict(title="F1 Score", color=GREEN),
        yaxis2=dict(title="Net Business Value (€)", color=AMBER, overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
    )

    # Metrics table
    metrics_rows = []
    display_keys = [
        ("model_name", "Model"), ("test_auc", "Test AUC"), ("test_f1", "Test F1"),
        ("test_precision", "Precision"), ("test_recall", "Recall"), ("test_accuracy", "Accuracy"),
        ("average_precision", "Avg Precision"), ("catch_rate", "Catch Rate"),
        ("false_alarm_rate", "False Alarm Rate"), ("f1_optimal_threshold", "F1 Threshold"),
        ("business_optimal_threshold", "Business Threshold"),
        ("net_value_at_optimal_threshold", "Net Value (€)"),
    ]
    for key, label in display_keys:
        val = metrics.get(key, "N/A")
        if isinstance(val, float):
            val_str = f"{val:.4f}" if val < 10 else f"€{val:,.2f}"
        else:
            val_str = str(val)
        metrics_rows.append(html.Tr([
            html.Td(label, style={"color": AMBER, "padding": "8px", "fontWeight": "bold"}),
            html.Td(val_str, style={"color": TEXT, "padding": "8px"}),
        ]))

    # Feature importance chart
    if not fi_df.empty:
        fig_fi = px.bar(fi_df.head(15)[::-1], x="mean_abs_shap", y="feature",
                        orientation="h", title="Top 15 Features (SHAP)",
                        color="mean_abs_shap",
                        color_continuous_scale=[[0, "#333"], [1, RED]])
        fig_fi.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT,
                             coloraxis_showscale=False)
    else:
        fig_fi = go.Figure().update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT)

    # ROC and PR curve images
    def _img_src(fname):
        path = os.path.join(FIGS_DIR, fname)
        try:
            with open(path, "rb") as fh:
                return "data:image/png;base64," + base64.b64encode(fh.read()).decode()
        except Exception:
            return None

    roc_src = _img_src("roc_curve.png")
    pr_src  = _img_src("pr_curve.png")

    roc_widget = (
        html.Img(src=roc_src, style={"width": "100%", "borderRadius": "6px"})
        if roc_src else html.P("ROC curve not found — re-run evaluate.py", style={"color": SUBTEXT})
    )
    pr_widget = (
        html.Img(src=pr_src, style={"width": "100%", "borderRadius": "6px"})
        if pr_src else html.P("PR curve not found — re-run evaluate.py", style={"color": SUBTEXT})
    )

    return html.Div([
        html.H4("Model Performance", style={"color": AMBER, "marginBottom": "16px"}),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_comp), width=6),
            dbc.Col(dcc.Graph(figure=thresh_fig), width=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(html.Div(roc_widget, style={**CARD_STYLE}), width=6),
            dbc.Col(html.Div(pr_widget,  style={**CARD_STYLE}), width=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_fi), width=6),
            dbc.Col([
                html.Div([
                    html.H5("Evaluation Metrics", style={"color": AMBER, "marginBottom": "12px"}),
                    html.Table(
                        metrics_rows,
                        style={"width": "100%", "borderCollapse": "collapse"},
                    )
                ], style={**CARD_STYLE}),
            ], width=6),
        ]),
    ], style={"padding": "20px"})


# ── App Layout ────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)
server = app.server

app.layout = html.Div([
    html.Div([
        html.H2("ChurnShield v2", style={"color": RED, "margin": "0", "fontWeight": "bold"}),
        html.Span("FinTech Customer Churn Intelligence Platform",
                  style={"color": SUBTEXT, "fontSize": "14px", "marginLeft": "16px"}),
        html.Span(f"Model: {metrics.get('model_name','—')} | AUC: {metrics.get('test_auc','—')}",
                  style={"color": GREEN, "fontSize": "12px", "marginLeft": "auto"}),
    ], style={"display": "flex", "alignItems": "center", "padding": "12px 24px",
              "backgroundColor": "#111", "borderBottom": f"2px solid {RED}"}),

    dcc.Tabs(id="tabs", value="tab1", children=[
        dcc.Tab(label="Executive Command", value="tab1",
                style={"backgroundColor": CARD_BG, "color": SUBTEXT},
                selected_style={"backgroundColor": RED, "color": TEXT, "fontWeight": "bold"}),
        dcc.Tab(label="Retention Priority", value="tab2",
                style={"backgroundColor": CARD_BG, "color": SUBTEXT},
                selected_style={"backgroundColor": RED, "color": TEXT, "fontWeight": "bold"}),
        dcc.Tab(label="Customer Deep Dive", value="tab3",
                style={"backgroundColor": CARD_BG, "color": SUBTEXT},
                selected_style={"backgroundColor": RED, "color": TEXT, "fontWeight": "bold"}),
        dcc.Tab(label="Cohort Intelligence", value="tab4",
                style={"backgroundColor": CARD_BG, "color": SUBTEXT},
                selected_style={"backgroundColor": RED, "color": TEXT, "fontWeight": "bold"}),
        dcc.Tab(label="Model Performance", value="tab5",
                style={"backgroundColor": CARD_BG, "color": SUBTEXT},
                selected_style={"backgroundColor": RED, "color": TEXT, "fontWeight": "bold"}),
    ], style={"backgroundColor": "#111"}),

    html.Div(id="tab-content", style={"backgroundColor": BG, "minHeight": "calc(100vh - 100px)"}),
], style={"backgroundColor": BG, "fontFamily": "'Segoe UI', sans-serif", "minHeight": "100vh"})


# ── Callbacks ─────────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab1": return build_tab1()
    if tab == "tab2": return build_tab2()
    if tab == "tab3": return build_tab3()
    if tab == "tab4": return build_tab4()
    if tab == "tab5": return build_tab5()
    return html.Div("Unknown tab")


@app.callback(
    Output("tab2-table", "children"),
    Output("tab2-insight", "children"),
    Input("filter-risk", "value"),
    Input("filter-geo", "value"),
    Input("filter-rv", "value"),
    Input("filter-active", "value"),
    Input("filter-sort", "value"),
)
def update_tab2(risk, geo, min_rv, active, sort_col):
    if test_df.empty:
        return html.P("No data"), html.P("")

    df = test_df.copy()
    if risk != "ALL":
        df = df[df["risk_level"] == risk]
    if geo != "ALL":
        df = df[df["Geography_Label"] == geo]
    if "retention_value" in df.columns:
        df = df[df["retention_value"] >= min_rv]
    if active == "Active":
        df = df[df["IsActiveMember"] == 1]
    elif active == "Inactive":
        df = df[df["IsActiveMember"] == 0]

    sort_col = sort_col if sort_col in df.columns else "retention_value"
    df = df.sort_values(sort_col, ascending=False).head(100)

    show_cols = [c for c in ["CustomerId", "Age", "Geography_Label", "Balance",
                              "clv_estimated", "churn_probability", "retention_value",
                              "risk_level", "recommended_action"] if c in df.columns]

    header = html.Tr([html.Th(c.replace("_Label", "").replace("_", " ").title(),
                               style={"color": AMBER, "padding": "6px", "fontSize": "12px"})
                       for c in show_cols])
    rows = []
    for _, row in df.iterrows():
        risk_val = str(row.get("risk_level", ""))
        row_color = "#3a0000" if risk_val == "HIGH" else ("#3a2a00" if risk_val == "MEDIUM" else CARD_BG)
        rows.append(html.Tr([
            html.Td(
                str(row[c])[:10] if c == "CustomerId" else
                f"€{row[c]:,.0f}" if c in ("Balance", "clv_estimated", "retention_value") else
                f"{row[c]:.1%}" if c == "churn_probability" else str(row[c])[:40],
                style={"padding": "5px", "color": TEXT, "fontSize": "11px"}
            )
            for c in show_cols
        ], style={"backgroundColor": row_color}))

    total_rv = df["retention_value"].sum() if "retention_value" in df.columns else 0
    insight = html.P(
        f"Calling these {len(df)} customers could save approximately €{total_rv:,.0f} "
        f"in annual revenue (assuming 30% retention success).",
        style={"color": AMBER, "margin": "0"},
    )

    table = html.Table(
        [header] + rows,
        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "12px"}
    )
    return table, insight


@app.callback(
    Output("download-csv", "data"),
    Input("btn-export", "n_clicks"),
    State("filter-risk", "value"),
    State("filter-geo", "value"),
    State("filter-rv", "value"),
    State("filter-active", "value"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, risk, geo, min_rv, active):
    if test_df.empty:
        return None
    df = test_df.copy()
    if risk != "ALL": df = df[df["risk_level"] == risk]
    if geo != "ALL": df = df[df["Geography_Label"] == geo]
    if "retention_value" in df.columns: df = df[df["retention_value"] >= min_rv]
    if active == "Active": df = df[df["IsActiveMember"] == 1]
    elif active == "Inactive": df = df[df["IsActiveMember"] == 0]
    return dcc.send_data_frame(df.to_csv, "churnshield_export.csv", index=False)


@app.callback(
    Output("customer-profile", "children"),
    Output("churn-gauge", "children"),
    Output("risk-factors", "children"),
    Output("shap-waterfall", "children"),
    Input("customer-select", "value"),
    Input("btn-analyze", "n_clicks"),
    State("in-cs", "value"), State("in-age", "value"),
    State("in-geo", "value"), State("in-gen", "value"),
    State("in-bal", "value"), State("in-sal", "value"),
    State("in-ten", "value"), State("in-prod", "value"),
    State("in-active", "value"),
    prevent_initial_call=True,
)
def analyze_customer(sel_idx, n_clicks, cs, age, geo, gen, bal, sal, ten, prod, active_val):
    from dash import ctx

    if model is None:
        msg = html.Div([
            html.P("Live predictions require the model file.",
                   style={"color": AMBER, "fontWeight": "bold", "margin": "0"}),
            html.P("All other tabs show historical analysis.",
                   style={"color": SUBTEXT, "fontSize": "12px", "margin": "4px 0 0 0"}),
        ], style={**CARD_STYLE, "borderColor": AMBER})
        return msg, msg, msg, msg

    # Determine source
    if ctx.triggered_id == "customer-select" and sel_idx is not None:
        row = test_df.loc[sel_idx]
        geo_label = row.get("Geography_Label", "Unknown")
        gen_label = row.get("Gender_Label", "Unknown")
        from api.schemas import CustomerInput
        try:
            c = CustomerInput(
                CreditScore=int(row["CreditScore"]),
                Geography=geo_label,
                Gender=gen_label,
                Age=int(row["Age"]),
                Tenure=int(row["Tenure"]),
                Balance=float(row["Balance"]),
                NumOfProducts=int(row["NumOfProducts"]),
                HasCrCard=int(row["HasCrCard"]),
                IsActiveMember=int(row["IsActiveMember"]),
                EstimatedSalary=float(row["EstimatedSalary"]),
            )
        except Exception:
            return html.P("Error loading customer"), html.P(""), html.P(""), html.P("")
    else:
        from api.schemas import CustomerInput
        try:
            c = CustomerInput(
                CreditScore=int(cs or 650), Geography=geo or "France", Gender=gen or "Male",
                Age=int(age or 40), Tenure=int(ten or 5), Balance=float(bal or 0),
                NumOfProducts=int(prod or 1), HasCrCard=1,
                IsActiveMember=int(active_val if active_val is not None else 1),
                EstimatedSalary=float(sal or 50000),
            )
        except Exception as e:
            msg = html.P(f"Invalid input: {e}", style={"color": RED})
            return msg, msg, msg, msg

    from api.main import customer_to_features, get_action, get_shap_factors, _state as api_state
    if not api_state:
        api_state.update(D)
        api_state["thresholds"] = thresholds

    feat = customer_to_features(c)
    x = feat[feat_names].values.reshape(1, -1)
    prob = float(model.predict_proba(x)[0, 1])
    clv = float(feat["clv_estimated"])
    rv = round(clv * prob * 0.30, 2)
    risk_lv = "HIGH" if prob >= 0.6 else ("MEDIUM" if prob >= 0.35 else "LOW")
    risk_color = RED if risk_lv == "HIGH" else (AMBER if risk_lv == "MEDIUM" else GREEN)

    # Profile card
    profile = html.Div([
        html.H5("Customer Profile", style={"color": AMBER, "marginBottom": "10px"}),
        html.P(f"Geography: {c.Geography}", style={"color": TEXT, "margin": "4px 0"}),
        html.P(f"Gender: {c.Gender}", style={"color": TEXT, "margin": "4px 0"}),
        html.P(f"Age: {c.Age}", style={"color": TEXT, "margin": "4px 0"}),
        html.P(f"Credit Score: {c.CreditScore}", style={"color": TEXT, "margin": "4px 0"}),
        html.P(f"Tenure: {c.Tenure} years", style={"color": TEXT, "margin": "4px 0"}),
        html.P(f"Balance: €{c.Balance:,.0f}", style={"color": TEXT, "margin": "4px 0"}),
        html.P(f"Products: {c.NumOfProducts}", style={"color": TEXT, "margin": "4px 0"}),
        html.P(f"Active: {'Yes' if c.IsActiveMember else 'No'}", style={"color": TEXT, "margin": "4px 0"}),
        html.P(f"Salary: €{c.EstimatedSalary:,.0f}", style={"color": TEXT, "margin": "4px 0"}),
        html.Hr(style={"borderColor": "#333"}),
        html.P(f"CLV Estimate: €{clv:,.2f}", style={"color": GREEN, "fontWeight": "bold"}),
        html.P(f"Retention Value: €{rv:,.2f}", style={"color": AMBER, "fontWeight": "bold"}),
    ])

    # Gauge
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        title={"text": "Churn Probability", "font": {"color": TEXT}},
        number={"suffix": "%", "font": {"color": risk_color, "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT},
            "bar": {"color": risk_color},
            "steps": [
                {"range": [0, 35], "color": "#0a2a0a"},
                {"range": [35, 60], "color": "#2a1a00"},
                {"range": [60, 100], "color": "#2a0000"},
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "value": BIZ_THRESH * 100},
        }
    ))
    gauge_fig.update_layout(
        paper_bgcolor=CARD_BG, font_color=TEXT, height=280,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    gauge_content = html.Div([
        dcc.Graph(figure=gauge_fig, config={"displayModeBar": False}),
        html.P(f"This customer has a {prob:.1%} probability of leaving within 30 days.",
               style={"color": SUBTEXT, "fontSize": "12px", "textAlign": "center"}),
    ])

    # Risk factors
    factors = get_shap_factors(feat, feat_names)
    action = get_action(feat)
    risk_content = html.Div([
        html.H5("Why is this customer at risk?", style={"color": AMBER}),
        html.Div([
            html.Div([
                html.Span("↑ " if "increases" in f else "↓ ", style={"color": RED if "increases" in f else GREEN}),
                html.Span(f.split(" (")[0], style={"color": TEXT}),
            ], style={"margin": "6px 0"})
            for f in factors
        ]),
        html.Hr(style={"borderColor": AMBER}),
        html.H6("Recommended Action", style={"color": AMBER}),
        html.P(action, style={"color": TEXT, "backgroundColor": "#2a1a00",
                               "padding": "8px", "borderRadius": "4px",
                               "border": f"1px solid {AMBER}"}),
        html.Div([
            html.Span("Risk Level: ", style={"color": SUBTEXT}),
            html.Span(risk_lv, style={"color": risk_color, "fontWeight": "bold"}),
        ]),
    ])

    # SHAP waterfall (simplified bar)
    try:
        inner = model
        x_shap = x.copy()
        if hasattr(model, "named_steps"):
            inner = model.named_steps.get("clf", model)
            scaler = model.named_steps.get("scaler", None)
            x_shap = scaler.transform(x) if scaler else x

        explainer_bundle = D.get("explainer_bundle")
        if explainer_bundle:
            expl = explainer_bundle["explainer"]
            sv = expl(x_shap)
            if hasattr(sv, "values"):
                shap_arr = sv.values[0]
                if shap_arr.ndim == 2:
                    shap_arr = shap_arr[:, 1]
            else:
                shap_arr = sv
                if isinstance(shap_arr, list):
                    shap_arr = shap_arr[1][0]
                else:
                    shap_arr = shap_arr[0]

            idx = np.argsort(np.abs(shap_arr))[::-1][:10]
            wf_feats = [feat_names[i] for i in idx]
            wf_vals = [float(shap_arr[i]) for i in idx]
            colors = [RED if v > 0 else GREEN for v in wf_vals]

            wf_fig = go.Figure(go.Bar(
                y=wf_feats[::-1], x=wf_vals[::-1],
                orientation="h", marker_color=colors[::-1],
            ))
            wf_fig.update_layout(
                title="SHAP Feature Contributions (Local Explanation)",
                paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT,
                xaxis_title="SHAP Value (impact on prediction)",
                height=350,
            )
            waterfall = dcc.Graph(figure=wf_fig, config={"displayModeBar": False})
        else:
            waterfall = html.P("SHAP explainer not available.", style={"color": SUBTEXT})
    except Exception as e:
        waterfall = html.P(f"SHAP error: {str(e)[:100]}", style={"color": SUBTEXT})

    return profile, gauge_content, risk_content, waterfall


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)
