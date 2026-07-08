

import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

st.set_page_config(
    page_title="Red Wine Quality Explorer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Red Wine Quality Explorer")
st.caption("Case study · EDA · Machine learning · Prediction playground · 3D exploration · Raw data")

# FEATURE METADATA
FEATURE_DESCRIPTIONS = {
    "fixed acidity": ("g/dm³", "Non-volatile acids (mainly tartaric) that give wine its base tartness."),
    "volatile acidity": ("g/dm³", "Acetic acid content; too high produces a vinegar-like taste."),
    "citric acid": ("g/dm³", "Adds freshness and a subtle citrus flavor in small amounts."),
    "residual sugar": ("g/dm³", "Sugar left over after fermentation stops."),
    "chlorides": ("g/dm³", "Salt content of the wine."),
    "free sulfur dioxide": ("mg/dm³", "Free SO2 that protects wine from oxidation and microbes."),
    "total sulfur dioxide": ("mg/dm³", "Free + bound SO2; noticeable in nose/taste at high levels."),
    "density": ("g/cm³", "Close to water's density, shifted by alcohol and sugar content."),
    "pH": ("0-14 scale", "Acidity/basicity; most wines sit between 3 and 4."),
    "sulphates": ("g/dm³", "Additive that boosts SO2 levels; acts as antioxidant/antimicrobial."),
    "alcohol": ("% by volume", "Alcohol content of the wine."),
}
TARGET = "quality"

# DATA LOADING (always uses the bundled dataset, no upload)
@st.cache_data
def load_data():
    default_path = Path(__file__).parent / "winequality-red.csv"
    return pd.read_csv(default_path)


@st.cache_resource
def train_default_model(df: pd.DataFrame):
    """A quick baseline model so the Prediction Playground works immediately."""
    features = [c for c in df.columns if c != TARGET]
    X, y = df[features], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return {
        "model": model,
        "task": "Regression",
        "scheme": None,
        "features": features,
        "scaler": None,
        "X_test": X_test,
        "y_test": y_test,
        "model_name": "Random Forest Regressor (default baseline)",
    }

df = load_data()
numeric_cols = [c for c in df.columns if c != TARGET]

# Sidebar – only global filter (no data source or upload)
with st.sidebar:
    st.header("Global Filter")
    st.caption("Applies to EDA, 3D Explorer, and Raw Data tabs.")
    q_min, q_max = int(df[TARGET].min()), int(df[TARGET].max())
    q_range = st.slider("Quality score range", q_min, q_max, (q_min, q_max))
    filtered_df = df[(df[TARGET] >= q_range[0]) & (df[TARGET] <= q_range[1])]
    st.metric("Rows in view", f"{len(filtered_df):,} / {len(df):,}")

    st.divider()
    st.caption("Built with Streamlit · scikit-learn · Plotly")

# initialize a session-state model slot with the default baseline
if "trained" not in st.session_state:
    st.session_state.trained = train_default_model(df)

# TABS
tab_case, tab_eda, tab_ml, tab_predict, tab_3d, tab_raw = st.tabs(
    [
        "Case Study",
        "EDA & Visualization",
        "Machine Learning",
        "Prediction Playground",
        "3D Explorer",
        "Raw Data & Export",
    ]
)

# TAB 1 — CASE STUDY
with tab_case:
    st.subheader("The business problem")
    st.markdown(
        """
        A winery's quality-control lab currently relies on human tasters to score each batch
        of red wine on a scale from 0 (very poor) to 10 (excellent). Sensory panels are
        expensive, slow, and somewhat subjective — two tasters can disagree on the same batch.

        Meanwhile, every batch already has cheap, fast, objective **physicochemical
        lab measurements** (acidity, sugar, sulphates, alcohol, etc.) taken during production.

        **The question this dataset lets us explore:** can we predict a wine's quality score
        directly from its lab chemistry, well enough to flag batches for extra tasting
        attention, guide blending decisions, or support pricing — without waiting on a full
        sensory panel every time?
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{len(df):,}")
    c2.metric("Features", f"{len(numeric_cols)}")
    c3.metric("Quality range", f"{df[TARGET].min()}–{df[TARGET].max()}")
    c4.metric("Avg. quality", f"{df[TARGET].mean():.2f}")

    st.divider()
    st.subheader("What each feature means")
    desc_df = pd.DataFrame(
        [{"Feature": k, "Unit": v[0], "Meaning": v[1]} for k, v in FEATURE_DESCRIPTIONS.items()]
    )
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Why this isn't a trivial prediction task")
    dist = df[TARGET].value_counts().sort_index().reset_index()
    dist.columns = ["quality", "count"]
    fig = px.bar(
        dist, x="quality", y="count", text="count",
        color="quality", color_continuous_scale="RdYlGn",
        title="Quality score distribution — notice the imbalance",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"""
        - Over **{(df[TARGET].between(5, 6)).mean()*100:.0f}%** of wines score only a **5 or 6** —
          the extremes (3, 4, 7, 8) are rare. A model can look "accurate" while barely
          distinguishing anything beyond "average."
        - This imbalance is why the **Machine Learning** tab lets you frame the problem either
          as **regression** (predict the exact 3–8 score) or **classification** (predict a
          coarser Good/Not-Good or Low/Medium/High bucket), so you can compare trade-offs.
        - There's no missing data in this dataset, but several features are **skewed** and a
          few contain outliers — explored next in the EDA tab.
        """
    )

    st.divider()
    st.subheader("Questions this app helps answer")
    st.markdown(
        """
        1. Which chemical properties correlate most strongly with perceived quality?
        2. Do high-alcohol wines really score better, and how consistent is that pattern?
        3. How well can a model predict quality from chemistry alone — and where does it fail?
        4. Given a *new* wine's lab readings, what quality would we expect (playground)?
        5. Do the top predictive features separate high vs. low quality wines visually in 3D?
        """
    )

# TAB 2 — EDA & VISUALIZATION
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    st.caption("Reflects the global quality filter set in the sidebar.")

    with st.expander("Summary statistics", expanded=False):
        st.dataframe(filtered_df.describe().T.style.format("{:.3f}"), use_container_width=True)

    st.markdown("#### Feature distributions")
    dist_col = st.selectbox("Choose a feature to inspect", numeric_cols, index=numeric_cols.index("alcohol"))
    colA, colB = st.columns([2, 1])
    with colA:
        fig = px.histogram(
            filtered_df, x=dist_col, color=TARGET, nbins=40, marginal="box",
            color_discrete_sequence=px.colors.sequential.RdBu,
            title=f"Distribution of {dist_col} by quality",
        )
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig2 = px.box(
            filtered_df, x=TARGET, y=dist_col, color=TARGET,
            color_discrete_sequence=px.colors.sequential.RdBu,
            title=f"{dist_col} by quality (boxplot)",
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("#### Correlation heatmap")
    corr = filtered_df.corr(numeric_only=True)
    fig3 = px.imshow(
        corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
        title="Correlation matrix (all numeric features + quality)",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### What correlates most with quality?")
    corr_q = corr[TARGET].drop(TARGET).sort_values()
    fig4 = px.bar(
        corr_q, orientation="h", color=corr_q.values, color_continuous_scale="RdBu_r",
        labels={"value": "correlation with quality", "index": "feature"},
        title="Feature correlation with quality score",
    )
    fig4.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.markdown("#### Relationship explorer")
    c1, c2, c3 = st.columns(3)
    x_ax = c1.selectbox("X axis", numeric_cols, index=numeric_cols.index("alcohol"))
    y_ax = c2.selectbox("Y axis", numeric_cols, index=numeric_cols.index("volatile acidity"))
    size_ax = c3.selectbox("Bubble size (optional)", ["(none)"] + numeric_cols, index=0)
    fig5 = px.scatter(
        filtered_df, x=x_ax, y=y_ax, color=TARGET,
        size=None if size_ax == "(none)" else size_ax,
        color_continuous_scale="RdYlGn", opacity=0.75,
        title=f"{y_ax} vs {x_ax}, colored by quality",
    )
    st.plotly_chart(fig5, use_container_width=True)

# TAB 3 — MACHINE LEARNING
with tab_ml:
    st.subheader("Train a model")
    st.caption("Training always uses the full dataset (ignores the sidebar quality filter) for a representative model.")

    col1, col2 = st.columns(2)
    with col1:
        task = st.radio("Task framing", ["Regression (predict exact score 3-8)", "Classification (predict a category)"])
    scheme = None
    with col2:
        if task.startswith("Classification"):
            scheme = st.radio("Classification scheme", ["Binary: Good (≥7) vs Not Good", "Multi-class: Low / Medium / High"])

    sel_features = st.multiselect("Features to train on", numeric_cols, default=numeric_cols)
    if len(sel_features) == 0:
        st.warning("Select at least one feature.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    random_state = c2.number_input("Random state", value=42, step=1)
    scale_feats = c3.checkbox("Standardize features", value=False, help="Recommended for KNN and SVM.")

    if task.startswith("Regression"):
        model_choice = st.selectbox(
            "Model", ["Random Forest", "Gradient Boosting", "Linear Regression", "KNN", "SVR"]
        )
    else:
        model_choice = st.selectbox(
            "Model", ["Random Forest", "Gradient Boosting", "Logistic Regression", "KNN", "SVC"]
        )

    train_clicked = st.button("Train model", type="primary")

    if train_clicked:
        X = df[sel_features].copy()

        if task.startswith("Regression"):
            y = df[TARGET].copy()
        elif scheme.startswith("Binary"):
            y = np.where(df[TARGET] >= 7, "Good (≥7)", "Not Good (<7)")
        else:
            y = pd.cut(
                df[TARGET], bins=[0, 4, 6, 10], labels=["Low (3-4)", "Medium (5-6)", "High (7-8)"]
            ).astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        scaler = None
        if scale_feats:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=sel_features, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=sel_features, index=X_test.index)

        reg_models = {
            "Random Forest": RandomForestRegressor(n_estimators=300, random_state=int(random_state)),
            "Gradient Boosting": GradientBoostingRegressor(random_state=int(random_state)),
            "Linear Regression": LinearRegression(),
            "KNN": KNeighborsRegressor(n_neighbors=7),
            "SVR": SVR(),
        }
        clf_models = {
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=int(random_state)),
            "Gradient Boosting": GradientBoostingClassifier(random_state=int(random_state)),
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "SVC": SVC(probability=True),
        }

        is_regression = task.startswith("Regression")
        model = reg_models[model_choice] if is_regression else clf_models[model_choice]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.trained = {
            "model": model,
            "task": "Regression" if is_regression else "Classification",
            "scheme": scheme,
            "features": sel_features,
            "scaler": scaler,
            "X_test": X_test,
            "y_test": y_test,
            "model_name": model_choice,
        }

        st.success(f"Trained **{model_choice}** on {len(X_train):,} rows, tested on {len(X_test):,} rows.")

        if is_regression:
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{rmse:.3f}")
            m2.metric("MAE", f"{mae:.3f}")
            m3.metric("R²", f"{r2:.3f}")

            fig = px.scatter(
                x=y_test, y=y_pred, labels={"x": "Actual quality", "y": "Predicted quality"},
                title="Actual vs. predicted quality", opacity=0.6,
            )
            lo, hi = df[TARGET].min(), df[TARGET].max()
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("Precision", f"{prec:.3f}")
            m3.metric("Recall", f"{rec:.3f}")
            m4.metric("F1 (weighted)", f"{f1:.3f}")

            labels = sorted(pd.unique(y))
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            fig = px.imshow(
                cm, x=labels, y=labels, text_auto=True, color_continuous_scale="RdBu_r",
                labels={"x": "Predicted", "y": "Actual"}, title="Confusion matrix",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Feature importance")
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=sel_features).sort_values()
            fig_imp = px.bar(imp, orientation="h", title="Feature importances")
            fig_imp.update_layout(showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        elif hasattr(model, "coef_"):
            coef = model.coef_
            coef = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
            imp = pd.Series(coef, index=sel_features).sort_values()
            fig_imp = px.bar(imp, orientation="h", title="|Coefficient| magnitude")
            fig_imp.update_layout(showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("This model type doesn't expose feature importances directly.")

        st.caption("This model is now active in the Prediction Playground tab.")
    else:
        st.info("Configure options above and click **Train model**. Until then, the Prediction Playground uses a default Random Forest baseline.")

# TAB 4 — PREDICTION PLAYGROUND
with tab_predict:
    st.subheader("Predict the quality of a new wine")
    trained = st.session_state.trained
    st.info(f"Active model: **{trained['model_name']}** ({trained['task']}"
            + (f", {trained['scheme']}" if trained.get("scheme") else "") + ")")

    st.markdown("Adjust the sliders to describe a wine's lab chemistry:")
    input_vals = {}
    cols = st.columns(3)
    for i, feat in enumerate(trained["features"]):
        col = cols[i % 3]
        fmin, fmax, fmean = float(df[feat].min()), float(df[feat].max()), float(df[feat].mean())
        step = (fmax - fmin) / 100 if fmax > fmin else 0.01
        input_vals[feat] = col.slider(
            f"{feat} ({FEATURE_DESCRIPTIONS.get(feat, ('', ''))[0]})",
            min_value=round(fmin, 3), max_value=round(fmax, 3), value=round(fmean, 3),
            step=round(step, 4), key=f"play_{feat}",
        )

    predict_clicked = st.button("Predict quality", type="primary")

    if predict_clicked:
        x_new = pd.DataFrame([input_vals])[trained["features"]]
        if trained["scaler"] is not None:
            x_new = pd.DataFrame(trained["scaler"].transform(x_new), columns=trained["features"])

        model = trained["model"]
        pred = model.predict(x_new)[0]

        left, right = st.columns([1, 1])

        if trained["task"] == "Regression":
            pred_val = float(pred)
            if pred_val < 4.5:
                label, color = "Poor", "#c0392b"
            elif pred_val < 5.5:
                label, color = "Below Average", "#e67e22"
            elif pred_val < 6.5:
                label, color = "Average", "#f1c40f"
            elif pred_val < 7.5:
                label, color = "Good", "#27ae60"
            else:
                label, color = "Excellent", "#16a085"

            with left:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred_val,
                    number={"suffix": " / 8"},
                    gauge={
                        "axis": {"range": [3, 8]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [3, 4.5], "color": "#fadbd8"},
                            {"range": [4.5, 5.5], "color": "#fdebd0"},
                            {"range": [5.5, 6.5], "color": "#fcf3cf"},
                            {"range": [6.5, 7.5], "color": "#d5f5e3"},
                            {"range": [7.5, 8], "color": "#d1f2eb"},
                        ],
                    },
                    title={"text": "Predicted quality"},
                ))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"### Verdict: **{label}**")
        else:
            with left:
                st.markdown(f"### Predicted class: **{pred}**")
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(x_new)[0]
                    classes = model.classes_
                    fig = px.bar(
                        x=classes, y=proba, labels={"x": "class", "y": "probability"},
                        title="Class probabilities", color=proba, color_continuous_scale="RdYlGn",
                    )
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("#### How this wine compares to the dataset average")
            avg_vals = df[trained["features"]].mean()
            norm_input, norm_avg = [], []
            for f in trained["features"]:
                fmin, fmax = df[f].min(), df[f].max()
                span = (fmax - fmin) or 1
                norm_input.append((input_vals[f] - fmin) / span)
                norm_avg.append((avg_vals[f] - fmin) / span)
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(r=norm_avg + norm_avg[:1], theta=trained["features"] + [trained["features"][0]], name="Dataset average", fill="toself", opacity=0.5))
            radar.add_trace(go.Scatterpolar(r=norm_input + norm_input[:1], theta=trained["features"] + [trained["features"][0]], name="Your input", fill="toself", opacity=0.5))
            radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Normalized feature comparison")
            st.plotly_chart(radar, use_container_width=True)

# TAB 5 — 3D EXPLORER
with tab_3d:
    st.subheader("3D feature explorer")
    st.caption("Reflects the global quality filter set in the sidebar.")

    c1, c2, c3, c4 = st.columns(4)
    default_x = "alcohol" if "alcohol" in numeric_cols else numeric_cols[0]
    default_y = "volatile acidity" if "volatile acidity" in numeric_cols else numeric_cols[1]
    default_z = "sulphates" if "sulphates" in numeric_cols else numeric_cols[2]
    x_ax = c1.selectbox("X axis", numeric_cols, index=numeric_cols.index(default_x), key="x3d")
    y_ax = c2.selectbox("Y axis", numeric_cols, index=numeric_cols.index(default_y), key="y3d")
    z_ax = c3.selectbox("Z axis", numeric_cols, index=numeric_cols.index(default_z), key="z3d")
    color_by = c4.selectbox("Color by", [TARGET] + numeric_cols, index=0, key="c3d")

    fig3d = px.scatter_3d(
        filtered_df, x=x_ax, y=y_ax, z=z_ax, color=color_by,
        color_continuous_scale="RdYlGn" if color_by == TARGET else "Viridis",
        opacity=0.75, height=700,
        hover_data=numeric_cols + [TARGET],
        title=f"{x_ax} × {y_ax} × {z_ax}, colored by {color_by}",
    )
    fig3d.update_traces(marker=dict(size=4))
    st.plotly_chart(fig3d, use_container_width=True)

    corr3 = filtered_df[[x_ax, y_ax, z_ax, TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
    st.caption(
        f"Correlation with quality — {x_ax}: {corr3[x_ax]:.2f} · "
        f"{y_ax}: {corr3[y_ax]:.2f} · {z_ax}: {corr3[z_ax]:.2f}"
    )

# TAB 6 — RAW DATA & EXPORT
with tab_raw:
    st.subheader("Raw data")
    st.caption("Table below reflects the global quality filter set in the sidebar.")
    st.dataframe(filtered_df, use_container_width=True, height=420)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows shown", f"{len(filtered_df):,}")
    c2.metric("Columns", f"{filtered_df.shape[1]}")
    c3.metric("Missing values", f"{int(filtered_df.isna().sum().sum())}")

    st.divider()
    st.subheader("Export")

    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button(
            "Download filtered data (CSV)",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="wine_filtered.csv",
            mime="text/csv",
        )
    with e2:
        st.download_button(
            "Download full dataset (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="wine_full.csv",
            mime="text/csv",
        )
    with e3:
        summary_csv = filtered_df.describe().T.to_csv().encode("utf-8")
        st.download_button(
            "Download summary statistics (CSV)",
            data=summary_csv,
            file_name="wine_summary_stats.csv",
            mime="text/csv",
        )

    trained = st.session_state.trained
    if trained is not None:
        buf = io.BytesIO()
        pickle.dump(trained["model"], buf)
        st.download_button(
            f"Download trained model — {trained['model_name']} (.pkl)",
            data=buf.getvalue(),
            file_name="wine_quality_model.pkl",
            mime="application/octet-stream",
        )