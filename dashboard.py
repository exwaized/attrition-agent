# ============================================================
# dashboard.py — Streamlit HR Attrition Intelligence Dashboard
# ============================================================
# PURPOSE: HR-facing visual interface over FastAPI endpoints
# FLOW: Streamlit → FastAPI endpoints → display results
# RUN: streamlit run dashboard.py
# ============================================================

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# --- Step 1: Page config ---
st.set_page_config(
    page_title="Jio Attrition Intelligence",
    page_icon="[ALERT]",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_URL = "http://localhost:8000"

# --- Helper functions ---
def get_health():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return None

def get_weekly():
    try:
        r = requests.get(f"{BASE_URL}/report/weekly", timeout=10)
        return r.json()
    except Exception:
        return None

def get_employee(emp_id):
    try:
        r = requests.get(f"{BASE_URL}/explain/{emp_id}", timeout=30)
        if r.status_code == 200:
            return r.json()
        return {"error": r.json().get("detail", "Employee not found")}
    except Exception as e:
        return {"error": str(e)}

def get_department(dept):
    try:
        r = requests.get(f"{BASE_URL}/report/department/{dept}", timeout=10)
        return r.json()
    except Exception:
        return None

def get_budget_sim(tier):
    try:
        r = requests.get(f"{BASE_URL}/report/budget-sim?correction_type={tier}", timeout=10)
        return r.json()
    except Exception:
        return None

def get_audit():
    try:
        r = requests.get(f"{BASE_URL}/audit/recent?limit=50", timeout=10)
        return r.json()
    except Exception:
        return None

TIER_COLORS = {
    "CRITICAL": "#e74c3c",
    "HIGH":     "#e67e22",
    "MEDIUM":   "#f1c40f",
    "LOW":      "#2ecc71"
}

TIER_EMOJI = {
    "CRITICAL": "[FAIL]",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🟢"
}

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("Attrition Intelligence")
    st.markdown("---")

    health = get_health()
    if health and health.get("status") == "healthy":
        st.success("API: Connected [OK]")
    else:
        st.error("API: Disconnected ❌")
        st.info("Start: uvicorn api.main:app --reload --port 8000")

    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "📊 Executive Overview",
            "👤 Employee Deep Dive",
            "🏢 Department Heatmap",
            "💰 Budget Simulator",
            "📋 Audit Log"
        ]
    )

    st.markdown("---")
    st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# ============================================================
# PAGE 1 — Executive Overview
# ============================================================
if page == "📊 Executive Overview":
    st.title("📊 Executive Overview")
    st.markdown("Real-time attrition risk summary across all employees")
    st.markdown("---")

    weekly = get_weekly()

    if not weekly:
        st.error("Could not load data — is FastAPI running?")
        st.stop()

    summary = weekly["summary"]
    budget  = weekly["budget_summary"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Employees", f"{summary['total_employees']:,}")
    with col2:
        st.metric("[FAIL] Critical Risk", summary["critical_count"],
                  delta=f"{summary['critical_count']/summary['total_employees']*100:.1f}% of workforce",
                  delta_color="inverse")
    with col3:
        st.metric("💰 Budget Needed", f"Rs {budget['total_ev_urgent']/1e7:.1f}Cr")
    with col4:
        st.metric("📈 Net ROI", f"{budget['net_roi']}x")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Tier Distribution")
        tier_data = pd.DataFrame({
            "Tier":  ["Critical", "High", "Medium", "Low"],
            "Count": [summary["critical_count"], summary["high_count"],
                      summary["medium_count"], summary["low_count"]]
        })
        fig = px.pie(
            tier_data, values="Count", names="Tier",
            color="Tier",
            color_discrete_map={
                "Critical": "#e74c3c", "High": "#e67e22",
                "Medium":   "#f1c40f", "Low":  "#2ecc71"
            },
            hole=0.4
        )
        fig.update_layout(height=350, margin=dict(t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ROI Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=budget["net_roi"],
            title={"text": "Net ROI (x)"},
            delta={"reference": 3, "increasing": {"color": "#2ecc71"}},
            gauge={
                "axis": {"range": [0, 10]},
                "bar":  {"color": "#2ecc71"},
                "steps": [
                    {"range": [0, 2],  "color": "#fadbd8"},
                    {"range": [2, 5],  "color": "#fdebd0"},
                    {"range": [5, 10], "color": "#d5f5e3"},
                ],
            }
        ))
        fig.update_layout(height=350, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("[FAIL] Top Critical Employees")

    if weekly.get("top_critical"):
        df = pd.DataFrame(weekly["top_critical"])
        df["p_attrition"] = df["p_attrition"].round(3)
        df["ev"] = df["ev"].apply(lambda x: f"Rs {x:,.0f}")
        df["median_survival_months"] = df["median_survival_months"].apply(
            lambda x: f"{x:.0f}m" if x != float('inf') else "N/A"
        )
        df.columns = ["Employee ID", "Band", "Department",
                      "P(Attrition)", "Expected Value", "Survival"]
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("💰 Financial Impact")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Replacement Cost if No Action",
                  f"Rs {budget['replacement_cost_avoided']/1e7:.1f}Cr")
    with col2:
        st.metric("Intervention Budget Needed",
                  f"Rs {budget['total_ev_urgent']/1e7:.1f}Cr")
    with col3:
        st.metric("Expected Replacements",
                  f"{budget['expected_replacements']:.0f} employees")

# ============================================================
# PAGE 2 — Employee Deep Dive
# ============================================================
elif page == "👤 Employee Deep Dive":
    st.title("👤 Employee Deep Dive")
    st.markdown("Full risk report for any individual employee")
    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        emp_id = st.text_input("Enter Employee ID", placeholder="e.g. JIO10407")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search = st.button("🔍 Analyse", use_container_width=True)

    if emp_id and search:
        with st.spinner(f"Running full pipeline for {emp_id}..."):
            result = get_employee(emp_id)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            tier  = result["risk_tier"]
            color = TIER_COLORS.get(tier, "#gray")
            emoji = TIER_EMOJI.get(tier, "⚪")

            st.markdown(
                f"""<div style='background-color:{color};padding:15px;
                border-radius:10px;margin-bottom:20px'>
                <h2 style='color:white;margin:0'>{emoji} {tier} RISK — {emp_id}</h2>
                </div>""",
                unsafe_allow_html=True
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("P(Attrition)", f"{result['p_attrition']:.3f}")
            with col2:
                surv = result['median_survival_months']
                st.metric("Survival Months",
                          f"{surv:.0f}" if surv != float('inf') else "N/A")
            with col3:
                st.metric("Expected Value", f"Rs {result['ev']:,.0f}")
            with col4:
                st.metric("Routed To", result["routed_to"])

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Employee Profile")
                profile = result["employee_profile"]
                profile_df = pd.DataFrame([{
                    "Field": k.replace("_", " ").title(),
                    "Value": str(v)
                } for k, v in profile.items()])
                st.dataframe(profile_df, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Risk Drivers (SHAP)")
                if result.get("shap_drivers"):
                    drivers = result["shap_drivers"]
                    colors  = ["#e74c3c" if d["direction"] == "RISK FACTOR"
                               else "#2ecc71" for d in drivers]
                    fig = go.Figure(go.Bar(
                        x=[d["shap"] for d in drivers],
                        y=[d["label"] for d in drivers],
                        orientation="h",
                        marker_color=colors
                    ))
                    fig.update_layout(height=250, margin=dict(t=0, b=0),
                                      xaxis_title="SHAP Impact")
                    fig.add_vline(x=0, line_color="black", line_width=1)
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("🤖 AI Recommendation")
            rec = result.get("recommendation", {})

            if rec:
                st.info(f"**Analysis:** {rec.get('narrative', 'N/A')}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Immediate Actions:**")
                    for i, action in enumerate(rec.get("immediate_actions", []), 1):
                        st.markdown(f"{i}. {action}")
                    st.markdown(f"**Timeline:** {rec.get('timeline', 'N/A')}")
                    st.markdown(f"**Window:** {rec.get('intervention_window', 'N/A')}")
                with col2:
                    st.markdown("**Policy References:**")
                    for ref in rec.get("policy_references", []):
                        st.markdown(f"📋 {ref}")

# ============================================================
# PAGE 3 — Department Heatmap
# ============================================================
elif page == "🏢 Department Heatmap":
    st.title("🏢 Department Risk Heatmap")
    st.markdown("---")

    dept = st.selectbox("Select Department",
                        ["Network", "Technology", "Sales", "HR",
                         "Finance", "Operations", "IT"])

    with st.spinner(f"Loading {dept} data..."):
        data = get_department(dept)

    if not data:
        st.error("Could not load data")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Employees", data["total_employees"])
        with col2:
            st.metric("Avg P(Attrition)", f"{data['avg_p_attrition']:.3f}")
        with col3:
            st.metric("Critical Employees", len(data.get("critical_employees", [])))

        st.markdown("---")

        if data.get("heatmap_by_band"):
            hm_df = pd.DataFrame(data["heatmap_by_band"])
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(hm_df, x="band", y="avg_p_attrition",
                             color="avg_p_attrition",
                             color_continuous_scale="RdYlGn_r",
                             title="Avg P(Attrition) by Band")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(hm_df, x="band", y="high_risk_count",
                             color="high_risk_count",
                             color_continuous_scale="Reds",
                             title="High Risk Count by Band")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        if data.get("critical_employees"):
            st.markdown("---")
            st.subheader(f"[FAIL] Critical Employees — {dept}")
            crit_df = pd.DataFrame(data["critical_employees"])
            crit_df["ev"] = crit_df["ev"].apply(lambda x: f"Rs {x:,.0f}")
            crit_df["p_attrition"] = crit_df["p_attrition"].round(3)
            st.dataframe(crit_df, use_container_width=True, hide_index=True)

# ============================================================
# PAGE 4 — Budget Simulator
# ============================================================
elif page == "💰 Budget Simulator":
    st.title("💰 Budget Simulator")
    st.markdown("---")

    tier = st.select_slider(
        "Intervention Scope",
        options=["CRITICAL", "HIGH", "ALL"],
        value="CRITICAL"
    )

    with st.spinner("Running simulation..."):
        sim = get_budget_sim(tier)

    if not sim:
        st.error("Simulation failed")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Employees Covered", f"{sim['employees_affected']:,}")
        with col2:
            st.metric("Intervention Cost",
                      f"Rs {sim['total_intervention_cost']/1e7:.2f}Cr")
        with col3:
            st.metric("Replacement Avoided",
                      f"Rs {sim['replacement_cost_avoided']/1e7:.2f}Cr",
                      delta=f"+Rs {(sim['replacement_cost_avoided']-sim['total_intervention_cost'])/1e7:.2f}Cr net")
        with col4:
            st.metric("Net ROI", f"{sim['net_roi']}x",
                      delta=f"Payback {sim['payback_months']} months")

        st.markdown("---")
        st.subheader("Cost vs Benefit")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Intervention Cost",
            x=["Financial Impact"],
            y=[sim["total_intervention_cost"]],
            marker_color="#e74c3c"
        ))
        fig.add_trace(go.Bar(
            name="Replacement Cost Avoided",
            x=["Financial Impact"],
            y=[sim["replacement_cost_avoided"]],
            marker_color="#2ecc71"
        ))
        fig.update_layout(barmode="group", height=400,
                          yaxis_title="Amount (Rs)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Expected Retentions",
                      f"{sim['expected_retentions']:.0f} employees")
        with col2:
            rate = sim["expected_retentions"] / max(sim["employees_affected"], 1) * 100
            st.metric("Retention Success Rate", f"{rate:.0f}%")

# ============================================================
# PAGE 5 — Audit Log
# ============================================================
elif page == "📋 Audit Log":
    st.title("📋 Audit Log")
    st.markdown("---")

    audit = get_audit()

    if not audit or not audit.get("records"):
        st.info("No audit records yet — run the agent first")
    else:
        st.metric("Total Records", audit["total_records"])
        records  = audit["records"]
        audit_df = pd.DataFrame(records)

        col1, col2 = st.columns(2)
        with col1:
            tier_filter = st.multiselect(
                "Filter by Risk Tier",
                ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                default=["CRITICAL", "HIGH"]
            )
        with col2:
            route_filter = st.multiselect(
                "Filter by Route",
                ["SLACK_ALERT", "WEEKLY_DIGEST", "MANAGER_AWARENESS", "AUDIT_LOG_ONLY"],
                default=["SLACK_ALERT", "WEEKLY_DIGEST"]
            )

        filtered = audit_df[
            audit_df["risk_tier"].isin(tier_filter) &
            audit_df["routed_to"].isin(route_filter)
        ] if tier_filter and route_filter else audit_df

        display_cols = ["employee_id", "p_attrition", "ev",
                        "risk_tier", "routed_to", "created_at"]
        available    = [c for c in display_cols if c in filtered.columns]

        if not filtered.empty:
            display          = filtered[available].copy()
            display["ev"]    = display["ev"].apply(lambda x: f"Rs {x:,.0f}")
            display["p_attrition"] = display["p_attrition"].round(3)
            st.dataframe(display, use_container_width=True, hide_index=True)

            csv = filtered[available].to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"audit_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No records match selected filters")