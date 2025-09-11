import pandas as pd
import streamlit as st
import numpy as np
import numpy_financial as npf
import plotly.express as px
import json

st.set_page_config(layout="wide")
st.title("⚡ Final DC-Coupled Solar + Battery Simulator")

# --- Upload/Download Input Parameters ---
st.sidebar.title("💾 Save or Load Inputs")

uploaded_params = st.sidebar.file_uploader("📤 Upload Parameters (.json)", type="json")
if uploaded_params:
    uploaded_config = json.load(uploaded_params)
    for k, v in uploaded_config.items():
        st.session_state[k] = v
    st.sidebar.success("Inputs loaded from file!")

# --- Upload Section ---
st.header("1. Upload Input Data")
col1, col2 = st.columns(2)
with col1:
    load_file = st.file_uploader("Load Profile (CSV)", type="csv")
with col2:
    pv_file = st.file_uploader("PV Output (CSV)", type="csv")

# --- System Inputs ---
st.header("2. System Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    dc_size = st.number_input("DC System Size (kW)", value=st.session_state.get("dc_size", 40.0))
    base_dc_size = st.number_input("Base DC Size in PV File (kW)", value=st.session_state.get("base_dc_size", 40.0))
with col2:
    inverter_capacity = st.number_input("Inverter Capacity (kW)", value=st.session_state.get("inverter_capacity", 30.0))
    inverter_eff = st.number_input("Inverter Efficiency (%)", value=st.session_state.get("inverter_eff", 98.0)) / 100
with col3:
    export_limit = st.number_input("Export Limit (kW)", value=st.session_state.get("export_limit", 30.0))
    import_limit = st.number_input("Import Limit (kW)", value=st.session_state.get("import_limit", 100.0))
# --- Utility Rates ---
st.header("3. Utility Tariff Inputs")
col1, col2 = st.columns(2)
with col1:
    import_rate = st.number_input("Import rate (£/kWh)", min_value=0.1, value=st.session_state.get("import_rate", 0.25),
                                  step=0.01)
with col2:
    export_rate = st.number_input("Export rate (£/kWh)", min_value=0.00,
                                  value=st.session_state.get("export_rate", 0.05), step=0.005)

# --- Financial Parameters ---
st.header("4. Financial Assumptions")
col1, col2, col3 = st.columns(3)
with col1:
    capex_per_kw = st.number_input("Capex (Cost per kW)", value=st.session_state.get("capex_per_kw", 650.0))
    cost_of_battery = st.number_input("Battery Capex (Cost per Battery)",
                                      value=st.session_state.get("cost_of_battery", 20000.0))
    o_and_m_rate = st.number_input("O&M Cost (% of Capex per year)",
                                   value=st.session_state.get("o_and_m_rate", 1.0)) / 100
with col2:
    apply_degradation = st.checkbox("Apply Degradation", value=st.session_state.get("apply_degradation", True))
    degradation_rate = st.number_input("Degradation per Year (%)",
                                       value=st.session_state.get("degradation_rate", 0.4)) / 100
    apply_battery_degardation = st.checkbox("Apply Battery Degradation",
                                            value=st.session_state.get("apply_battery_degradation", False))
    battery_degradation = st.number_input("Battery Degradation per Year(%)",
                                          value=st.session_state.get("battery_degradation", 0.5)) / 100
with col3:
    import_esc = st.number_input("Import Tariff Escalation (%/year)",
                                 value=st.session_state.get("import_esc", 2.0)) / 100
    export_esc = st.number_input("Export Tariff Escalation (%/year)",
                                 value=st.session_state.get("export_esc", 1.0)) / 100
    inflation = st.number_input("General Inflation Rate (%/year)", value=st.session_state.get("inflation", 3.0)) / 100

# --- Battery Inputs ---
st.header("5. Battery Configuration")
with st.expander("🔋 Battery Settings"):
    battery_qty = st.number_input("Battery Quantity", value=st.session_state.get("battery_qty", 1))
    battery_capacity = st.number_input("Battery Capacity per Unit (kWh)",
                                       value=st.session_state.get("battery_capacity", 50.0))
    dod = st.number_input("Depth of Discharge (%)", value=st.session_state.get("dod", 95.0)) / 100
    min_soc = st.number_input("Minimum SOC (%)", value=st.session_state.get("min_soc", 5.0)) / 100
    initial_soc = st.number_input("Initial SOC (%)", value=st.session_state.get("initial_soc", 100.0)) / 100
    c_rate = st.number_input("Battery C-rate", value=st.session_state.get("c_rate", 0.5))
    battery_eff = st.number_input("Battery Round-Trip Efficiency (%)",
                                  value=st.session_state.get("battery_eff", 96.0)) / 100

# --- Save Current Input Parameters ---
if st.sidebar.button("📥 Save Inputs"):
    input_params = {
        "dc_size": dc_size,
        "base_dc_size": base_dc_size,
        "inverter_capacity": inverter_capacity,
        "inverter_eff": inverter_eff*100,
        "export_limit": export_limit,
        "import_limit": import_limit,
        "import_rate": import_rate,
        "export_rate": export_rate,
        "capex_per_kw": capex_per_kw,
        "cost_of_battery": cost_of_battery,
        "o_and_m_rate": o_and_m_rate*100,
        "apply_degradation": apply_degradation,
        "degradation_rate": degradation_rate*100,
        "apply_battery_degradation": apply_battery_degardation,
        "battery_degradation": battery_degradation*100,
        "import_esc": import_esc*100,
        "export_esc": export_esc*100,
        "inflation": inflation*100,
        "battery_qty": battery_qty,
        "battery_capacity": battery_capacity,
        "dod": dod*100,
        "min_soc": min_soc*100,
        "initial_soc": initial_soc*100,
        "c_rate": c_rate,
        "battery_eff": battery_eff*100
    }

    json_string = json.dumps(input_params, indent=2)
    st.sidebar.download_button("⬇️ Download JSON", json_string, file_name="saved_inputs.json", mime="application/json")

# --- Simulation Execution ---
if load_file and pv_file:
    load_df = pd.read_csv(load_file)
    pv_df = pd.read_csv(pv_file)

    df = pd.DataFrame()
    df["Time"] = pd.to_datetime(load_df.iloc[:, 0], dayfirst=True)
    df["Load"] = load_df.iloc[:, 1]
    df["PV_base"] = pv_df.iloc[:, 1]
    df["Month"] = df["Time"].dt.to_period("M")
    df["Hour"] = df["Time"].dt.strftime("%H:%M")

    scaling = dc_size / base_dc_size
    df["PV Production"] = df["PV_base"] * scaling
    usable_capacity = battery_capacity * dod * battery_qty
    soc = usable_capacity * initial_soc
    avg_profile = df.groupby("Hour")["Load"].mean().reset_index(name="Average Load")
    peak_profile = df.groupby("Hour")["Load"].max().reset_index(name="Peak Load")
    total_discharge = 0
    charge_eff = np.sqrt(battery_eff)
    discharge_eff = np.sqrt(battery_eff)

    results = {
        "PV to Load": [],
        "PV to Load [AC]": [],
        "Battery Charge [Useful]": [],
        "Battery Charge [Raw PV Input]": [],
        "Battery Discharge [Useful]": [],
        "Battery Discharge to Load [AC]": [],
        "SOC (%)": [],
        "Import": [],
        "Export": [],
        "Export [Raw PV Input]": [],
        "Excess": [],
        "Battery Losses": [],
        "Inverter Losses": [],
        "Clipped": [],
        "PV Balance Sum Flows": [],
        "PV Balance Error": [],
        "Inverter AC Output": []
    }

    for i in df.index:
        pv = df.at[i, "PV Production"]
        load = df.at[i, "Load"]

        max_charge = battery_capacity * battery_qty * c_rate
        max_discharge = battery_capacity * battery_qty * c_rate

        # ---- PV to Load first (via DC Bus → Inverter → AC Load)
        pv_to_load_dc = min(pv, inverter_capacity, load / inverter_eff)
        pv_to_load_ac = pv_to_load_dc * inverter_eff
        remaining_load_ac = max(0, load - pv_to_load_ac)

        # ---- Battery Discharge (via DC Bus → Inverter → AC Load)
        max_available_discharge = soc - usable_capacity * min_soc
        max_available_discharge = max(0, max_available_discharge)

        required_raw_discharge = remaining_load_ac / discharge_eff
        raw_discharge = min(required_raw_discharge, max_discharge, max_available_discharge)
        useful_discharge = raw_discharge * discharge_eff
        soc -= raw_discharge
        soc = max(soc, usable_capacity * min_soc)
        total_discharge += useful_discharge

        # ---- PV to Battery Charge
        remaining_pv_dc = max(0, pv - pv_to_load_dc)
        max_possible_charge = usable_capacity - soc
        raw_charge = min(remaining_pv_dc, max_charge, max_possible_charge / charge_eff)
        useful_charge = raw_charge * charge_eff
        soc += useful_charge

        # ---- Battery Losses
        battery_losses = raw_charge * (1 - charge_eff) + raw_discharge * (1 - discharge_eff)

        # ---- Remaining PV after Load and Battery charge
        remaining_pv_dc_after_charge = max(0, remaining_pv_dc - raw_charge)

        # ---- Precompute potential Export DC
        potential_export_dc = min(remaining_pv_dc_after_charge, export_limit / inverter_eff)

        # ---- Priority dispatch inside inverter (AC level first):

        # Priority 1: PV to Load AC
        pv_to_load_ac_final = min(pv_to_load_ac, inverter_capacity * inverter_eff)
        remaining_ac_capacity = max(0, inverter_capacity * inverter_eff - pv_to_load_ac_final)

        # Priority 2: Battery Discharge to Load AC
        batt_discharge_to_load_ac_final = min(useful_discharge * inverter_eff, remaining_ac_capacity)
        remaining_ac_capacity -= batt_discharge_to_load_ac_final

        # Priority 3: Export AC
        export_ac_final = min(potential_export_dc * inverter_eff, export_limit, remaining_ac_capacity)
        remaining_ac_capacity -= export_ac_final

        # ---- Now compute ACTUAL inverter_input_dc based on what really passed:
        e_inv_dc = (pv_to_load_ac_final / inverter_eff) + (batt_discharge_to_load_ac_final / inverter_eff) + (
                    export_ac_final / inverter_eff)
        e_use_ac = e_inv_dc * inverter_eff

        # ---- Inverter Losses — now correct:
        inv_losses = e_inv_dc * (1 - inverter_eff)

        # ---- Import Energy
        import_energy = max(0, load - (pv_to_load_ac_final + batt_discharge_to_load_ac_final))

        # ---- Clipping (correct DC Bus calculation)
        clipped = max(0, pv - pv_to_load_dc - raw_charge - potential_export_dc)

        # ---- Energy balance terms per timestep
        pv_to_battery_dc = raw_charge
        pv_to_export_dc = potential_export_dc
        energy_balance_sum = pv_to_load_dc + pv_to_battery_dc + pv_to_export_dc + clipped
        energy_balance_error = pv - energy_balance_sum

        # ---- Final Outputs
        pv_load = pv_to_load_dc  # PV to Load in DC → matches PV balance

        results["PV to Load"].append(pv_load)
        results["PV to Load [AC]"].append(pv_to_load_ac_final)
        results["Battery Charge [Useful]"].append(useful_charge)
        results["Battery Charge [Raw PV Input]"].append(pv_to_battery_dc)
        results["Battery Discharge [Useful]"].append(useful_discharge)
        results["Battery Discharge to Load [AC]"].append(batt_discharge_to_load_ac_final)
        results["SOC (%)"].append((soc / usable_capacity) * 100)
        results["Import"].append(min(import_energy, import_limit))
        results["Export"].append(export_ac_final)
        results["Export [Raw PV Input]"].append(pv_to_export_dc)
        results["Excess"].append(max(0, remaining_pv_dc_after_charge - potential_export_dc))
        results["Battery Losses"].append(battery_losses)
        results["Inverter Losses"].append(inv_losses)
        results["Clipped"].append(clipped)
        results["PV Balance Sum Flows"].append(energy_balance_sum)
        results["PV Balance Error"].append(energy_balance_error)
        results["Inverter AC Output"].append(e_use_ac)

    for key in results:
        df[key] = results[key]

    monthly = df.groupby("Month").agg({
        "Load": "sum", "PV Production": "sum", "PV to Load": "sum",
        "Battery Discharge [Useful]": "sum", "Import": "sum", "Export": "sum",
        "Excess": "sum", "Battery Losses": "sum", "Inverter Losses": "sum"
    }).rename(columns={
        "Load": "Load", "PV Production": "Production", "PV to Load": "Solar On-site",
        "Battery Discharge [Useful]": "Battery", "Import": "Grid", "Export": "Export",
        "Excess": "Excess", "Battery Losses": "Battery Losses", "Inverter Losses": "Inverter Losses"
    }).reset_index()

    monthly["Month"] = monthly["Month"].astype(str)

    with st.expander("📅 Monthly Summary Table", expanded=False):
        st.dataframe(monthly)

    # --- Financial Projection ---
    st.header("6. 25-Year Financial Results")
    initial_capex = (dc_size * capex_per_kw) + (battery_qty * cost_of_battery)
    years = list(range(26))
    degradation_factors = [(1 - degradation_rate) ** (y - 1) if apply_degradation and y > 0 else 1.0 for y in years]
    battery_degardation_factor = [(1 - battery_degradation) ** (y - 1) if apply_battery_degardation and y > 0 else 0.0
                                  for y in years]
    cashflow = []
    cumulative = -initial_capex

    for y in years:
        if y == 0:
            cashflow.append({
                "Year": 0,
                "System Price (£)": -initial_capex,
                "O&M Costs (£)": 0,
                "Net Bill Savings (£)": 0,
                "Export Income (£)": 0,
                "Annual Cash Flow (£)": -initial_capex,
                "Cumulative Cash Flow (£)": -initial_capex
            })
            continue
        total_pv = df['PV Production'].sum()
        total_load = df['Load'].sum()
        deg = degradation_factors[y]
        pv_prod = total_pv * deg
        base_self_use_ratio = (df['PV to Load'].sum() + df['Battery Discharge [Useful]'].sum()) / (
            df['PV Production'].sum())
        pv_to_load = pv_prod * base_self_use_ratio
        base_export_ratio = df['Export'].sum() / pv_prod
        pv_export = pv_prod * base_export_ratio
        renewable_fraction = ((df['PV to Load'].sum() + df['Battery Discharge [Useful]'].sum())/df['Load'].sum())*100
        import_required = total_load - pv_to_load
        import_to_load = (df['Import'].sum() / df['Load'].sum())*100
        yearly_savings = ((df['PV to Load'].sum() + df['Battery Discharge [Useful]'].sum()) * import_rate) + (
                    df['Export'].sum() * export_rate) - (initial_capex * o_and_m_rate)
        battery_to_load = df['Battery Discharge [Useful]'].sum()/df['Load'].sum()
        export_to_grid = (df['Export'].sum()/df['PV Production'].sum())*100

        imp_price = import_rate * ((1 + import_esc) ** (y - 1))
        exp_price = export_rate * ((1 + export_esc) ** (y - 1))

        savings = (total_load - import_required) * imp_price
        export_income = pv_export * exp_price
        om = initial_capex * o_and_m_rate * ((1 + inflation) ** (y - 1))

        annual_cashflow = savings + export_income - om
        cumulative += annual_cashflow

        cashflow.append({
            "Year": y,
            "System Price (£)": -initial_capex if y == 0 else 0,
            "O&M Costs (£)": -om if y > 0 else 0,
            "Net Bill Savings (£)": savings,
            "Export Income (£)": export_income,
            "Annual Cash Flow (£)": annual_cashflow,
            "Cumulative Cash Flow (£)": cumulative
        })

    fin_df = pd.DataFrame(cashflow)
    irr = npf.irr(fin_df['Annual Cash Flow (£)'])
    roi = (fin_df['Cumulative Cash Flow (£)'].iloc[-1] + initial_capex) / initial_capex

    payback = None
    payback_display = "Not achieved"
    for i in range(1, len(fin_df)):
        if fin_df.loc[i, 'Cumulative Cash Flow (£)'] >= 0:
            prev_cum = fin_df.loc[i - 1, 'Cumulative Cash Flow (£)']
            annual_cash = fin_df.loc[i, 'Annual Cash Flow (£)']
            if annual_cash != 0:
                payback = i - 1 + abs(prev_cum) / annual_cash
                years = int(payback)
                months = int(round((payback - years) * 12))
                payback_display = f"{years} years {months} months"
            break

    lcoe = initial_capex / sum([total_pv * d for d in degradation_factors[1:]])

    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Capex (£)", f"{initial_capex:,.2f}")
    col2.metric("Payback Period", payback_display)
    col3.metric("First Year Savings (£) ", f"{yearly_savings:.2f}")


    col4,col5,col6 = st.columns(3)
    col4.metric("IRR (%)", f"{irr * 100:.2f}")
    col5.metric("LCOE (£/kWh)", f"{lcoe:.2f}")
    col6.metric("ROI (%)", f"{roi * 100:.2f}")

    with st.expander("📋 Show Cash Flow Table"):
        st.dataframe(fin_df.style.format({
            "System Price (£)": "£{:,.2f}",
            "O&M Costs (£)": "£{:,.2f}",
            "Net Bill Savings (£)": "£{:,.2f}",
            "Export Income (£)": "£{:,.2f}",
            "Annual Cash Flow (£)": "£{:,.2f}",
            "Cumulative Cash Flow (£)": "£{:,.2f}"
        }))

    with st.expander("📊 Annual Summary (Metrics)"):
        total = df[["Load", "PV Production", "PV to Load", "Battery Discharge [Useful]", "Import", "Export", "Excess",
                    "Battery Losses", "Inverter Losses"]].sum()

        row1 = st.columns(4)
        row1[0].metric("🔌 Total Load (kWh)", f"{total['Load']:.2f}")
        row1[1].metric("🔄 Solar + Battery On-site (kWh)", f"{total['PV to Load'] + total['Battery Discharge [Useful]']:.2f}")
        row1[2].metric("⚡ Grid Import (kWh)", f"{total['Import']:.2f} ")
        row1[3].metric("📤 Exported  (kWh)", f"{total['Export']:.2f}")

        row2 = st.columns(4)
        row2[0].metric("☀️ PV Production  (kWh)", f"{total['PV Production']:.2f}")
        row2[1].metric("🔄 Solar + Battery On-site (%)",f"{renewable_fraction:.2f}%")
        row2[2].metric("⚡ Grid Import (kWh)",f"{import_to_load:.2f} %")
        row2[3].metric("📤 Exported (%)",f"{export_to_grid:.2f}%")

        row3 = st.columns(4)
        row3[0].metric("🌞 Solar On-site  (kWh)", f"{total['PV to Load']:.2f}")
        row3[1].metric("🔋 Battery Use  (kWh)", f"{total['Battery Discharge [Useful]']:.2f}")
        row3[2].metric("🗑️ Excess Energy  (kWh)", f"{total['Excess']:.2f}")
        row3[3].metric("🔻 Inverter Losses  (kWh)", f"{total['Inverter Losses']:.2f}")

        row4 = st.columns(4)
        row4[0].metric("🌞 Solar On-site (%)",f"{(total['PV to Load']/total['PV Production'])*100:.2f}%")
        row4[1].metric("🔋 Battery Use (%)",f"{(total['Battery Discharge [Useful]']/total['PV Production'])*100:.2f}%")
        row4[2].metric("🗑️ Excess Energy (%)", f"{(total['Excess']/total['PV Production'])*100:.2f}")
        row4[3].metric("🔻 Inverter Losses (%)",f"{(total['Inverter Losses']/total['PV Production'])*100:.2f}%")

        row5 = st.columns(4)
        row5[0].metric("🔻 Battery Losses (kWh)", f"{total['Battery Losses']:.2f}")
        row5[1].metric("🔻  Battery Losses (%)", f"{(total['Battery Losses']/total['PV Production'])*100:.2f}%")
        row5[2].metric("🔁 Battery Cycles", f"{total_discharge / usable_capacity:.2f}")
        row5[3].metric("🔋📈 Battery Utilization (%)",
                       f"{((total['Battery Discharge [Useful]']) / (usable_capacity * 365)) * 100:.2f}")

    with st.expander(" 📈Load Profile"):
        st.plotly_chart(px.line(avg_profile, x="Hour", y="Average Load", title="Average Load Over Time"),
                        use_container_width=True)
        st.plotly_chart(px.line(peak_profile, x="Hour", y="Peak Load", title="Peak Load Over Time"),
                        use_container_width=True)

    with st.expander("🔋 Battery Charts"):
        st.plotly_chart(px.line(df, x="Time", y="SOC (%)", title="Battery SOC Over Time"), use_container_width=True)
        st.plotly_chart(px.line(df, x="Time", y=["Battery Charge [Useful]", "Battery Discharge [Useful]"],
                                title="Battery Charge & Discharge"), use_container_width=True)

    with st.expander("☀️ Renewable Energy Charts"):
        st.plotly_chart(px.line(df, x="Time",
                                y=["Load", "PV Production", "PV to Load", "Battery Discharge [Useful]", "Import",
                                   "Export", "Excess"], title="Load vs System Flows"), use_container_width=True)
        st.plotly_chart(px.bar(monthly, x="Month", y=["Load", "Production", "Export", "Excess"], barmode="group",
                               title="Monthly Solar Use & Export"), use_container_width=True)

        # --- Download Results ---
    with st.expander("📥 Download Results"):
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "final_simulation.csv",
            "text/csv",
            key="download_final_result"
        )

    # --- Batch Simulation Toggle ---
    simulate_batch = st.radio(
        "Batch Simulation", ["No", "Yes"], index=0, horizontal=True
    )
    batch_inputs = []

    if simulate_batch == "Yes":
        with st.expander("📊 Batch Simulation (Compare Multiple Systems)", expanded=False):
            num_systems = st.number_input("Number of Systems", min_value=1, max_value=10, value=3)
            dod = st.number_input("Battery Depth of Discharge (%)", value=95.0) / 100
            export_lock = st.checkbox("🔒 Lock Export Limit for All Systems", value=True)
            default_export = st.number_input("Export Limit for Batch (kW)", value=30.0)

        for i in range(int(num_systems)):
            with st.expander(f"System {i + 1} Inputs", expanded=(i == 0)):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    dc = st.number_input(f"DC Size (kW) - System {i + 1}", key=f"dc_{i}", value=40.0)
                with col2:
                    ac = st.number_input(f"Inverter Size (kW) - System {i + 1}", key=f"ac_{i}", value=30.0)
                with col3:
                    batt = st.number_input(f"Battery Capacity (kWh) - System {i + 1}", key=f"batt_{i}", value=50.0)
                with col4:
                    crate = st.number_input(f"Battery C-rate - System {i + 1}", key=f"crate_{i}", value=0.5)
                    export = default_export if export_lock else st.number_input(
                        f"Export Limit (kW) - System {i + 1}", key=f"export_{i}", value=30.0
                    )

                batch_inputs.append({
                    "DC": dc,
                    "AC": ac,
                    "Batt": batt,
                    "C-rate": crate,
                    "Export": export
                })

        # --- Run Batch Simulation ---
        if load_file and pv_file:
            load_d = load_df
            pv_d = load_df

            time_series = pd.to_datetime(load_df.iloc[:, 0], dayfirst=True)
            load_values = load_df.iloc[:, 1]
            pv_base = pv_df.iloc[:, 1]
            load_sum = load_values.sum()

            results = []
            for i, config in enumerate(batch_inputs):
                scaling = config["DC"] / base_dc_size
                pv_scaled = pv_base * scaling
                usable_capacity = config["Batt"] * dod * battery_qty
                soc = usable_capacity * initial_soc

                discharge_total = 0
                solar_to_load = 0
                battery_to_load = 0
                import_total = 0
                export_total = 0
                excess_total = 0
                battery_loss = 0
                inverter_loss = 0

                for t in range(len(time_series)):
                    pv = pv_scaled[t]
                    load = load_values[t]
                    inverter_limit = config["AC"]
                    export_limit = config["Export"]

                    pv_to_load = min(pv, load)
                    solar_to_load += pv_to_load
                    remaining_load = max(0, load - pv_to_load)

                    # Battery discharge
                    max_discharge = config["Batt"] * config["C-rate"]
                    useful_discharge = min(remaining_load, max_discharge)
                    raw_discharge = useful_discharge / np.sqrt(battery_eff)
                    raw_discharge = min(raw_discharge, soc)
                    useful_discharge = raw_discharge * np.sqrt(battery_eff)
                    battery_to_load += useful_discharge
                    soc -= raw_discharge
                    discharge_total += useful_discharge
                    battery_loss += (raw_discharge - useful_discharge)

                    # Battery charge
                    available_for_charge = max(0, pv - pv_to_load)
                    max_charge = config["Batt"] * config["C-rate"]
                    charge_raw = min(max_charge, usable_capacity - soc, available_for_charge)
                    charge_useful = charge_raw * np.sqrt(battery_eff)
                    soc += charge_useful
                    battery_loss += (charge_raw - charge_useful)

                    # Inverter output
                    total_inverter_input = pv + useful_discharge
                    e_inv = min(total_inverter_input, inverter_limit)
                    e_use = e_inv * inverter_eff
                    inverter_loss += (e_inv - e_use)

                    # Import from grid
                    import_total += max(0, load - e_use)

                    # Export and excess
                    pv_after_load_charge = max(0, pv - pv_to_load - charge_raw)
                    export_cap = max(0, inverter_limit - pv_to_load)
                    export = min(pv_after_load_charge, export_limit, export_cap)
                    export_total += export
                    excess_total += max(0, pv_after_load_charge - export)

                ren_fraction = (solar_to_load + battery_to_load) / load_sum * 100
                battery_util = (discharge_total / (usable_capacity * 365)) * 100

                results.append({
                    "System": f"System {i + 1}",
                    "DC Size": config["DC"],
                    "AC Size": config["AC"],
                    "Battery Capacity": config["Batt"],
                    "Solar Used (kWh)": solar_to_load,
                    "Solar Used (%)": (solar_to_load / load_sum) * 100,
                    "Battery Used (kWh)": battery_to_load,
                    "Battery Used (%)": (battery_to_load / load_sum) * 100,
                    "Import (kWh)": import_total,
                    "Export (kWh)": export_total,
                    "Excess (kWh)": excess_total,
                    "Renewable Fraction (%)": ren_fraction,
                    "Battery Cycles": discharge_total / usable_capacity,
                    "Battery Utilization (%)": battery_util,
                    "Battery Losses (kWh)": battery_loss,
                    "Inverter Losses (kWh)": inverter_loss
                })

            result_df = pd.DataFrame(results)

            st.subheader("📊 Batch Results Table")
            st.dataframe(result_df)

            st.download_button(
                "📥 Download Batch Results as CSV",
                result_df.to_csv(index=False),
                "batch_simulation_results.csv",
                "text/csv",
                key="download_batch_result"
            )

            st.subheader("📈 Comparison Chart")
            fig = px.bar(
                result_df, x="System",
                y=["Solar Used (kWh)", "Battery Used (kWh)", "Import (kWh)", "Export (kWh)"],
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)


else:
    st.warning("Please upload both Load and PV files to proceed.")
