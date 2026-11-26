from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from algorithms.transportation import METHODS, run_all_methods, solve_modi
from route_map import build_map_html, default_coordinates


st.set_page_config(
    page_title="SCMS | Global Transportation Optimization",
    page_icon="🌍",
    layout="wide",
)

st.title("SCMS | Global Transportation Optimization")
st.caption("Transportation model · NWC · LCM · VAM · MODI · OpenStreetMap")


@st.cache_data(show_spinner=False)
def load_workbook(file_bytes: bytes):
    """Read and align the three worksheets from the uploaded SCMS workbook."""
    buffer = BytesIO(file_bytes)
    excel = pd.ExcelFile(buffer)
    required = {"Data_Bersih", "Supply_Demand", "Matriks_Biaya"}
    missing = required.difference(excel.sheet_names)
    if missing:
        raise ValueError(f"Worksheet tidak ditemukan: {', '.join(sorted(missing))}")

    data = pd.read_excel(excel, sheet_name="Data_Bersih")
    sd = pd.read_excel(excel, sheet_name="Supply_Demand")
    raw_cost = pd.read_excel(excel, sheet_name="Matriks_Biaya", index_col=0)

    raw_cost.index = raw_cost.index.astype(str).str.strip()
    raw_cost.columns = raw_cost.columns.astype(str).str.strip()
    raw_cost = raw_cost.drop(columns=["Supply (kg)"], errors="ignore")
    raw_cost = raw_cost.apply(pd.to_numeric, errors="coerce")

    # Keep only actual source/destination names represented in the cost matrix.
    source_table = sd[["Sumber", "Supply (kg)"]].dropna(subset=["Sumber"]).copy()
    source_table["Sumber"] = source_table["Sumber"].astype(str).str.strip()
    source_table = source_table[source_table["Sumber"].isin(raw_cost.index)]
    source_table = source_table.drop_duplicates(subset=["Sumber"], keep="first")

    destination_table = sd[["Tujuan", "Demand (kg)"]].dropna(subset=["Tujuan"]).copy()
    destination_table["Tujuan"] = destination_table["Tujuan"].astype(str).str.strip()
    destination_table = destination_table[destination_table["Tujuan"].isin(raw_cost.columns)]
    destination_table = destination_table.drop_duplicates(subset=["Tujuan"], keep="first")

    source_table["Supply (kg)"] = pd.to_numeric(source_table["Supply (kg)"], errors="coerce")
    destination_table["Demand (kg)"] = pd.to_numeric(destination_table["Demand (kg)"], errors="coerce")

    source_names = source_table["Sumber"].tolist()
    destination_names = destination_table["Tujuan"].tolist()

    supply = pd.Series(
        pd.to_numeric(source_table["Supply (kg)"], errors="coerce").to_numpy(),
        index=source_names,
        dtype="float64",
    )
    demand = pd.Series(
        pd.to_numeric(destination_table["Demand (kg)"], errors="coerce").to_numpy(),
        index=destination_names,
        dtype="float64",
    )

    costs = raw_cost.reindex(index=source_names, columns=destination_names)
    costs = costs.apply(pd.to_numeric, errors="coerce")

    if supply.isna().any() or demand.isna().any():
        raise ValueError("Terdapat supply atau demand yang tidak terbaca sebagai angka.")

    return data, source_table, destination_table, supply, demand, costs


@st.cache_data(show_spinner="Menghitung NWC, LCM, dan VAM...")
def calculate_all_methods(
    supply_items: tuple,
    demand_items: tuple,
    cost_values: tuple,
    source_names: tuple,
    destination_names: tuple,
):
    supply = pd.Series(dict(supply_items), dtype="float64").reindex(source_names)
    demand = pd.Series(dict(demand_items), dtype="float64").reindex(destination_names)
    costs = pd.DataFrame(cost_values, index=source_names, columns=destination_names)
    initial_results = run_all_methods(supply, demand, costs)
    # VAM is selected as MODI's starting feasible basis because it has the
    # lowest initial cost among the three heuristics for this SCMS workbook.
    final_modi = solve_modi(
        supply,
        demand,
        costs,
        initial_results["VAM"]["allocation"],
    )
    return initial_results, final_modi


@st.cache_data(show_spinner="Menjalankan MODI untuk NWC, LCM, dan VAM...")
def calculate_modi_each(
    supply_items: tuple,
    demand_items: tuple,
    cost_values: tuple,
    source_names: tuple,
    destination_names: tuple,
):
    """MODI dari tiap solusi awal (untuk peta). Bila basis awal tidak valid, pakai hasil VAM-MODI."""
    initial_results, vam_modi = calculate_all_methods(
        supply_items, demand_items, cost_values, source_names, destination_names
    )
    supply = pd.Series(dict(supply_items), dtype="float64").reindex(source_names)
    demand = pd.Series(dict(demand_items), dtype="float64").reindex(destination_names)
    costs = pd.DataFrame(cost_values, index=source_names, columns=destination_names)
    out = {}
    for code in ("NWC", "LCM", "VAM"):
        if code == "VAM":
            out[code] = {"final": vam_modi["allocation"],
                         "iterations": len(vam_modi["iterations"]), "reference": False}
            continue
        try:
            r = solve_modi(supply, demand, costs, initial_results[code]["allocation"])
            out[code] = {"final": r["allocation"], "iterations": len(r["iterations"]),
                         "reference": False}
        except Exception:
            out[code] = {"final": vam_modi["allocation"], "iterations": None, "reference": True}
    return out


# ---------------------------
# Upload and prepare workbook
# ---------------------------
st.sidebar.header("Data input")
uploaded_file = st.sidebar.file_uploader(
    "Upload SCMS Excel workbook",
    type=["xlsx"],
    help="Workbook harus memiliki sheet Data_Bersih, Supply_Demand, dan Matriks_Biaya.",
)

if uploaded_file is None:
    st.info("Unggah file Excel SCMS melalui panel sebelah kiri untuk membuka dashboard.")
    st.markdown(
        "**Worksheet yang dibutuhkan:** `Data_Bersih`, `Supply_Demand`, dan `Matriks_Biaya`."
    )
    st.stop()

try:
    (
        shipment_data,
        source_table,
        destination_table,
        supply,
        demand,
        cost_matrix,
    ) = load_workbook(uploaded_file.getvalue())
except Exception as exc:
    st.error(f"Gagal membaca workbook: {exc}")
    st.stop()

supply_total = int(round(supply.sum()))
demand_total = int(round(demand.sum()))
available_routes = int(cost_matrix.notna().to_numpy().sum())

if supply_total != demand_total:
    st.error(
        f"Model belum seimbang: supply {supply_total:,.0f} kg, "
        f"demand {demand_total:,.0f} kg."
    )
    st.stop()

try:
    supply_items = tuple((str(k), int(round(v))) for k, v in supply.items())
    demand_items = tuple((str(k), int(round(v))) for k, v in demand.items())
    cost_values = tuple(
        tuple(None if pd.isna(v) else float(v) for v in row)
        for row in cost_matrix.itertuples(index=False, name=None)
    )
    results, modi_result = calculate_all_methods(
        supply_items,
        demand_items,
        cost_values,
        tuple(cost_matrix.index.astype(str)),
        tuple(cost_matrix.columns.astype(str)),
    )
except Exception as exc:
    st.error(f"Perhitungan algoritma gagal: {exc}")
    st.stop()

# Sidebar navigation
page = st.sidebar.radio(
    "Navigasi",
    ["Overview", "Data Management", "Algorithm Analysis", "Distribution Map"],
)
st.sidebar.divider()
st.sidebar.caption(f"Sumber: {len(supply)} · Tujuan: {len(demand)} · Rute tersedia: {available_routes}")


# ---------------------------
# Overview
# ---------------------------
if page == "Overview":
    st.subheader("Distribution Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source nodes", f"{len(supply):,}")
    c2.metric("Destination nodes", f"{len(demand):,}")
    c3.metric("Shipment records", f"{len(shipment_data):,}")
    c4.metric("Available routes", f"{available_routes:,}")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Top 10 supply sources")
        st.bar_chart(source_table.set_index("Sumber")["Supply (kg)"].nlargest(10))
    with right:
        st.markdown("#### Top 10 destination demand")
        st.bar_chart(destination_table.set_index("Tujuan")["Demand (kg)"].nlargest(10))

    st.divider()
    st.markdown("#### Supply-demand balance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total supply", f"{supply_total:,.0f} kg")
    m2.metric("Total demand", f"{demand_total:,.0f} kg")
    m3.metric("Balance", "Balanced" if supply_total == demand_total else "Unbalanced")

    st.caption(
        f"Matriks memiliki {len(supply) * len(demand):,} kemungkinan pasangan rute; "
        f"{available_routes:,} rute memiliki biaya dan dapat dipertimbangkan. "
        "Sel biaya kosong diperlakukan sebagai rute terlarang, bukan biaya nol."
    )


# ---------------------------
# Data management
# ---------------------------
elif page == "Data Management":
    st.subheader("Data Management")
    t1, t2, t3 = st.tabs(["Shipment records", "Supply & demand", "Cost matrix"])
    with t1:
        st.dataframe(shipment_data, use_container_width=True, height=520)
    with t2:
        a, b = st.columns(2)
        with a:
            st.markdown("**Supply sources**")
            st.dataframe(source_table, use_container_width=True, height=520)
        with b:
            st.markdown("**Destination demand**")
            st.dataframe(destination_table, use_container_width=True, height=520)
    with t3:
        st.caption("Biaya dalam USD/kg sesuai matriks biaya pada workbook.")
        st.dataframe(cost_matrix, use_container_width=True, height=650)


# ---------------------------
# Algorithm analysis
# ---------------------------
elif page == "Algorithm Analysis":
    st.subheader("Transportation Algorithm Analysis")
    st.caption(
        "Ketiga metode menggunakan rute dengan biaya tersedia. "
        "Untuk matriks yang memiliki rute kosong, digunakan adaptasi dengan pemeriksaan kelayakan." 
    )

    comparison_rows = []
    for code, result in results.items():
        comparison_rows.append({
            "Algoritma": METHODS[code],
            "Total biaya (USD)": result["total_cost"],
            "Total dialokasikan (kg)": result["allocated_kg"],
            "Jumlah rute terpakai": result["nonzero_routes"],
        })
    comparison_rows.append({
        "Algoritma": "MODI (setelah VAM)",
        "Total biaya (USD)": modi_result["total_cost"],
        "Total dialokasikan (kg)": modi_result["allocated_kg"],
        "Jumlah rute terpakai": modi_result["nonzero_routes"],
    })
    comparison = pd.DataFrame(comparison_rows).sort_values("Total biaya (USD)")

    st.markdown("#### Perbandingan hasil")
    st.dataframe(
        comparison.style.format({
            "Total biaya (USD)": "${:,.2f}",
            "Total dialokasikan (kg)": "{:,.0f}",
            "Jumlah rute terpakai": "{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(comparison.set_index("Algoritma")["Total biaya (USD)"])

    st.divider()
    st.markdown("#### Solusi akhir setelah MODI")
    st.caption(
        "MODI dimulai dari solusi awal VAM (biaya awal terendah di antara tiga "
        "heuristik), lalu memperbaiki basis sampai semua indeks perbaikan "
        "rute nonbasis yang tersedia bernilai tidak negatif."
    )
    final_col1, final_col2, final_col3, final_col4 = st.columns(4)
    final_col1.metric("Biaya awal VAM", f"${modi_result['starting_cost']:,.2f}")
    final_col2.metric("Biaya akhir MODI", f"${modi_result['total_cost']:,.2f}")
    final_col3.metric("Penghematan", f"${modi_result['cost_reduction']:,.2f}")
    final_col4.metric("Iterasi MODI", f"{len(modi_result['iterations']):,}")

    saving_pct = (
        modi_result["cost_reduction"] / modi_result["starting_cost"] * 100
        if modi_result["starting_cost"] else 0.0
    )
    st.success(
        f"MODI menurunkan biaya sebesar {saving_pct:.2f}% dari solusi awal VAM. "
        f"Kondisi optimalitas terverifikasi: indeks perbaikan minimum "
        f"${modi_result['minimum_reduced_cost']:.8f}/kg (≥ 0)."
    )

    st.markdown("#### Riwayat iterasi MODI")
    if modi_result["iterations"].empty:
        st.info("Solusi awal sudah memenuhi kondisi optimalitas MODI; tidak diperlukan pivot.")
    else:
        st.dataframe(
            modi_result["iterations"].style.format({
                "Indeks perbaikan (USD/kg)": "${:,.6f}",
                "Theta (kg)": "{:,.0f}",
                "Biaya sebelum (USD)": "${:,.2f}",
                "Biaya sesudah (USD)": "${:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
            height=360,
        )

    with st.expander("Lihat potensial akhir dan indeks perbaikan"):
        u_table = modi_result["u_potentials"].rename_axis("Sumber").reset_index()
        v_table = modi_result["v_potentials"].rename_axis("Tujuan").reset_index()
        a, b = st.columns(2)
        with a:
            st.markdown("**Potensial sumber (uᵢ)**")
            st.dataframe(u_table, use_container_width=True, hide_index=True)
        with b:
            st.markdown("**Potensial tujuan (vⱼ)**")
            st.dataframe(v_table, use_container_width=True, hide_index=True)
        st.markdown("**Matriks indeks perbaikan (cᵢⱼ − uᵢ − vⱼ)**")
        st.dataframe(modi_result["reduced_costs"], use_container_width=True)

    with st.expander("Lihat matriks alokasi akhir MODI"):
        st.dataframe(
            modi_result["allocation"].style.format("{:,.0f}"),
            use_container_width=True,
            height=520,
        )

    modi_csv = modi_result["allocation"].to_csv(index=True).encode("utf-8-sig")
    st.download_button(
        "Download final MODI allocation matrix (CSV)",
        data=modi_csv,
        file_name="alokasi_akhir_MODI_SCMS.csv",
        mime="text/csv",
    )
    route_rows = []
    for source in modi_result["allocation"].index:
        for destination in modi_result["allocation"].columns:
            quantity = float(modi_result["allocation"].loc[source, destination])
            if quantity > 0:
                unit_cost = float(cost_matrix.loc[source, destination])
                route_rows.append({
                    "Sumber": source,
                    "Tujuan": destination,
                    "Alokasi (kg)": quantity,
                    "Biaya (USD/kg)": unit_cost,
                    "Subtotal (USD)": quantity * unit_cost,
                })
    route_csv = pd.DataFrame(route_rows).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download MODI positive routes (CSV)",
        data=route_csv,
        file_name="rute_positif_MODI_SCMS.csv",
        mime="text/csv",
    )

    st.divider()
    selected = st.selectbox(
        "Pilih algoritma untuk melihat detail alokasi",
        options=["NWC", "LCM", "VAM"],
        format_func=lambda x: METHODS[x],
    )
    result = results[selected]

    m1, m2, m3 = st.columns(3)
    m1.metric("Total freight cost", f"${result['total_cost']:,.2f}")
    m2.metric("Allocated quantity", f"{result['allocated_kg']:,.0f} kg")
    m3.metric("Used routes", f"{result['nonzero_routes']}")

    st.markdown("#### Allocation detail")
    st.dataframe(
        result["steps"].style.format({
            "Alokasi (kg)": "{:,.0f}",
            "Biaya (USD/kg)": "${:,.4f}",
            "Subtotal (USD)": "${:,.2f}",
            "Sisa supply sumber (kg)": "{:,.0f}",
            "Sisa demand tujuan (kg)": "{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=540,
    )

    csv_data = result["steps"].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download allocation steps (CSV)",
        data=csv_data,
        file_name=f"alokasi_{selected.lower()}_scms.csv",
        mime="text/csv",
    )


# ---------------------------
# Distribution map: peta interaktif (NWC / LCM / VAM, solusi awal vs MODI)
# ---------------------------
elif page == "Distribution Map":
    st.subheader("Interactive Distribution Map")
    st.caption(
        "Pilih metode (NWC, LCM, VAM) dan tahap (solusi awal atau setelah MODI) di panel peta. "
        "Garis biru adalah rute yang dipakai pada solusi optimal; garis oranye adalah rute "
        "solusi awal yang tidak dipakai lagi setelah MODI."
    )

    defaults = default_coordinates()
    coords = dict(defaults)

    with st.expander("Koordinat node (opsional: ubah atau lengkapi dengan CSV)"):
        st.caption(
            "Peta memakai koordinat representatif bawaan (kota kantor pusat produsen dan "
            "ibu kota negara tujuan), bukan alamat pabrik yang pasti. Untuk mengganti, unduh "
            "template, ubah latitude/longitude, lalu unggah kembali."
        )
        rows = []
        for jenis, names in (("sumber", supply.index), ("tujuan", demand.index)):
            for name in names:
                v = defaults.get((jenis, name))
                rows.append({
                    "jenis": jenis, "node": name,
                    "latitude": "" if v is None else v[0],
                    "longitude": "" if v is None else v[1],
                })
        st.download_button(
            "Download coordinate template (CSV)",
            data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
            file_name="coordinates_template_scms.csv",
            mime="text/csv",
        )
        coordinate_file = st.file_uploader(
            "Upload coordinates CSV (kolom: jenis, node, latitude, longitude)",
            type=["csv"],
            key="coordinates_upload",
        )
        if coordinate_file is not None:
            try:
                uploaded = pd.read_csv(coordinate_file)
                if not {"jenis", "node", "latitude", "longitude"}.issubset(uploaded.columns):
                    raise ValueError("Kolom wajib: jenis, node, latitude, longitude.")
                uploaded["jenis"] = uploaded["jenis"].astype(str).str.strip().str.lower()
                uploaded["node"] = uploaded["node"].astype(str).str.strip()
                uploaded["latitude"] = pd.to_numeric(uploaded["latitude"], errors="coerce")
                uploaded["longitude"] = pd.to_numeric(uploaded["longitude"], errors="coerce")
                uploaded = uploaded.dropna(subset=["latitude", "longitude"])
                if not uploaded["latitude"].between(-90, 90).all():
                    raise ValueError("Latitude harus berada antara -90 dan 90.")
                if not uploaded["longitude"].between(-180, 180).all():
                    raise ValueError("Longitude harus berada antara -180 dan 180.")
                for row in uploaded.itertuples(index=False):
                    kota = defaults.get((row.jenis, row.node), (0, 0, ""))[2]
                    coords[(row.jenis, row.node)] = (float(row.latitude), float(row.longitude), kota)
                st.success(f"{len(uploaded)} koordinat dari CSV dipakai.")
            except Exception as exc:
                st.error(f"CSV koordinat tidak dapat dipakai, koordinat bawaan dipakai: {exc}")

    try:
        modi_each = calculate_modi_each(
            supply_items,
            demand_items,
            cost_values,
            tuple(cost_matrix.index.astype(str)),
            tuple(cost_matrix.columns.astype(str)),
        )
        per_method = {
            code: {
                "initial": results[code]["allocation"],
                "final": modi_each[code]["final"],
                "iterations": modi_each[code]["iterations"],
                "reference": modi_each[code]["reference"],
            }
            for code in ("NWC", "LCM", "VAM")
        }
        actual_cost = None
        if "biaya_freight_usd" in shipment_data.columns:
            actual_cost = pd.to_numeric(shipment_data["biaya_freight_usd"], errors="coerce").sum()

        map_html, missing_nodes = build_map_html(
            supply, demand, cost_matrix, per_method, actual_cost, coords
        )
    except Exception as exc:
        st.error(f"Tidak dapat membuat peta: {exc}")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final MODI cost", f"${modi_result['total_cost']:,.2f}")
    m2.metric("Allocated volume", f"{modi_result['allocated_kg']:,.0f} kg")
    m3.metric("Allocated routes", f"{modi_result['nonzero_routes']:,}")
    if actual_cost:
        m4.metric("Hemat vs biaya aktual", f"{(1 - modi_result['total_cost'] / actual_cost) * 100:.2f}%")

    if missing_nodes:
        st.warning(
            f"Koordinat belum tersedia untuk {len(missing_nodes)} node "
            f"({', '.join(missing_nodes[:6])}{'...' if len(missing_nodes) > 6 else ''}). "
            "Rute yang ujungnya tidak punya koordinat tidak digambar; lengkapi lewat CSV."
        )

    components.html(map_html, height=780, scrolling=False)
    st.caption(
        "Garis mengikuti lintasan lingkaran besar, bukan jalur kapal atau pesawat sebenarnya. "
        "Peta memuat Leaflet dan tile OpenStreetMap dari internet (© OpenStreetMap contributors)."
    )
