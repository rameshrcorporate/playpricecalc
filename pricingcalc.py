import uuid
import re
import pandas as pd
import streamlit as st

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Cost Estimator", layout="wide")
st.title("Playground Work Cost Estimator")
st.caption("Column 1: Items editor | Column 2: Price summary | Column 3: Image preview")

# ================== CONSTANTS ==================
GST_RATE = 18.0  # fixed GST %
COMPUTED_LABEL_GST = f"GST ({GST_RATE:.0f}%)"
COMPUTED_LABEL_TOTAL = "Grand Total"

# ================== SEED DATA ==================
def seeded_rows():
    # Your seeded data (with images for preview). Edit freely.
    return [
        {"uid": str(uuid.uuid4()), "Include": False,  "Equipment/Work Detail": "Shed Work (Other Side)",        "Price": 14000, "Comment": "Other Side", "Image URL": "https://i.postimg.cc/qMvMzq43/jswsheet.jpg"},
        {"uid": str(uuid.uuid4()), "Include": False,  "Equipment/Work Detail": "Shed Work (Near Tree)",          "Price": 11200, "Comment": "Near Tree", "Image URL": "https://i.postimg.cc/qMvMzq43/jswsheet.jpg"},
        {"uid": str(uuid.uuid4()), "Include": False,  "Equipment/Work Detail": "Swing Separate",                 "Price": 22000, "Comment": "",          "Image URL": "https://i.postimg.cc/5QvZwCSX/swing.jpg"},
        {"uid": str(uuid.uuid4()), "Include": False,  "Equipment/Work Detail": "Dora Swing Separate",            "Price": 22000, "Comment": "",          "Image URL": "https://i.postimg.cc/ZCzkJwJp/Dora-swing.jpg"},
        {"uid": str(uuid.uuid4()), "Include": False,  "Equipment/Work Detail": "Slide Separate (Steel)",         "Price": 30000, "Comment": "Steel",      "Image URL": "https://i.postimg.cc/p9mxmy4F/Slideseparate-steel.jpg"},
        {"uid": str(uuid.uuid4()), "Include": False,  "Equipment/Work Detail": "Slide Separate (FRP)",           "Price": 40000, "Comment": "FRP",        "Image URL": "https://i.postimg.cc/zGDWD8Pt/slide-frp.png"},
        {"uid": str(uuid.uuid4()), "Include": False,  "Equipment/Work Detail": "Circular Swing",                 "Price": 22000, "Comment": "",           "Image URL": "https://i.postimg.cc/wynCyVLK/circular-swing.png"},
        {"uid": str(uuid.uuid4()), "Include": False,  "Equipment/Work Detail": "Combination Set Playground",     "Price": 64000, "Comment": "",           "Image URL": "https://i.postimg.cc/HjyDGq1T/combo-sing-seesaw-slide.jpg"},
        {"uid": str(uuid.uuid4()), "Include": False,  "Equipment/Work Detail": "P Sand JCB Work (1 Units)",      "Price": 8333, "Comment": "1 Unit",     "Image URL": "https://i.postimg.cc/0QZ1sG6H/psand.png"},
        {"uid": str(uuid.uuid4()), "Include": False,  "Equipment/Work Detail": "Repair and Provision Existing as Addon",
         "Price": 5000, "Comment": "", "Image URL": "https://i.postimg.cc/DZpp6x4F/existingaddon.jpg"},
    ]

# ================== HELPERS ==================
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["uid", "Include", "Equipment/Work Detail", "Price", "Comment", "Image URL"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

def assign_uids(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["uid"].isna() | (df["uid"].astype(str).str.strip() == "")
    if mask.any():
        df.loc[mask, "uid"] = [str(uuid.uuid4()) for _ in range(mask.sum())]
    return df

def coerce_price_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0).astype(int)
    return df

# ================== SESSION STATE ==================
if "line_items_df" not in st.session_state:
    st.session_state.line_items_df = ensure_columns(pd.DataFrame(seeded_rows()))
if "selected_preview_uid" not in st.session_state:
    st.session_state.selected_preview_uid = None

# TV block defaults
if "tv_include" not in st.session_state:
    st.session_state.tv_include = True
if "tv_name" not in st.session_state:
    st.session_state.tv_name = "Samsung TV 43"
if "tv_units" not in st.session_state:
    st.session_state.tv_units = 1
if "tv_price" not in st.session_state:
    st.session_state.tv_price = 26500

# ================== MAIN LAYOUT: 3 COLUMNS ==================
col1, col2, col3 = st.columns([0.45, 0.35, 0.20])

# --- Column 1: Items Editor + TV Block ---
with col1:
    st.subheader("Items (Inline add/edit)")
    st.write("• Toggle **Include** to include/exclude. • Add rows at the bottom.")

    editor_cfg = {
        "Include": st.column_config.CheckboxColumn("Include", default=True),
        "Equipment/Work Detail": st.column_config.TextColumn("Equipment/Work Detail", required=True),
        "Price": st.column_config.NumberColumn("Price", min_value=0, step=500),
        "Comment": st.column_config.TextColumn("Comment"),
        "uid": None,
        "Image URL": None,
    }
    column_order = ["Include", "Equipment/Work Detail", "Price", "Comment", "uid", "Image URL"]

    edited_df = st.data_editor(
        st.session_state.line_items_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config=editor_cfg,
        column_order=column_order,
        disabled=("uid",),
        key="editor",
    )

    edited_df = ensure_columns(edited_df)
    edited_df = assign_uids(edited_df)
    st.session_state.line_items_df = edited_df.copy()

    # Separate TV block
    st.markdown("### 📺 Additional Item (Not a Play Item)")
    st.session_state.tv_include = st.checkbox("Include TV", value=st.session_state.tv_include)
    st.session_state.tv_name = st.text_input("TV Model", value=st.session_state.tv_name)
    st.session_state.tv_units = st.number_input("TV Units", min_value=1, value=st.session_state.tv_units, step=1)
    st.session_state.tv_price = st.number_input("TV Price (per unit)", min_value=0, value=st.session_state.tv_price, step=500)

    calc_clicked = st.button("Calculate", type="primary")

# --- Column 2: Price Summary ---
with col2:
    st.subheader("Price Summary")
    if calc_clicked:
        df = st.session_state.line_items_df.copy()
        df["Include"] = df["Include"].fillna(True).astype(bool)
        df = coerce_price_numeric(df)
        included = df[df["Include"]].copy()

        play_subtotal = 0
        breakdown_rows = []

        for _, row in included.iterrows():
            detail = str(row["Equipment/Work Detail"])
            price = int(row["Price"])
            comment = str(row.get("Comment") or "")

            # Unit-aware JCB logic
            base_units_match = re.search(r"\((\d+)\s*Unit", detail, flags=re.IGNORECASE)
            base_units = int(base_units_match.group(1)) if base_units_match else 1
            desired_units_match = re.search(r"(\d+)", comment)
            desired_units = int(desired_units_match.group(1)) if desired_units_match else base_units

            if base_units > 1 or "JCB Work" in detail:
                per_unit = price / max(base_units, 1)
                total_price = per_unit * desired_units
                play_subtotal += total_price
                breakdown_rows.append({
                    "Equipment/Work Detail": detail,
                    "Price": int(round(total_price)),
                    "Comment": f"{desired_units} Unit(s) @ {int(round(per_unit))} each"
                })
            else:
                play_subtotal += price
                breakdown_rows.append({
                    "Equipment/Work Detail": detail,
                    "Price": price,
                    "Comment": comment
                })

        # GST only on play items
        gst_val = round(play_subtotal * (GST_RATE / 100))
        grand_total = play_subtotal + gst_val

        # Add TV separately (excluded from GST)
        if st.session_state.tv_include:
            tv_total = st.session_state.tv_units * st.session_state.tv_price
            grand_total += tv_total
            breakdown_rows.append({
                "Equipment/Work Detail": st.session_state.tv_name,
                "Price": tv_total,
                "Comment": f"{st.session_state.tv_units} Unit(s) @ {st.session_state.tv_price} each (GST included)"
            })

        m1, m2, m3 = st.columns(3)
        m1.metric("Play Subtotal", f"₹{int(round(play_subtotal)):,}")
        m2.metric(f"GST ({GST_RATE:.0f}%)", f"₹{gst_val:,}")
        m3.metric("Grand Total", f"₹{grand_total:,}")

        breakdown = pd.DataFrame(breakdown_rows + [
            {"Equipment/Work Detail": f"GST ({GST_RATE:.0f}%)", "Price": gst_val, "Comment": ""},
            {"Equipment/Work Detail": "Grand Total", "Price": grand_total, "Comment": ""},
        ])
        st.dataframe(breakdown, hide_index=True, height=420)


# --- Column 3: Image Preview ---
with col3:
    st.subheader("Image Preview")
    prev_df = st.session_state.line_items_df.copy()
    prev_df["Include"] = prev_df["Include"].fillna(True).astype(bool)
    prev_df = prev_df[prev_df["Include"] == True].copy()

    if not prev_df.empty:
        prev_df["Price"] = pd.to_numeric(prev_df["Price"], errors="coerce").fillna(0).astype(int)
        labels = (prev_df["Equipment/Work Detail"] + " (₹" + prev_df["Price"].astype(str) + ")").tolist()
        uid_map = dict(zip(labels, prev_df["uid"].tolist()))

        if st.session_state.selected_preview_uid not in uid_map.values():
            st.session_state.selected_preview_uid = prev_df.iloc[0]["uid"]

        inv = {v: k for k, v in uid_map.items()}
        default_label = inv.get(st.session_state.selected_preview_uid, labels[0])

        chosen_label = st.selectbox("Choose item to preview", options=labels, index=labels.index(default_label))
        st.session_state.selected_preview_uid = uid_map[chosen_label]

        row = prev_df[prev_df["uid"] == st.session_state.selected_preview_uid].iloc[0]
        img_url = str(row.get("Image URL") or "").strip()
        if img_url:
            st.image(img_url, use_container_width=True)
