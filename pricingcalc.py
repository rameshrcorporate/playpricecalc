import uuid
import re
import pandas as pd
import streamlit as st

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Cost Estimator", layout="wide")
st.title("Playground Work Cost Estimator")
# st.caption("Column 1: Items editor | Column 2: Price summary | Column 3: Image preview")

# ================== CONSTANTS ==================
GST_RATE = 18.0  # fixed GST %
COMPUTED_LABEL_GST = f"GST ({GST_RATE:.0f}%)"
COMPUTED_LABEL_TOTAL = "Grand Total"

# ================== SEED DATA ==================
def seeded_rows():
    return [
        {"uid": str(uuid.uuid4()), "Include": False, "Equipment/Work Detail": "Shed Work (Other Side)", "Price": 14000, "Comment": "Other Side", "Image URL": "https://i.postimg.cc/qMvMzq43/jswsheet.jpg"},
        {"uid": str(uuid.uuid4()), "Include": False, "Equipment/Work Detail": "Shed Work (Near Tree)", "Price": 11200, "Comment": "Near Tree", "Image URL": "https://i.postimg.cc/qMvMzq43/jswsheet.jpg"},
        {"uid": str(uuid.uuid4()), "Include": False, "Equipment/Work Detail": "Swing Separate", "Price": 22000, "Comment": "", "Image URL": "https://i.postimg.cc/5QvZwCSX/swing.jpg"},
        {"uid": str(uuid.uuid4()), "Include": False, "Equipment/Work Detail": "See Saw Separate", "Price": 9000, "Comment": "", "Image URL": "https://i.postimg.cc/FKB19YP9/see-saw-image.png"},
        {"uid": str(uuid.uuid4()), "Include": False, "Equipment/Work Detail": "Dora Swing Separate", "Price": 22000, "Comment": "", "Image URL": "https://i.postimg.cc/ZCzkJwJp/Dora-swing.jpg"},
        {"uid": str(uuid.uuid4()), "Include": False, "Equipment/Work Detail": "Slide Separate (Steel)", "Price": 30000, "Comment": "Steel", "Image URL": "https://i.postimg.cc/p9mxmy4F/Slideseparate-steel.jpg"},
        {"uid": str(uuid.uuid4()), "Include": False, "Equipment/Work Detail": "Slide Separate (FRP)", "Price": 40000, "Comment": "FRP", "Image URL": "https://i.postimg.cc/zGDWD8Pt/slide-frp.png"},
        {"uid": str(uuid.uuid4()), "Include": False, "Equipment/Work Detail": "Circular Swing", "Price": 22000, "Comment": "", "Image URL": "https://i.postimg.cc/wynCyVLK/circular-swing.png"},
        {"uid": str(uuid.uuid4()), "Include": False, "Equipment/Work Detail": "Combination Set Playground", "Price": 64000, "Comment": "", "Image URL": "https://i.postimg.cc/HjyDGq1T/combo-sing-seesaw-slide.jpg"},
        {"uid": str(uuid.uuid4()), "Include": False, "Equipment/Work Detail": "P Sand JCB Work (2 Units)", "Price": 16666, "Comment": "2 Unit", "Image URL": "https://i.postimg.cc/0QZ1sG6H/psand.png"},
        {"uid": str(uuid.uuid4()), "Include": False, "Equipment/Work Detail": "Repair Rainbow Ladder and other)", "Price": 5000, "Comment": "", "Image URL": "https://i.postimg.cc/DZpp6x4F/existingaddon.jpg"},
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
if "tv_include" not in st.session_state:
    st.session_state.tv_include = False
if "tv_name" not in st.session_state:
    st.session_state.tv_name = "Samsung TV 43"
if "tv_units" not in st.session_state:
    st.session_state.tv_units = 2
if "tv_price" not in st.session_state:
    st.session_state.tv_price = 26500

# ================== CUSTOM CSS FOR TEXT WRAPPING AND THUMBNAILS ==================
st.markdown("""
    <style>
    /* Wrap text in data_editor and dataframe */
    .stDataFrame [data-testid="stTable"] td,
    .stDataFrame [data-testid="stTable"] th {
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        text-overflow: initial !important;
        max-width: 300px;
    }
    .stDialog [data-testid="stTable"] td,
    .stDialog [data-testid="stTable"] th {
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        text-overflow: initial !important;
        max-width: 300px;
    }
    /* Style for thumbnail images */
    .thumbnail-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
    }
    .thumbnail {
        width: 100px;
        height: 100px;
        object-fit: cover;
        cursor: pointer;
        border: 2px solid #ddd;
        border-radius: 5px;
    }
    .thumbnail:hover {
        border-color: #007bff;
    }
    </style>
""", unsafe_allow_html=True)

# ================== MAIN LAYOUT: 3 COLUMNS ==================
col1, col2, col3 = st.columns([0.45, 0.20, 0.35])

# --- Column 1: Items Editor + TV Block ---
with col1:
    st.subheader("Play Equipments")
    # st.write("• Toggle **Include** to include/exclude. • Add rows at the bottom.")

    def update_line_items():
        """Callback to update session state when data_editor changes."""
        edited_df = st.session_state.editor["edited_rows"]
        df = st.session_state.line_items_df.copy()
        for row_idx, changes in edited_df.items():
            for col, val in changes.items():
                df.at[row_idx, col] = val
        df = ensure_columns(df)
        df = assign_uids(df)
        df = coerce_price_numeric(df)
        st.session_state.line_items_df = df

    editor_cfg = {
        "Include": st.column_config.CheckboxColumn("Include", default=True),
        "Equipment/Work Detail": st.column_config.TextColumn("Equipment/Work Detail", required=True),
        "Price": st.column_config.NumberColumn("Price", min_value=0, step=500),
        "Comment": st.column_config.TextColumn("Comment"),
        "uid": None,
        "Image URL": None,
    }
    column_order = ["Include", "Equipment/Work Detail", "Price", "Comment", "uid", "Image URL"]

    st.data_editor(
        st.session_state.line_items_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config=editor_cfg,
        column_order=column_order,
        disabled=("uid",),
        key="editor",
        on_change=update_line_items,
        height=450
    )
    
with col2:
    # Separate TV block
    st.markdown("### 📺 TV")
    st.session_state.tv_include = st.checkbox("Include TV", value=st.session_state.tv_include)
    st.session_state.tv_name = st.text_input("TV Model", value=st.session_state.tv_name)
    st.session_state.tv_units = st.number_input("TV Units", min_value=1, value=st.session_state.tv_units, step=1)
    st.session_state.tv_price = st.number_input("TV Price (per unit)", min_value=0, value=st.session_state.tv_price, step=500)

    calc_clicked = st.button("Calculate", type="primary")

# --- Column 3: Image Preview (Thumbnails) ---
# --- Column 3: Image Preview (Thumbnails) ---
# --- Column 3: Image Preview (Thumbnails) ---
with col3:
    st.subheader("Image Preview")

    @st.dialog("Image Viewer")
    def show_full_image(img_url, item_name):
        """Display full-size image in a modal."""
        st.subheader(item_name)
        try:
            st.image(img_url, use_container_width=True)
        except Exception:
            st.write("Failed to load image.")

    prev_df = st.session_state.line_items_df.copy()
    prev_df["Include"] = prev_df["Include"].fillna(True).astype(bool)
    prev_df = prev_df[prev_df["Include"] == True].copy()

    # Prepare items for thumbnail grid (including TV if selected)
    items = []
    if not prev_df.empty:
        prev_df["Price"] = pd.to_numeric(prev_df["Price"], errors="coerce").fillna(0).astype(int)
        for _, item in prev_df.iterrows():
            items.append({
                "uid": item["uid"],
                "img_url": str(item.get("Image URL", "")).strip(),
                "item_name": f"{item['Equipment/Work Detail']} (₹{item['Price']:,})"
            })

    # Add TV to items if included
    if st.session_state.tv_include:
        tv_total = st.session_state.tv_units * st.session_state.tv_price
        items.append({
            "uid": "tv_image",
            "img_url": "https://i.postimg.cc/g0Tbw8FT/tv43inch.png",
            "item_name": f"{st.session_state.tv_name} (₹{tv_total:,})"
        })

    if items:
        # Create a grid of thumbnails (3 per row)
        num_cols = 3
        rows = [items[i:i + num_cols] for i in range(0, len(items), num_cols)]
        
        for row in rows:
            cols = st.columns(num_cols)
            for idx, (col, item) in enumerate(zip(cols, row)):
                with col:
                    img_url = item["img_url"]
                    item_name = item["item_name"]
                    if img_url:
                        try:
                            # Display thumbnail with click-to-enlarge
                            if st.button("View", key=f"thumbnail_{item['uid']}_{idx}"):
                                show_full_image(img_url, item_name)
                            st.image(img_url, width=100, caption=item_name[:30] + "..." if len(item_name) > 30 else item_name)
                        except Exception:
                            st.write("Image failed to load.")
                            st.caption(item_name[:30] + "..." if len(item_name) > 30 else item_name)
                    else:
                        st.write("No image available.")
                        st.caption(item_name[:30] + "..." if len(item_name) > 30 else item_name)
    else:
        st.write("No items included for preview.")
# --- Modal: Price Breakdown ---
@st.dialog("Price Breakdown")
def show_price_breakdown(breakdown, play_subtotal, gst_val, tv_total, grand_total):
    st.subheader("Detailed Price Breakdown")
    st.metric("Play Subtotal", f"₹{int(round(play_subtotal)):,}")
    st.metric(f"GST ({GST_RATE:.0f}%)", f"₹{gst_val:,}")
    st.metric("TV Price", f"₹{tv_total:,}")
    st.metric("Grand Total", f"₹{grand_total:,}")
    st.dataframe(breakdown, hide_index=True, height=420)

# --- Column 2: Price Summary ---
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
    tv_total = 0
    if st.session_state.tv_include:
        tv_total = st.session_state.tv_units * st.session_state.tv_price
        grand_total += tv_total
        breakdown_rows.append({
            "Equipment/Work Detail": st.session_state.tv_name,
            "Price": tv_total,
            "Comment": f"{st.session_state.tv_units} Unit(s) @ {st.session_state.tv_price} each (GST included)"
        })

    breakdown = pd.DataFrame(breakdown_rows + [
        {"Equipment/Work Detail": f"GST ({GST_RATE:.0f}%)", "Price": gst_val, "Comment": ""},
        {"Equipment/Work Detail": "Grand Total", "Price": grand_total, "Comment": ""},
    ])

    show_price_breakdown(breakdown, play_subtotal, gst_val, tv_total, grand_total)
