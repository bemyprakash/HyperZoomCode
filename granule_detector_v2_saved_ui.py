import streamlit as st
import cv2
import numpy as np
import os
import glob
import pandas as pd
import shutil
from datetime import datetime
from skimage import measure
from streamlit_image_coordinates import streamlit_image_coordinates
import plotly.express as px
# import tkinter as tk
# from tkinter import filedialog

st.set_page_config(page_title="Granule Detector UI v2", layout="wide", page_icon="🔬")

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def ensure_odd(k):
    k = int(k)
    return k if k % 2 == 1 else k + 1

def isolate_color_channel(img_bgr, mode, custom_bgr=None):
    b, g, r = cv2.split(img_bgr)
    if mode == "Red":
        return cv2.subtract(r, g)
    elif mode == "Green":
        return cv2.subtract(g, r)
    elif mode == "Blue":
        return cv2.subtract(b, g)
    elif mode == "Custom" and custom_bgr is not None:
        tb, tg, tr = map(float, custom_bgr)
        mag = max((tb**2 + tg**2 + tr**2)**0.5, 1e-5)
        tb, tg, tr = tb/mag, tg/mag, tr/mag
        
        img_f = img_bgr.astype(np.float32)
        proj = img_f[:,:,0]*tb + img_f[:,:,1]*tg + img_f[:,:,2]*tr
        
        rej_b = img_f[:,:,0] - proj*tb
        rej_g = img_f[:,:,1] - proj*tg
        rej_r = img_f[:,:,2] - proj*tr
        dist = (rej_b**2 + rej_g**2 + rej_r**2)**0.5
        
        score = proj - dist
        return np.clip(score, 0, 255).astype(np.uint8)
    else:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def get_nucleus_mask(img_bgr, mode):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if mode == "Red":
        nucleus_mask = cv2.inRange(hsv, np.array([75, 40, 40]), np.array([105, 255, 255]))
    else:
        nucleus_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        
    k_nuc = np.ones((5, 5), np.uint8)
    nucleus_mask = cv2.morphologyEx(nucleus_mask, cv2.MORPH_CLOSE, k_nuc, iterations=2)
    nucleus_mask = cv2.morphologyEx(nucleus_mask, cv2.MORPH_OPEN, k_nuc, iterations=1)
    return nucleus_mask

def compute_granule_stats(props, img_bgr, intensity_channel=None):
    if not props:
        return dict(granule_count=0), []

    areas = np.array([p.area for p in props], dtype=float)
    img_area = (img_bgr.shape[0] * img_bgr.shape[1])
    
    intensities = []
    if intensity_channel is not None:
        for p in props:
            coords = p.coords
            vals = intensity_channel[coords[:, 0], coords[:, 1]]
            intensities.append(float(np.mean(vals)))
    else:
        intensities = [0.0] * len(props)
        
    summary = dict(
        granule_count=len(props),
        mean_area_px=round(float(np.mean(areas)), 2),
        median_area_px=round(float(np.median(areas)), 2),
        min_area_px=int(areas.min()),
        max_area_px=int(areas.max()),
        mean_intensity=round(float(np.mean(intensities)), 2) if intensities else 0.0,
        density_per1000px=round((len(props) / img_area) * 1000, 4),
    )
    
    per_granule = []
    for i, p in enumerate(props):
        per_granule.append(dict(
            granule_id=i+1,
            area_px=p.area,
            equiv_diameter_px=round(p.equivalent_diameter_area, 2),
            mean_intensity=round(intensities[i], 2),
            centroid_row=round(p.centroid[0], 1),
            centroid_col=round(p.centroid[1], 1),
        ))
    return summary, per_granule

# ==============================================================================
# PIPELINE FUNCTIONS
# ==============================================================================

def run_stage1_image(img_path, mode, custom_bgr=None, apply_clahe=False, manual_thresh=None):
    img = cv2.imread(img_path)
    if img is None: return None, None
    orig = img.copy()
    
    isolated = isolate_color_channel(img, mode, custom_bgr)
    
    # Optional Normalization to handle overly bright backgrounds
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        isolated = clahe.apply(isolated)
        
    isolated = cv2.medianBlur(isolated, 3)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(isolated, cv2.MORPH_TOPHAT, kernel)
    
    pixel_values = tophat.flatten()
    cands = pixel_values[pixel_values > 5].astype(np.float32)
    
    count = 0
    centroids_xy = []
    
    # Set default threshold to 55.0 if not manually specified
    if manual_thresh is None:
        manual_thresh = 55.0

    learned_thresh = float(manual_thresh)
    _, t_img = cv2.threshold(tophat, learned_thresh, 255, cv2.THRESH_BINARY)
    nl, lmap, sts, cens = cv2.connectedComponentsWithStats(t_img, connectivity=8)
    for i in range(1, nl):
        if sts[i, cv2.CC_STAT_AREA] >= 2:
            count += 1
            cx, cy = int(cens[i][0]), int(cens[i][1])
            cv2.circle(orig, (cx, cy), 3, (0, 255, 0), -1)
            centroids_xy.append((cx, cy))
                    
    cv2.putText(orig, f"S1 Count: {count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    res = {
        "image_path": img_path,
        "base_name": os.path.basename(img_path),
        "s1_count": count,
        "s1_centroids": centroids_xy,
        "s1_thresh": learned_thresh
    }
    return res, orig

def run_s2_core(img, intensity_ch, nuc_mask, block_size, c_value, bg_blur, min_area, max_area):
    bk = ensure_odd(bg_blur)
    blur = cv2.GaussianBlur(intensity_ch, (bk, bk), 0)
    bg_removed = cv2.subtract(intensity_ch, blur)
    smooth = cv2.GaussianBlur(bg_removed, (5, 5), 0)
    bs = ensure_odd(block_size)
    
    thresh = cv2.adaptiveThreshold(
        smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, int(c_value)
    )
    cleaned = thresh.copy()
    cleaned[nuc_mask == 255] = 0
    labels = measure.label(cleaned, connectivity=2)
    props_all = measure.regionprops(labels, intensity_image=intensity_ch)
    return [p for p in props_all if min_area < p.area < max_area], cleaned

def s2_objective(params, img_s, int_s, nuc_s, s1_cents_s, s1_count):
    block_size, c_value, bg_blur, min_area, max_area = params
    try:
        kept, _ = run_s2_core(img_s, int_s, nuc_s, block_size, c_value, bg_blur, min_area, max_area)
    except:
        return 999.0
    
    s2_count = len(kept)
    if s2_count == 0: return 999.0
    
    count_err = abs(s2_count - s1_count) / max(s1_count, 1)
    return float(count_err)

def autotune_random_search(img_full, mode, custom_bgr, s1_cents_full, s1_count, calls, bounds, scale=1.0, apply_clahe=False):
    if scale < 1.0:
        h, w = img_full.shape[:2]
        img_s = cv2.resize(img_full, (int(w*scale), int(h*scale)))
    else:
        img_s = img_full.copy()
        
    int_s = isolate_color_channel(img_s, mode, custom_bgr)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        int_s = clahe.apply(int_s)
        
    nuc_s = get_nucleus_mask(img_s, mode)
    
    if scale < 1.0:
        s1_cents_s = [(int(c[0]*scale), int(c[1]*scale)) for c in s1_cents_full]
    else:
        s1_cents_s = s1_cents_full
    
    rng = np.random.RandomState(42)
    best_loss = 999.0
    best_p = [21, -9, 101, 2, 200]
    
    for _ in range(calls):
        bs = ensure_odd(rng.randint(bounds['block'][0], bounds['block'][1]+1))
        cv = rng.uniform(bounds['cval'][0], bounds['cval'][1])
        bb = ensure_odd(rng.randint(bounds['blur'][0], bounds['blur'][1]+1))
        ma = rng.randint(bounds['min_area'][0], bounds['min_area'][1]+1)
        mx = rng.randint(bounds['max_area'][0], bounds['max_area'][1]+1)
        if mx <= ma: continue
        
        p = [bs, cv, bb, ma, mx]
        loss = s2_objective(p, img_s, int_s, nuc_s, s1_cents_s, s1_count)
        if loss < best_loss:
            best_loss, loss, best_p = loss, loss, p
            
    return dict(block_size=best_p[0], c_value=best_p[1], bg_blur=best_p[2], min_area=best_p[3], max_area=best_p[4], score=best_loss)

def run_stage2_image(img_path, mode, custom_bgr, s1_res, s2_params_auto=True, manual_params=None, calls=30, bounds=None, scale=1.0, apply_clahe=False):
    img = cv2.imread(img_path)
    if img is None: return None, None
    
    int_ch = isolate_color_channel(img, mode, custom_bgr)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        int_ch = clahe.apply(int_ch)
        
    nuc_mask = get_nucleus_mask(img, mode)
    
    if s2_params_auto:
        tuned = autotune_random_search(img, mode, custom_bgr, s1_res['s1_centroids'], s1_res['s1_count'], calls, bounds, scale=scale, apply_clahe=apply_clahe)
        if tuned['min_area'] > 50:
            tuned['min_area'] = 5
    else:
        tuned = manual_params
        
    kept, cleaned = run_s2_core(img, int_ch, nuc_mask, tuned['block_size'], tuned['c_value'], tuned['bg_blur'], tuned['min_area'], tuned['max_area'])
    
    colored_out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for p in kept:
        mr, mc, mxr, mxc = p.bbox
        cv2.rectangle(colored_out, (mc, mr), (mxc, mxr), (0, 255, 0), 2)
        
    final_count = len(kept)
    txt = f"S2 Count: {final_count}  block={tuned['block_size']} C={tuned['c_value']:.1f}"
    cv2.putText(colored_out, txt, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    stats, gran_list = compute_granule_stats(kept, img, int_ch)
    
    res = {
        "image_path": img_path,
        "base_name": os.path.basename(img_path),
        "s2_count": final_count,
        "tuned_params": tuned,
        "stats": stats,
        "gran_list": gran_list
    }
    return res, cv2.cvtColor(colored_out, cv2.COLOR_RGB2BGR)

# ==============================================================================
# UI DESIGN
# ==============================================================================

st.title("🔬 Interactive Granule Detection Tool v2")
st.markdown("End-to-end processing pipeline for locating granules with normalizations for bright/dim backgrounds.")

# ---- SIDEBAR ----
st.sidebar.header("📂 1. Input/Output Settings")

# def select_folder(title="Select Folder"):
#     root = tk.Tk()
#     root.withdraw()
#     root.wm_attributes('-topmost', 1)
#     folder_path = filedialog.askdirectory(master=root, title=title)
#     root.destroy()
#     return folder_path

def select_folder(title="Select Folder"):
    st.warning("Folder selection via dialog is not supported on Streamlit Cloud.")
    return None

if "in_dir" not in st.session_state: st.session_state.in_dir = "./"
if "out_dir" not in st.session_state: st.session_state.out_dir = "./pipeline_output_v2"

# if st.sidebar.button("Browse Input Folder..."):
#     f = select_folder("Select Input Images Folder")
#     if f: st.session_state.in_dir = f
st.sidebar.info("Enter folder path manually (no file dialog support)")

input_dir = st.sidebar.text_input("Input Images Folder", value=st.session_state.in_dir, help="The folder containing your raw .jpg or .png image files to be analyzed.")
st.session_state.in_dir = input_dir

st.sidebar.info("Enter output folder path manually")

out_dir = st.sidebar.text_input("Output Results Folder", value=st.session_state.out_dir, help="Where the final images, statistics, and plots will be saved.")
st.session_state.out_dir = out_dir

st.sidebar.header("🎨 2. Channel Configuration")
color_mode = st.sidebar.selectbox("Granule Color Mode", ["Red", "Green", "Blue", "Custom"], help="Select the color channel that best isolates your granules. 'Red' looks at red pixels, 'Custom' lets you pick a color from the image.")

if "custom_bgr" not in st.session_state:
    st.session_state.custom_bgr = (0, 0, 255) # default Red

if color_mode == "Custom":
    st.sidebar.markdown("**Pick a specific color from an image.**")
    images_avail = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    if images_avail:
        sample_img = cv2.imread(images_avail[0])
        sample_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        st.sidebar.write("Click on the image to select the granule color:")
        click_data = streamlit_image_coordinates(sample_rgb, key="color_picker", width=300)
        
        if click_data:
            x, y = click_data["x"], click_data["y"]
            r, g, b = sample_rgb[y, x]
            st.session_state.custom_bgr = (int(b), int(g), int(r)) # OpenCV uses BGR
            
        st.sidebar.markdown(f"**Selected Color (RGB):** `{st.session_state.custom_bgr[::-1]}`")
        st.sidebar.markdown(
            f'<div style="width: 50px; height: 50px; background-color: rgb{st.session_state.custom_bgr[::-1]}; border: 1px solid black;"></div>', 
            unsafe_allow_html=True
        )
    else:
        st.sidebar.warning("No images found in Input Folder to pick color from.")

st.sidebar.header("🔧 3. Advanced Normalization")
apply_clahe = st.sidebar.checkbox("Apply CLAHE Normalization", value=False, help="Enable this to normalize extreme bright/dim background fluorescence via Contrast Limited Adaptive Histogram Equalization. Use this if the base intensity is extremely high.")


st.sidebar.header("🔍 4. Stage 1 Settings")
global_s1_thresh = st.sidebar.number_input("Stage 1 Default Base Threshold", value=65.0, help="Initial intensity threshold for very simple images. A higher value requires pixels to be brighter to be considered a granule. Adjust if it's missing obvious granules (lower it) or picking up noise (raise it).")

st.sidebar.header("⚙️ 5. Stage 2 Tuning")
tune_mode = st.sidebar.radio("Stage 2 Tuning Mode", ["Auto (Random Search)", "Manual"], help="Auto will try many combinations to find the best settings for complex images. Manual lets you set them yourself.")

if tune_mode == "Auto (Random Search)":
    st.sidebar.markdown("*Mandatorily using Random Search instead of Bayesian.*")
    downscale_factor = st.sidebar.slider("Downscaling Factor (1.0 = Original Size)", 0.1, 1.0, 0.35, 0.05, help="Makes images smaller before tuning to save time. 0.35 means 35% of original size. Lower is faster but might miss tiny details.")
    rs_calls = st.sidebar.slider("Number of Random Search Calls", 10, 100, 30, help="How many random combinations to try. More calls = better tuning but takes longer.")
    st.sidebar.subheader("Search Bounds")
    c_b_min, c_b_max = st.sidebar.slider("Block Size Bound", 11, 101, (11, 41), step=2, help="The range of window sizes to look around a pixel. Larger blocks are better for uneven backgrounds, smaller blocks are better for small granules close together.")
    c_c_min, c_c_max = st.sidebar.slider("C Value Bound", -30, 0, (-18, -2), help="The range of offset values from the average. More negative values pick up fainter objects but also more noise.")
    
    area_mode = st.sidebar.radio("Area Constraint Mode", ["Hardcoded (15-300)", "Adaptive (Slider)"], index=0, help="Choose how to filter granule sizes during auto-tuning. Hardcoded usually works well.")
    if area_mode == "Hardcoded (15-300)":
        bounds = {'block': (c_b_min, c_b_max), 'cval': (c_c_min, c_c_max), 'blur': (51, 151), 'min_area': (15, 15), 'max_area':(300, 300)}
    else:
        c_a_min, c_a_max = st.sidebar.slider("Area Range Bound", 1, 1000, (1, 500), help="Only keep granules within this size range (in pixels). Helps remove tiny dust or large clumps.")
        bounds = {'block': (c_b_min, c_b_max), 'cval': (c_c_min, c_c_max), 'blur': (51, 151), 'min_area': (c_a_min, c_a_max), 'max_area':(c_a_min, c_a_max)}
else:
    downscale_factor, rs_calls, bounds = 1.0, 0, None
    st.sidebar.subheader("Manual Parameters")
    m_bs = st.sidebar.slider("Block Size", 11, 101, 21, step=2, help="Size of the local neighborhood for thresholding. A larger block handles uneven lighting better. Must be an odd number.")
    m_cv = st.sidebar.slider("C Value", -20.0, 0.0, -9.0, help="Offset constant. More negative picks up fainter structures but increases noise. Closer to 0 means stricter thresholding.")
    m_bb = st.sidebar.slider("Background Blur", 11, 151, 101, step=2, help="How much to blur the background before subtracting it. Larger values remove larger background gradients.")
    m_amin = st.sidebar.slider("Min Area", 1, 100, 2, help="Minimum number of pixels a granule can have. Filters out tiny noise specs.")
    m_amax = st.sidebar.slider("Max Area", 10, 500, 200, help="Maximum number of pixels a granule can have. Filters out huge artifacts or clumps.")
    manual_params = {"block_size": m_bs, "c_value": m_cv, "bg_blur": m_bb, "min_area": m_amin, "max_area": m_amax}

st.sidebar.header("🚀 6. Execution")
escalate_thresh = st.sidebar.number_input("Auto Escalate S1 count threshold", value=100, help="If Stage 1 finds more than this many potential granules, the image is considered 'complex' and is automatically queued for the more rigorous Stage 2 analysis.")

# ==============================================================================
# MAIN VIEW - STATE MANAGEMENT
# ==============================================================================
if "s1_results" not in st.session_state: st.session_state.s1_results = {}
if "s2_results" not in st.session_state: st.session_state.s2_results = {}
if "manual_escalations" not in st.session_state: st.session_state.manual_escalations = {}
if "per_image_overrides" not in st.session_state: st.session_state.per_image_overrides = {}
if "s1_overrides" not in st.session_state: st.session_state.s1_overrides = {}

tab1, tab2, tab3 = st.tabs(["▶️ Stage 1: Preliminary Analysis", "⚙️ Stage 2: Refinement", "📊 Results & Export"])

# ---- TAB 1: STAGE 1 ----
with tab1:
    st.markdown("### Stage 1: Fast Unsupervised Detector")
    st.info("💡 **NOTE:** This is Stage 1. Stage 1 images are either control images or have very less counts. K-Means clustering is used on a Top-Hat image to quickly gauge granule presence.")
    
    if st.button("Run Stage 1 on Input Folder", type="primary"):
        images = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
        if not images:
            st.error("No images found in the specified directory.")
        else:
            prog = st.progress(0)
            st.session_state.s1_results = {}
            st.session_state.s2_results = {} 
            st.session_state.manual_escalations = {}
            st.session_state.s1_overrides = {}
            for i, p in enumerate(images):
                res, img_out = run_stage1_image(p, color_mode, st.session_state.custom_bgr, apply_clahe=apply_clahe, manual_thresh=global_s1_thresh)
                if res:
                    st.session_state.s1_results[res["base_name"]] = {"res": res, "img": img_out, "path": p}
                prog.progress((i+1)/len(images))
            st.success(f"Stage 1 completed on {len(images)} images!")

    if st.session_state.s1_results:
        st.markdown("---")
        st.markdown("#### Stage 1 Output")
        
        cols = st.columns(3)
        for idx, (bname, data) in enumerate(st.session_state.s1_results.items()):
            c = data["res"]["s1_count"]
            auto_escalate = c >= escalate_thresh
            
            with cols[idx % 3]:
                st.image(cv2.cvtColor(data["img"], cv2.COLOR_BGR2RGB), caption=f"{bname} | Count: {c}")
                if auto_escalate:
                    st.warning(f"⚠️ Auto-escalated (count {c} >= {escalate_thresh})")
                    st.session_state.manual_escalations[bname] = True
                else:
                    st.success(f"✅ Control / Low Count (count {c} < {escalate_thresh})")
                    is_checked = st.checkbox("Manually move to Stage 2", key=f"esc_{bname}", value=st.session_state.manual_escalations.get(bname, False))
                    st.session_state.manual_escalations[bname] = is_checked

                with st.expander(f"⚙️ Overide Stage 1 Threshold"):
                    curr_th = st.session_state.s1_overrides.get(bname, data["res"]["s1_thresh"])
                    new_th = st.slider("Absolute Intensity Threshold", 1.0, 255.0, float(curr_th), key=f"s1_th_slider_{bname}", help="Override the global threshold for this specific image. Higher values mean only brighter pixels are kept.")
                    if st.button("Re-run Stage 1 specific to this image", key=f"s1_rr_btn_{bname}"):
                        st.session_state.s1_overrides[bname] = new_th
                        res_new, img_new = run_stage1_image(
                            data["path"], color_mode, st.session_state.custom_bgr, 
                            apply_clahe=apply_clahe, manual_thresh=new_th
                        )
                        st.session_state.s1_results[bname] = {"res": res_new, "img": img_new, "path": data["path"]}
                        st.rerun()

# ---- TAB 2: STAGE 2 ----
with tab2:
    st.markdown("### Stage 2: Classical Detector with Auto-Tuning")
    st.info("💡 **NOTE:** This is Stage 2. Adaptive thresholding maps the complex structures. Only escalated images (manual or auto) are processed here.")
    
    if st.button("Run Stage 2 on Escalated Images", type="primary"):
        escalated_list = [name for name, val in st.session_state.manual_escalations.items() if val and name in st.session_state.s1_results]
        
        if not escalated_list:
            st.warning("No images marked for Stage 2.")
        else:
            st.session_state.s2_results = {}
            st.session_state.per_image_overrides = {} 
            prog2 = st.progress(0)
            for i, name in enumerate(escalated_list):
                s1_data = st.session_state.s1_results[name]
                s2_auto = (tune_mode == "Auto (Random Search)")
                res2, img2 = run_stage2_image(
                    s1_data["path"], color_mode, st.session_state.custom_bgr,
                    s1_data["res"], s2_auto,
                    bounds=bounds, calls=rs_calls, manual_params=manual_params if not s2_auto else None, scale=downscale_factor, apply_clahe=apply_clahe
                )
                if res2:
                    st.session_state.s2_results[name] = {"res": res2, "img": img2}
                prog2.progress((i+1)/len(escalated_list))
            st.success(f"Stage 2 completed on {len(escalated_list)} images!")

    if st.session_state.s2_results:
        st.markdown("---")
        st.markdown("#### Stage 2 Output")
        
        cols2 = st.columns(2)
        for idx, (bname, data) in enumerate(st.session_state.s2_results.items()):
            with cols2[idx % 2]:
                st.image(cv2.cvtColor(data["img"], cv2.COLOR_BGR2RGB), caption=f"{bname} | Stage 2 Count: {data['res']['s2_count']}", use_column_width=True)
                
                with st.expander("📊 Show Statistics & Plots"):
                    st.json(data["res"]["stats"])
                    if data["res"]["tuned_params"]:
                        st.json(data["res"]["tuned_params"])
                        
                    df_g = pd.DataFrame(data["res"]["gran_list"])
                    if not df_g.empty:
                        fig_area = px.histogram(df_g, x="area_px", nbins=20, title="Granule Area Distribution")
                        st.plotly_chart(fig_area, use_container_width=True)
                        if "mean_intensity" in df_g.columns:
                            fig_int = px.histogram(df_g, x="mean_intensity", nbins=20, title="Granule Intensity Distribution", color_discrete_sequence=["indianred"])
                            st.plotly_chart(fig_int, use_container_width=True)

                with st.expander(f"⚙️ Manually Tune specific to '{bname}'"):
                    tp = data["res"].get("tuned_params", {})
                    curr_bs = st.session_state.per_image_overrides.get(bname, {}).get("block_size", tp.get("block_size", 21))
                    curr_cv = st.session_state.per_image_overrides.get(bname, {}).get("c_value", tp.get("c_value", -9))
                    curr_bb = st.session_state.per_image_overrides.get(bname, {}).get("bg_blur", tp.get("bg_blur", 101))
                    curr_min = st.session_state.per_image_overrides.get(bname, {}).get("min_area", tp.get("min_area", 2))
                    curr_max = st.session_state.per_image_overrides.get(bname, {}).get("max_area", tp.get("max_area", 200))
                    
                    bs_new = st.slider("Block Size", 11, 101, int(curr_bs), step=2, key=f"t_bs_{bname}", help="Size of the local neighborhood for adaptive thresholding. Larger handles uneven illumination better. (Odd numbers only)")
                    cv_new = st.slider("C Value", -40.0, 0.0, float(curr_cv), key=f"t_cv_{bname}", help="Offset constant. More negative values detect fainter objects but increase noise.")
                    bb_new = st.slider("Background Blur", 11, 151, int(curr_bb), step=2, key=f"t_bb_{bname}", help="Amount of blur for background subtraction. Larger values smooth out bigger background gradients.")
                    min_new = st.slider("Min Area", 1, 100, int(curr_min), key=f"t_min_{bname}", help="Minimum pixel area for a granule. Discards smaller noise dots.")
                    max_new = st.slider("Max Area", 10, 1000, int(curr_max), key=f"t_max_{bname}", help="Maximum pixel area for a granule. Discards unexpectedly large blobs.")
                    
                    if st.button("Re-run this Image", key=f"btn_rerun_{bname}"):
                        mp = {"block_size": bs_new, "c_value": cv_new, "bg_blur": bb_new, "min_area": min_new, "max_area": max_new}
                        st.session_state.per_image_overrides[bname] = mp
                        s1_data = st.session_state.s1_results[bname]
                        
                        new_res, new_img = run_stage2_image(
                            s1_data["path"], color_mode, st.session_state.custom_bgr,
                            s1_data["res"], s2_params_auto=False, manual_params=mp, apply_clahe=apply_clahe
                        )
                        st.session_state.s2_results[bname] = {"res": new_res, "img": new_img}
                        st.rerun()

# ---- TAB 3: RESULTS ----
with tab3:
    st.markdown("### Overall Statistics and Export")
    
    if st.session_state.s1_results or st.session_state.s2_results:
        # Build DataFrames
        df_summary_rows = []
        df_granule_rows = []
        
        for name, s1_data in st.session_state.s1_results.items():
            row = {"Image": name, "S1_Count": s1_data["res"]["s1_count"]}
            if name in st.session_state.s2_results:
                s2_res = st.session_state.s2_results[name]["res"]
                row["Status"] = "Stage 2"
                row["S2_Count"] = s2_res["s2_count"]
                row.update(s2_res["stats"])
                
                for g in s2_res["gran_list"]:
                    g_row = {"Image": name}
                    g_row.update(g)
                    df_granule_rows.append(g_row)
            else:
                row["Status"] = "Stage 1 Only"
                row["S2_Count"] = None
                
            df_summary_rows.append(row)
            
        df_summary = pd.DataFrame(df_summary_rows)
        df_granules = pd.DataFrame(df_granule_rows)
        
        st.subheader("Image-wise Statistics")
        st.dataframe(df_summary, use_container_width=True)
        
        # COMBINED PLOTS
        if not df_summary.empty:
            st.markdown("#### Visual Statistics (Combined)")
            col_p1, col_p2 = st.columns(2)
            
            df_plot_counts = df_summary.copy()
            df_plot_counts['Display_Count'] = df_plot_counts['S2_Count'].fillna(df_plot_counts['S1_Count'])
            
            with col_p1:
                fig_counts = px.bar(df_plot_counts, x='Image', y='Display_Count', title='Granule Counts by Image', color='Status')
                st.plotly_chart(fig_counts, use_container_width=True)
            with col_p2:
                if not df_granules.empty:
                    fig_areas = px.box(df_granules, x='Image', y='area_px', title='Area Distribution per Image (Stage 2)', points="all")
                    st.plotly_chart(fig_areas, use_container_width=True)
        
        st.subheader("Granule-wise Details (Stage 2)")
        st.dataframe(df_granules, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Export Results")
        if st.button("Save Results to Output Directory", type="primary"):
            os.makedirs(out_dir, exist_ok=True)
            df_summary.to_csv(os.path.join(out_dir, "image_summary_stats.csv"), index=False)
            if not df_granules.empty:
                df_granules.to_csv(os.path.join(out_dir, "granule_details.csv"), index=False)
            
            s1_dir = os.path.join(out_dir, "stage1")
            os.makedirs(s1_dir, exist_ok=True)
            for name, s1_data in st.session_state.s1_results.items():
                cv2.imwrite(os.path.join(s1_dir, "s1_"+name), s1_data["img"])

            if st.session_state.s2_results:
                s2_marked = os.path.join(out_dir, "stage2", "marked")
                s2_sbs = os.path.join(out_dir, "stage2", "side_by_side")
                os.makedirs(s2_marked, exist_ok=True)
                os.makedirs(s2_sbs, exist_ok=True)
                
                prog_save = st.progress(0)
                tot = len(st.session_state.s2_results)
                for i, (name, s2_data) in enumerate(st.session_state.s2_results.items()):
                    orig = cv2.imread(st.session_state.s1_results[name]["path"])
                    marked = s2_data["img"]
                    
                    h1, w1 = orig.shape[:2]
                    h2, w2 = marked.shape[:2]
                    
                    if h1 != h2 or w1 != w2:
                        marked = cv2.resize(marked, (w1, h1))
                        
                    sbs = np.hstack((orig, marked)) # Side-by-side (Original | Marked)
                    
                    cv2.imwrite(os.path.join(s2_marked, "marked_"+name), marked)
                    cv2.imwrite(os.path.join(s2_sbs, "sbs_"+name), sbs)
                    prog_save.progress((i+1)/tot)
            
            # Save HTML interactive plots
            if not df_summary.empty:
                fig_counts.write_html(os.path.join(out_dir, "plot_counts.html"))
            if not df_granules.empty:
                fig_areas.write_html(os.path.join(out_dir, "plot_areas.html"))
                
            st.success(f"Results successfully saved to `{out_dir}`!")
            st.info("Check the directory for CSVs, HTML plots, and annotated images (Marked and Side-By-Side variants in the `stage2` subfolder).")
    else:
        st.info("Run Stage 1 and Stage 2 to see the statistics and export options.")
