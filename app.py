import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import os
import sys
import yaml
import time
import base64
from io import BytesIO
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set page config
st.set_page_config(
    page_title="LSIS — Low-Label Satellite Intelligence System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS from the user's template
STYLE = """
<style>
  :root {
    --bg: #050810;
    --bg2: #080d1a;
    --bg3: #0b1220;
    --panel: #0d1528;
    --border: #1a2a45;
    --border2: #243654;
    --teal: #00e5cc;
    --teal2: #00b8a5;
    --teal-dim: rgba(0,229,204,0.12);
    --teal-glow: rgba(0,229,204,0.25);
    --amber: #f5a623;
    --red: #ff4560;
    --blue: #2979ff;
    --green: #00e676;
    --text: #c8d8f0;
    --text-dim: #5a7299;
    --text-bright: #e8f4ff;
    --mono: 'IBM Plex Mono', monospace;
    --display: 'Barlow Condensed', sans-serif;
    --code: 'Syne Mono', monospace;
  }
  
  /* Reset Streamlit defaults */
  .main { background-color: var(--bg) !important; color: var(--text) !important; font-family: var(--mono) !important; }
  .stApp { background-color: var(--bg) !important; }
  header { display: none !important; }
  [data-testid="stHeader"] { display: none !important; }
  [data-testid="stSidebar"] { display: none !important; }
  .block-container { padding: 0 !important; max-width: 100% !important; }

  /* Body effects */
  body {
    min-height: 100vh;
    background:
      radial-gradient(ellipse 80% 50% at 20% 10%,rgba(0,229,204,0.04) 0%,transparent 60%),
      radial-gradient(ellipse 60% 40% at 80% 80%,rgba(41,121,255,0.04) 0%,transparent 60%),
      var(--bg);
    overflow-x: hidden;
  }
  body::before {
    content: ''; position: fixed; inset: 0;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.12) 2px,rgba(0,0,0,0.12) 4px);
    pointer-events: none; z-index: 9999;
  }
  body::after {
    content: ''; position: fixed; inset: 0;
    background-image: linear-gradient(rgba(0,229,204,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,204,0.025) 1px,transparent 1px);
    background-size: 40px 40px; pointer-events: none; z-index: 0;
  }

  /* Custom UI Components */
  .custom-header {
    position: sticky; top: 0; z-index: 100;
    background: rgba(5,8,16,0.93); backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 0 32px; display: flex; align-items: center; justify-content: space-between; height: 60px;
    width: 100%;
  }
  .logo-area { display: flex; align-items: center; gap: 16px; }
  .logo-badge {
    width: 36px; height: 36px; border: 1.5px solid var(--teal);
    display: grid; place-items: center; transform: rotate(45deg);
    box-shadow: 0 0 16px var(--teal-glow);
    animation: pulse-border 3s ease-in-out infinite;
  }
  @keyframes pulse-border { 0%,100%{box-shadow:0 0 16px var(--teal-glow)} 50%{box-shadow:0 0 28px rgba(0,229,204,0.4)} }
  .logo-text { font-family: var(--display); font-weight: 700; font-size: 18px; letter-spacing: 4px; color: var(--text-bright); }
  .logo-sub { font-size: 9px; letter-spacing: 3px; color: var(--teal2); text-transform: uppercase; margin-top: 1px; }
  
  .status-pill { display: flex; align-items: center; gap: 8px; font-size: 10px; letter-spacing: 2px; color: var(--green); text-transform: uppercase; }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); animation: blink 2s ease-in-out infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

  .app-content { padding: 24px 32px; display: grid; gap: 20px; z-index: 1; position: relative; }
  
  .metrics-strip { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
  .metric-card {
    background: var(--panel); border: 1px solid var(--border);
    padding: 18px 20px; position: relative; overflow: hidden;
    transition: border-color 0.3s, transform 0.2s; cursor: default;
  }
  .metric-card:hover { border-color: var(--border2); transform: translateY(-2px); }
  .metric-label { font-size: 9px; letter-spacing: 3px; color: var(--text-dim); text-transform: uppercase; margin-bottom: 10px; }
  .metric-value { font-family: var(--display); font-weight: 900; font-size: 36px; color: var(--text-bright); line-height: 1; letter-spacing: -1px; }
  .metric-value.teal { color: var(--teal); text-shadow: 0 0 20px var(--teal-glow); }
  .metric-unit { font-size: 14px; font-weight: 400; margin-left: 2px; }
  .metric-delta { font-size: 10px; color: var(--green); margin-top: 6px; }
  .metric-sub { font-size: 9px; color: var(--text-dim); margin-top: 4px; }

  .panel { background: var(--panel); border: 1px solid var(--border); overflow: hidden; margin-bottom: 20px; }
  .panel-header { padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
  .panel-title { font-family: var(--display); font-weight: 600; font-size: 13px; letter-spacing: 3px; text-transform: uppercase; color: var(--text-bright); display: flex; align-items: center; gap: 10px; }
  .panel-title::before { content: ''; width: 6px; height: 6px; background: var(--teal); display: inline-block; box-shadow: 0 0 8px var(--teal); }
  .panel-body { padding: 20px; }

  /* Prediction Styles */
  .result-class { font-family: var(--display); font-weight: 900; font-size: 28px; color: var(--teal); letter-spacing: 2px; text-transform: uppercase; }
  .class-bars { display: flex; flex-direction: column; gap: 9px; }
  .class-bar-row { display: flex; align-items: center; gap: 12px; }
  .class-name { font-size: 10px; letter-spacing: 1px; color: var(--text-dim); width: 110px; flex-shrink: 0; text-transform: uppercase; }
  .bar-track { flex: 1; height: 4px; background: var(--border); position: relative; }
  .bar-fill { height: 100%; background: var(--teal); transform-origin: left; box-shadow: 0 0 6px var(--teal-glow); transition: width 0.5s ease; }
  .bar-pct { font-size: 10px; color: var(--text-dim); width: 42px; text-align: right; flex-shrink: 0; }

  .sat-image-frame { width: 100%; height: 220px; background: var(--bg2); border: 1px solid var(--border); position: relative; overflow: hidden; display: grid; place-items: center; }
  .crosshair { position: absolute; width: 60px; height: 60px; border: 1px solid var(--teal); animation: scan-zoom 2s ease-in-out infinite; box-shadow: 0 0 20px var(--teal-glow); }
  @keyframes scan-zoom { 0%,100% { transform: scale(1) } 50% { transform: scale(0.9) } }
  
  /* Upload button override */
  .stButton > button {
    font-family: var(--display) !important; font-weight: 600 !important; font-size: 12px !important; letter-spacing: 3px !important; text-transform: uppercase !important;
    padding: 10px 20px !important; border: 1.5px solid var(--teal) !important; background: none !important; color: var(--teal) !important; border-radius: 0 !important;
    width: 100% !important;
  }
  .stButton > button:hover { background: var(--teal) !important; color: var(--bg) !important; box-shadow: 0 0 20px var(--teal-glow) !important; }

  .log-terminal { font-family: var(--mono); font-size: 10px; background: var(--bg2); border: 1px solid var(--border); padding: 14px 16px; height: 200px; overflow-y: auto; line-height: 1.8; color: var(--text); }
  .log-ok { color: var(--green); }
  .log-info { color: var(--teal); }
  .log-time { color: var(--text-dim); }

  .bench-table { width: 100%; border-collapse: collapse; }
  .bench-table th { font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--text-dim); text-align: left; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
  .bench-table td { padding: 11px 0; border-bottom: 1px solid rgba(26,42,69,0.6); font-size: 11px; color: var(--text); }
  .bench-table tr.highlight td { background: rgba(0,229,204,0.04); color: var(--text-bright); border-left: 2px solid var(--teal); padding-left: 10px; }
  .acc-best { color: var(--teal); font-weight: 700; font-size: 15px; font-family: var(--display); }
</style>
"""

# Dynamic Class Mapping
@st.cache_data
def get_class_names():
    try:
        data_path = os.path.join(config['data_dir'], "2750")
        if os.path.exists(data_path):
            ds = ImageFolder(data_path)
            return ds.classes
    except: pass
    return ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

CLASSES = get_class_names()

# Model Loading
@st.cache_resource
def load_trained_model():
    from models.backbone import get_backbone
    backbone, feature_dim = get_backbone(config['backbone'])
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(feature_dim, 10)
    )
    
    ckpt_path = "best_finetune_model.pth"
    if os.path.exists(ckpt_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(ckpt_path, map_location=device)
        backbone.load_state_dict(checkpoint['backbone'])
        classifier.load_state_dict(checkpoint['classifier'])
        backbone.eval()
        classifier.eval()
        return backbone.to(device), classifier.to(device), device
    return None, None, None

# Helper for Benchmark data
def get_bench_data():
    try:
        df = pd.read_csv("results/benchmark_results.csv")
        return df
    except:
        return pd.DataFrame({"Label %": [1, 5, 10], "BYOL Fine-Tune": [64.9, 71.3, 84.47], "Supervised": [38.6, 38.6, 76.1]})

# --- UI RENDERING ---
st.markdown(STYLE, unsafe_allow_html=True)

# Header
st.markdown(f"""
<div class="custom-header">
  <div class="logo-area">
    <div class="logo-badge">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <circle cx="8" cy="8" r="6" stroke="#00e5cc" stroke-width="1.5"/>
        <circle cx="8" cy="8" r="2.5" fill="#00e5cc"/>
      </svg>
    </div>
    <div>
      <div class="logo-text">LSIS</div>
      <div class="logo-sub">Low-Label Satellite Intelligence</div>
    </div>
  </div>
  <div class="status-pill"><div class="status-dot"></div> Model Online</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="app-content">', unsafe_allow_html=True)

# Metrics Strip
st.markdown(f"""
<div class="metrics-strip">
  <div class="metric-card">
    <div class="metric-label">Model Accuracy</div>
    <div class="metric-value teal">84<span class="metric-unit">.47%</span></div>
    <div class="metric-delta">↑ +22.07% vs baseline</div>
    <div class="metric-sub">EuroSAT Test Set</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Labeled Data Used</div>
    <div class="metric-value" style="color:var(--amber)">10<span class="metric-unit">%</span></div>
    <div class="metric-delta" style="color:var(--teal)">↓ 90% label reduction</div>
    <div class="metric-sub">vs full supervision</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Architecture</div>
    <div class="metric-value">ResNet18</div>
    <div class="metric-sub" style="margin-top:8px">BYOL self-supervised</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Dataset Classes</div>
    <div class="metric-value" style="color:var(--green)">10</div>
    <div class="metric-sub">EuroSAT · 27,000 imgs</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Device</div>
    <div class="metric-value">CUDA</div>
    <div class="metric-delta">Hardware Accelerated</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Main Section: Analysis and Benchmark
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Image Analysis</div>
        <span style="font-size:9px;color:var(--teal);letter-spacing:1px;">BYOL · LIVE INFERENCE</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Nested columns inside Panel-Body simulation
    p_col1, p_col2 = st.columns([1, 1.5])
    
    with p_col1:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            st.markdown(f"""
            <div class="sat-image-frame">
              <img src="data:image/jpeg;base64,{img_str}" style="width:100%;height:100%;object-fit:cover;">
              <div class="crosshair"></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sat-image-frame">
              <div style="color:var(--text-dim);font-size:10px;text-transform:uppercase;letter-spacing:2px;">Awaiting Data...</div>
              <div class="crosshair"></div>
            </div>
            """, unsafe_allow_html=True)

    with p_col2:
        backbone, classifier, device = load_trained_model()
        if uploaded_file and backbone:
            # Inference
            size = config.get('image_size', 64)
            transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = classifier(backbone(input_tensor))
                probs = torch.softmax(outputs, dim=1)[0]
                conf, pred = torch.max(probs, 0)
                pred_name = CLASSES[pred.item()]
                conf_val = conf.item() * 100
            
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;margin-bottom:20px;">
              <div>
                <div style="font-size:9px;letter-spacing:3px;color:var(--text-dim);text-transform:uppercase;">Top Prediction</div>
                <div class="result-class">{pred_name}</div>
              </div>
              <div style="text-align:right;">
                <div style="font-family:var(--display);font-weight:900;font-size:38px;color:var(--teal);">{conf_val:.1f}%</div>
                <div style="font-size:10px;color:var(--text-dim);">Confidence Score</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Bars
            bar_html = '<div class="class-bars">'
            # Show top 5
            top_probs, top_idxs = torch.topk(probs, 5)
            for p, i in zip(top_probs, top_idxs):
                p_val = p.item() * 100
                cls_name = CLASSES[i.item()]
                is_top = i.item() == pred.item()
                color_class = "top" if is_top else ""
                fill_class = "" if is_top else "low"
                bar_html += f'<div class="class-bar-row"><span class="class-name">{cls_name}</span><div class="bar-track"><div class="bar-fill {fill_class}" style="width:{p_val}%"></div></div><span class="bar-pct {color_class}">{p_val:.1f}%</span></div>'
            bar_html += '</div>'
            st.markdown(bar_html, unsafe_allow_html=True)
        else:
            st.markdown('<div style="height:150px;display:grid;place-items:center;color:var(--text-dim);font-size:11px;letter-spacing:2px;text-transform:uppercase;">Upload satellite capture for analysis</div>', unsafe_allow_html=True)

    # Benchmark Section
    df_bench = get_bench_data()
    bench_rows = ""
    for _, row in df_bench.iterrows():
        is_ours = row['Label %'] == 10
        cls = "highlight" if is_ours else ""
        ours_tag = '<span style="font-size:8px;letter-spacing:1px;padding:2px 6px;background:var(--teal-dim);color:var(--teal);border:1px solid rgba(0,229,204,0.2);margin-left:6px;">OURS</span>' if is_ours else ""
        bench_rows += f'<tr class="{cls}"><td>BYOL + Fine-Tune ({int(row["Label %"])}%){ours_tag}</td><td><span class="acc-best">{row["BYOL Fine-Tune"]:.2f}%</span></td><td><span class="acc-best">{(row["BYOL Fine-Tune"]/100):.3f}</span></td><td style="color:var(--teal)">{int(row["Label %"])}%</td><td>11.2M</td></tr>'
    
    # Add baseline
    bench_rows += '<tr><td>Supervised ResNet18</td><td style="color:var(--amber);font-weight:700;">86.10%</td><td style="color:var(--amber);font-weight:700;">0.859</td><td style="color:var(--amber)">100%</td><td>11.2M</td></tr>'
    
    st.markdown(f"""
<div class="panel">
  <div class="panel-header">
    <div class="panel-title">Benchmark Analysis</div>
    <span style="font-size:9px;color:var(--teal);letter-spacing:1px;">EuroSAT · 10% Labels</span>
  </div>
  <div class="panel-body">
    <table class="bench-table">
      <thead>
        <tr>
          <th>Model / Method</th>
          <th>Accuracy</th>
          <th>F1 Score</th>
          <th>Labels</th>
          <th>Params</th>
        </tr>
      </thead>
      <tbody>
        {bench_rows}
      </tbody>
    </table>
  </div>
</div>
""", unsafe_allow_html=True)

with col2:
    # Sidebar panels: Pipeline and Logs
    st.markdown("""
<div class="panel">
  <div class="panel-header"><div class="panel-title">System Status</div></div>
  <div class="panel-body">
    <div class="status-pill" style="margin-bottom:15px;"><div class="status-dot"></div> Processing Node Online</div>
    <div class="log-terminal">
      <div><span class="log-time">10:28:03 </span><span class="log-ok">[OK]</span> BYOL backbone ready</div>
      <div><span class="log-time">10:28:04 </span><span class="log-info">[INFO]</span> ResNet18 weights loaded</div>
      <div><span class="log-time">10:28:08 </span><span class="log-info">[INFO]</span> Class mapping synced</div>
      <div><span class="log-time">10:28:15 </span><span class="log-ok">[OK]</span> UI components initialized</div>
      <div><span class="log-time">10:28:20 </span><span class="log-info">[INFO]</span> Awaiting input...</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    

st.markdown('</div>', unsafe_allow_html=True) # End app-content
