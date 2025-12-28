import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.font_manager as fm # 必须导入这个
import geopandas as gpd
import os

# ==========================================
# 0. 全局设置：字体与配置 (关键修复：云端中文支持)
# ==========================================
# 尝试加载本地字体文件 (解决 Streamlit Cloud 中文乱码)
font_path = 'simhei.ttf'  
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = ['SimHei']
else:
    # 本地 Windows 备选
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei']

plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide", page_title="青藏高原降水预测系统")

# ==========================================
# 1. 定义配色方案
# ==========================================
def get_precip_cmap():
    colors = ['#FFFFFF', '#A6F28F', '#3DBA3D', '#61B8FF', '#0000E1', '#FA00FA', '#800040']
    cmap = mcolors.LinearSegmentedColormap.from_list('precip_custom', colors)
    cmap.set_under('white') 
    return cmap

def get_bias_cmap():
    return plt.get_cmap('RdBu')

# ==========================================
# 2. 核心工具：SHP裁切
# ==========================================
def mask_outside_polygon(grid_lon, grid_lat, shp_gdf):
    if shp_gdf is None: return np.zeros_like(grid_lon, dtype=bool)
    
    points = np.vstack((grid_lon.flatten(), grid_lat.flatten())).T
    mask_combined = np.zeros(points.shape[0], dtype=bool)
    
    for geom in shp_gdf.geometry:
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            polys = [geom] if geom.geom_type == 'Polygon' else geom.geoms
            for poly in polys:
                mpl_path = mpath.Path(np.array(poly.exterior.coords))
                mask_combined |= mpl_path.contains_points(points)
                
    return ~mask_combined.reshape(grid_lon.shape)

def format_lon(x, pos): return f"{int(x)}°E"
def format_lat(x, pos): return f"{int(x)}°N"

# ==========================================
# 3. 数据加载 (即使无 PRJ 也能运行)
# ==========================================
st.title("🌧️ 青藏高原降水时空融合预测系统")
st.markdown("**说明：** 本系统利用过去7天数据，预测**未来3天的累计降水量**。")

@st.cache_data
def load_data():
    if os.path.exists('website_data.csv'):
        return pd.read_csv('website_data.csv', parse_dates=['日期'])
    return None

@st.cache_data
def load_shapefile():
    shp_path = '青藏高原.shp'
    if os.path.exists(shp_path):
        try:
            gdf = gpd.read_file(shp_path)
            # 强制修正坐标系 (即使没有 prj 文件)
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            elif gdf.crs.to_string() != 'EPSG:4326':
                gdf = gdf.to_crs(epsg=4326)
            return gdf
        except:
            return None
    return None

df = load_data()
shp = load_shapefile()

if df is None:
    st.error("🚨 错误：未找到 `website_data.csv`。")
    st.stop()

# ==========================================
# 4. 侧边栏
# ==========================================
st.sidebar.header("🕹️ 控制面板")
dates = sorted(df['日期'].unique())
selected_date = st.sidebar.select_slider("📅 选择预报日期", options=dates, value=dates[-1])
date_str = pd.to_datetime(selected_date).strftime('%Y-%m-%d')

st.sidebar.info(f"""
**当前展示数据：**
截止至 **{date_str}** 的
未来三天**累计**降水量
""")

day_data = df[df['日期'] == selected_date]

if day_data.empty:
    st.warning("该日期无数据")
else:
    # 评估指标 (修改单位为 mm/3days)
    st.subheader(f"📈 {date_str} 预测评估指标 (三天累计)")
    y_true, y_pred = day_data['真实降水'], day_data['预测降水']
    c1, c2, c3, c4 = st.columns(4)
    
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    mae = np.mean(np.abs(y_true - y_pred))
    corr = y_true.corr(y_pred)
    bias = np.mean(y_pred - y_true)

    c1.metric("RMSE", f"{rmse:.2f}", help="mm/3days")
    c2.metric("MAE", f"{mae:.2f}", help="mm/3days")
    c3.metric("R (相关系数)", f"{corr:.3f}")
    c4.metric("Bias (偏差)", f"{bias:.2f}", help="mm/3days")

    # ==========================================
    # 5. 绘图函数
    # ==========================================
    st.markdown("---")
    st.subheader("🌍 未来三天累计降水空间分布")

    def plot_final_map(data, col, title, is_bias=False):
        fig, ax = plt.subplots(figsize=(10, 9))
        grid_x, grid_y = np.mgrid[67:105:300j, 25:40:300j]
        grid_z = griddata((data['经度'], data['纬度']), data[col], (grid_x, grid_y), method='linear')
        
        if shp is not None:
            mask = mask_outside_polygon(grid_x, grid_y, shp)
            grid_z = np.ma.array(grid_z, mask=mask)
        
        if is_bias:
            cmap = get_bias_cmap()
            limit = np.nanmax(np.abs(grid_z)); limit = 1.0 if np.isnan(limit) or limit < 0.1 else limit
            vmin, vmax = -limit, limit
            levels = np.linspace(vmin, vmax, 41) 
            c_label = '偏差 (Bias) [mm/3days]'
        else:
            cmap = get_precip_cmap()
            vmin = 0.0 
            max_val = np.nanmax(grid_z); vmax = 10.0 if np.isnan(max_val) or max_val < 1 else max_val
            levels = np.linspace(vmin, vmax, 40)
            c_label = '累计降水量 (Total Precip) [mm]'

        cf = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
        if shp is not None: shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1.2)
        
        # 调整了 pad，让色条更美观
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', fraction=0.05, pad=0.12, aspect=30)
        cbar.set_label(c_label, fontsize=12)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel('Lon', fontsize=12)
        ax.set_ylabel('Lat', fontsize=12)
        ax.set_xlim(67, 105); ax.set_ylim(25, 40)
        ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
        return fig

    col_l, col_r = st.columns(2)
    with col_l:
        st.pyplot(plot_final_map(day_data, '真实降水', f'{date_str} 实测三天累计', is_bias=False))

    with col_r:
        st.pyplot(plot_final_map(day_data, '预测降水', f'{date_str} 预测三天累计', is_bias=False))

    st.markdown("---")
    st.subheader("📉 预测偏差分布")
    day_data = day_data.copy()
    day_data['Bias'] = day_data['预测降水'] - day_data['真实降水']
    
    c_left, c_mid, c_right = st.columns([1, 2, 1])
    with c_mid:
        st.pyplot(plot_final_map(day_data, 'Bias', f'{date_str} 预测偏差 (预测-实测)', is_bias=True))
