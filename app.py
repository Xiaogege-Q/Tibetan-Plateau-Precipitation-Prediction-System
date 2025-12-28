import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter # 引入刻度格式化工具
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import geopandas as gpd
import os

# ==========================================
#  0. 全局设置：字体与配置
# ==========================================
# 尝试加载本地字体文件 (专门解决 Streamlit Cloud 中文乱码)
font_path = 'simhei.ttf'  # 确保这个文件名和你上传的一模一样

if os.path.exists(font_path):
    # 如果找到了字体文件，就注册它
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = ['SimHei'] # 设置为该字体名
else:
    # 本地没有文件时的备选 (Windows本地运行时依然可用)
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide", page_title="青藏高原降水预测系统")

# ==========================================
# 1. 定义高颜值气象配色方案
# ==========================================
def get_precip_cmap():
    colors = [
        '#FFFFFF', # 0: 白色
        '#A6F28F', # 小雨
        '#3DBA3D', # 中雨
        '#61B8FF', # 大雨
        '#0000E1', # 暴雨
        '#FA00FA', # 大暴雨
        '#800040'  # 特大暴雨
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list('precip_custom', colors)
    cmap.set_under('white') 
    return cmap

def get_bias_cmap():
    return plt.get_cmap('RdBu')

# ==========================================
#  2. 核心工具：SHP裁切与刻度格式化
# ==========================================
def mask_outside_polygon(grid_lon, grid_lat, shp_gdf):
    if shp_gdf is None:
        return np.zeros_like(grid_lon, dtype=bool)
    
    points = np.vstack((grid_lon.flatten(), grid_lat.flatten())).T
    mask_combined = np.zeros(points.shape[0], dtype=bool)
    
    for geom in shp_gdf.geometry:
        if geom.geom_type == 'Polygon':
            polys = [geom]
        elif geom.geom_type == 'MultiPolygon':
            polys = geom.geoms
        else:
            continue
            
        for poly in polys:
            exterior_coords = np.array(poly.exterior.coords)
            mpl_path = mpath.Path(exterior_coords)
            mask = mpl_path.contains_points(points)
            mask_combined |= mask
            
    return ~mask_combined.reshape(grid_lon.shape)

# 定义经纬度显示的格式函数
def format_lon(x, pos):
    return f"{int(x)}°E"

def format_lat(x, pos):
    return f"{int(x)}°N"

# ==========================================
# 3. 数据加载
# ==========================================
st.title("🌧️ 青藏高原降水时空融合预测系统")

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
            if gdf.crs and gdf.crs.to_string() != 'EPSG:4326':
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
st.sidebar.header("⚙️ 控制面板")
dates = sorted(df['日期'].unique())
selected_date = st.sidebar.select_slider("📅 选择日期", options=dates, value=dates[-1])
date_str = pd.to_datetime(selected_date).strftime('%Y-%m-%d')
st.sidebar.markdown(f"## 当前: **{date_str}**")

day_data = df[df['日期'] == selected_date]

if day_data.empty:
    st.warning("该日期无数据")
else:
    # 评估指标
    st.subheader(f"📈 {date_str} 评估指标")
    y_true, y_pred = day_data['真实降水'], day_data['预测降水']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{np.sqrt(((y_true - y_pred) ** 2).mean()):.2f} (mm)")
    c2.metric("MAE", f"{np.mean(np.abs(y_true - y_pred)):.2f} (mm)")
    c3.metric("R (相关系数)", f"{y_true.corr(y_pred):.3f}")
    c4.metric("Bias (偏差)", f"{np.mean(y_pred - y_true):.2f} (mm)")

    # ==========================================
    # 5. 绘图函数
    # ==========================================
    st.markdown("---")
    st.subheader("🌧️ 降水空间分布对比")

    def plot_final_map(data, col, title, is_bias=False):
        fig, ax = plt.subplots(figsize=(10, 9))
        
        # 1. 插值
        grid_x, grid_y = np.mgrid[67:105:300j, 25:40:300j]
        grid_z = griddata((data['经度'], data['纬度']), data[col], (grid_x, grid_y), method='linear')
        
        # 2. SHP 裁切
        if shp is not None:
            mask = mask_outside_polygon(grid_x, grid_y, shp)
            grid_z = np.ma.array(grid_z, mask=mask)
        
        # 3. 设置颜色参数
        if is_bias:
            cmap = get_bias_cmap()
            limit = np.nanmax(np.abs(grid_z))
            if np.isnan(limit) or limit < 0.1: limit = 1.0
            vmin, vmax = -limit, limit
            levels = np.linspace(vmin, vmax, 41) 
            c_label = '偏差 (mm)'
        else:
            cmap = get_precip_cmap()
            vmin = 0.0 
            max_val = np.nanmax(grid_z)
            if np.isnan(max_val) or max_val < 1: 
                vmax = 10.0
            else:
                vmax = max_val
            levels = np.linspace(vmin, vmax, 40)
            c_label = '降水量 (mm)'

        # 4. 绘图
        cf = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
        
        if shp is not None:
            shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1.2)
        
        # 5. 横向颜色条
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', fraction=0.05, pad=0.08, aspect=30)
        cbar.set_label(c_label, fontsize=12)
        
        # 6. 设置标题和坐标轴
        ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel('lon', fontsize=12)
        ax.set_ylabel('lat', fontsize=12)
        ax.set_xlim(67, 105)
        ax.set_ylim(25, 40)
        
        # 7. 应用经纬度格式化 (30°N, 90°E)
        ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
        
        return fig

    # 第一行：实测与预测对比
    col_l, col_r = st.columns(2)
    with col_l:
        st.pyplot(plot_final_map(day_data, '真实降水', f'{date_str} 实测降水', is_bias=False))

    with col_r:
        st.pyplot(plot_final_map(day_data, '预测降水', f'{date_str} 模型预测', is_bias=False))

    # 第二行：Bias 偏差图 (直出，不用点击)
    st.markdown("---")
    st.subheader("📈 预测偏差分布")
    
    # 准备偏差数据
    day_data = day_data.copy() # 避免警告
    day_data['Bias'] = day_data['预测降水'] - day_data['真实降水']
    
    # 使用居中布局显示 Bias 图
    c_left, c_mid, c_right = st.columns([1, 2, 1])
    with c_mid:
        st.write("注：蓝色表示预测偏多(湿)，红色表示预测偏少(干)")

        st.pyplot(plot_final_map(day_data, 'Bias', f'{date_str} 预测偏差', is_bias=True))
