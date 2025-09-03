# 請將此函式貼至 d:\home\2_MIT\11_SimilarTyphRoutes\03_ai_similar_route\src\plot_paths_on_map.py

import numpy as np
try:
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def plot_paths_on_map(base_path, base_name, similar_paths, output_image_path, top_n, sort_by, analysis_params=None):
    """將基準路徑與最相似的路徑繪製在地圖上"""
    if not MATPLOTLIB_AVAILABLE:
        print("\n警告：無法繪製地圖。請安裝 matplotlib 和 cartopy 函式庫：")
        print("pip install matplotlib cartopy")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # 繪製基準路徑
    ax.plot(base_path[:, 1], base_path[:, 0], color='red', linewidth=2.5, marker='o', markersize=3, transform=ccrs.Geodetic(), label=f'基準: {base_name}')

    # 繪製相似路徑
    colors = plt.cm.viridis(np.linspace(0, 1, len(similar_paths)))
    for i, (typhoon_id, path) in enumerate(similar_paths.items()):
        ax.plot(path[:, 1], path[:, 0], color=colors[i], linewidth=1.5, marker='.', markersize=2, transform=ccrs.Geodetic(), label=f'#{i+1}: {typhoon_id}')

    # 自動設定地圖範圍
    all_paths = [base_path] + list(similar_paths.values())
    min_lon = min(p[:, 1].min() for p in all_paths) - 5
    max_lon = max(p[:, 1].max() for p in all_paths) + 5
    min_lat = min(p[:, 0].min() for p in all_paths) - 5
    max_lat = max(p[:, 0].max() for p in all_paths) + 5
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # --- 設定圖例和標題 ---
    ax.legend(loc='best')
    
    # 組合主標題和次要標題資訊
    main_title_line1 = f'與 {base_name} 最相似的前 {top_n} 個颱風路徑 (排序依據: {sort_by.upper()})'
    
    subtitle_parts = []
    if analysis_params:
        # 只有當排序依據是 LCSS 時才顯示 Epsilon
        if sort_by.lower() == 'lcss' and analysis_params.get('epsilon') is not None:
            subtitle_parts.append(f"LCSS ε={analysis_params['epsilon']:.1f}")
        if analysis_params.get('clip_dist') is not None:
            subtitle_parts.append(f"裁切範圍={analysis_params['clip_dist']:.1f} km")
        if analysis_params.get('year_range') is not None:
            subtitle_parts.append(f"年份: {analysis_params['year_range']}")

    full_title = main_title_line1
    if subtitle_parts:
        # 將分析參數作為第二行，並用換行符 \n 分隔
        main_title_line2 = "分析參數: " + " | ".join(subtitle_parts)
        full_title = f"{main_title_line1}\n{main_title_line2}"
    
    # 使用 ax.set_title 設置一個統一的多行標題
    ax.set_title(full_title, pad=15, fontsize=16)

    try:
        # 使用單一的 tight_layout 自動調整所有元素，包含多行標題
        plt.tight_layout()
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\n地圖已成功儲存至 {output_image_path}")
    except Exception as e:
        print(f"\n錯誤：儲存地圖時發生問題: {e}")

