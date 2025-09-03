# compare_json_paths.py
# 2025-09-02
# Description:
# 比較不同颱風路徑的相似度，使用 DTW, LCSS, Hausdorff, 和 Feature Vector 等多種演算法
# 從 CSV 和 JSON 檔案中讀取路徑
# version 4.4: 新增時間範圍篩選功能
# Usage: python compare_json_paths.py data\測試用\2025PODUL.csv --top_n 5 --sort_by lcss -o --plot --start_year 2010 --end_year 2020
# 綜合排名: --sort_by composite --weights 0.4 0.4 0.1 0.1

import csv
import json
import argparse
import numpy as np
import os
import multiprocessing
from functools import partial
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm

from src.plot_paths_on_map import plot_paths_on_map

# --- 相似度計算函式 ---

def haversine(lat1, lon1, lat2, lon2):
    """計算兩點之間的哈佛正弦距離（公里）"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371  # 地球半徑（公里）
    return c * r

def dtw_distance(path1, path2):
    """使用動態時間規整演算法計算兩個路徑的距離"""
    n, m = len(path1), len(path2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(path1[i - 1] - path2[j - 1])
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n, m]

def lcss_similarity(path1, path2, epsilon):
    """計算兩路徑的 LCSS 相似度分數"""
    n1, n2 = len(path1), len(path2)
    # 使用 NumPy 陣列以獲得更好的效能
    dp_table = np.zeros((n1 + 1, n2 + 1))
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if haversine(path1[i-1][0], path1[i-1][1], path2[j-1][0], path2[j-1][1]) < epsilon:
                dp_table[i][j] = 1 + dp_table[i-1][j-1]
            else:
                dp_table[i][j] = max(dp_table[i-1][j], dp_table[i][j-1])
    lcss_len = dp_table[n1, n2] 
    min_len = min(n1, n2)
    return lcss_len / min_len if min_len > 0 else 0

def hausdorff_distance(path1, path2):
    """計算兩路徑之間的豪斯多夫距離"""
    n1, n2 = len(path1), len(path2)
    if n1 == 0 or n2 == 0:
        return np.inf
    # 使用預先分配的 NumPy 陣列和迴圈，通常比 list comprehension 更快
    dist_matrix = np.empty((n1, n2))
    for i in range(n1):
        for j in range(n2):
            dist_matrix[i, j] = haversine(path1[i, 0], path1[i, 1], path2[j, 0], path2[j, 1])
    h1 = np.max(np.min(dist_matrix, axis=1))
    h2 = np.max(np.min(dist_matrix, axis=0))
    return max(h1, h2)

def get_path_length(path):
    """計算路徑總長度（公里）"""
    length = 0
    for i in range(len(path) - 1):
        length += haversine(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
    return length

def extract_features(path):
    """提取路徑的特徵向量：[起點緯度, 起點經度, 終點緯度, 終點經度, 路徑長度/100]"""
    start_point = path[0]
    end_point = path[-1]
    length = get_path_length(path)
    # 長度做一個簡單的縮放，使其與經緯度的數量級大致相同
    return np.array([start_point[0], start_point[1], end_point[0], end_point[1], length / 100.0])

def clip_path_by_proximity(path_to_clip, reference_path, threshold):
    """
    裁切一個路徑，只保留那些與參考路徑在一定距離內的點。
    返回一個新的、可能更短的路徑。
    """
    # 如果沒有設定閾值或閾值無效，直接返回原路徑
    if threshold is None or threshold <= 0:
        return path_to_clip

    # 使用 NumPy 的廣播 (broadcasting) 機制來向量化計算
    # 避免在迴圈中重複呼叫 haversine
    clipped_points = []
    ref_path_rad = np.radians(reference_path)
    r = 6371  # 地球半徑

    for p_clip in path_to_clip:
        p_clip_rad = np.radians(p_clip)
        
        dlon = ref_path_rad[:, 1] - p_clip_rad[1]
        dlat = ref_path_rad[:, 0] - p_clip_rad[0]
        
        a = np.sin(dlat / 2)**2 + np.cos(p_clip_rad[0]) * np.cos(ref_path_rad[:, 0]) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distances = r * c
        
        if np.min(distances) <= threshold:
            clipped_points.append(p_clip)
            
    return np.array(clipped_points) if clipped_points else np.array([])

# --- 平行計算的工作函式 ---

def process_one_typhoon(item, base_path, base_features, epsilon, clip_dist):
    """
    為單一颱風計算所有相似度指標。這是給 multiprocessing.Pool 使用的工作函式。
    """
    typhoon_id, other_path_original = item
    if other_path_original is None or len(other_path_original) < 2:
        return None

    # 新增：根據基準路徑範圍裁切歷史路徑
    other_path = clip_path_by_proximity(other_path_original, base_path, clip_dist)

    # 如果裁切後路徑點太少，則跳過
    if len(other_path) < 2:
        return None

    # 為了避免與自身比較 (使用原始路徑進行比較)
    if np.array_equal(base_path, other_path_original):
        return None

    # 注意：特徵向量和其他相似度計算都應基於裁切後的路徑
    other_features = extract_features(other_path)

    return {'id': typhoon_id,
            'dtw': dtw_distance(base_path, other_path),
            'lcss': lcss_similarity(base_path, other_path, epsilon),
            'hausdorff': hausdorff_distance(base_path, other_path),
            'feature': np.linalg.norm(base_features - other_features)}

# --- 資料讀取函式 ---

def read_typhoon_path(file_path):
    """從 CSV 檔案中讀取颱風路徑資料"""
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 {file_path}")
        return None
    if os.path.isdir(file_path):
        print(f"錯誤：提供的路徑是一個目錄，而非檔案：{file_path}")
        return None

    path = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                first_row = next(reader)
                f.seek(0)
                reader = csv.reader(f)
                has_header = any(c.isalpha() for c in first_row[0])
                if has_header:
                    next(reader)
                for row in reader:
                    try:
                        if len(row) >= 2:
                            lat, lon = float(row[-2]), float(row[-1])
                            path.append((lat, lon))
                    except (ValueError, IndexError):
                        continue
            except StopIteration:
                print(f"警告：檔案 {file_path} 是空的。")
                return None
    except PermissionError:
        print(f"錯誤：權限不足，無法讀取檔案 {file_path}。請檢查檔案權限或確認路徑是否為檔案而非目錄。")
        return None
    except Exception as e:
        print(f"讀取檔案 {file_path} 時發生未預期的錯誤: {e}")
        return None
    return np.array(path) if path else None

def read_json_paths(file_path):
    """從 JSON 檔案中讀取所有颱風路徑"""
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 {file_path}")
        return None
    if os.path.isdir(file_path):
        print(f"錯誤：提供的路徑是一個目錄，而非檔案：{file_path}")
        return None

    paths = {}
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            for typhoon_id, path_data in data.items():
                path = [(point['Lat'], point['Lon']) for point in path_data]
                paths[typhoon_id] = np.array(path)
    except json.JSONDecodeError:
        print(f"錯誤：無法解析 JSON 檔案 {file_path}")
        return None
    except PermissionError:
        print(f"錯誤：權限不足，無法讀取檔案 {file_path}。請檢查檔案權限或確認路徑是否為檔案而非目錄。")
        return None
    except Exception as e:
        print(f"讀取檔案 {file_path} 時發生未預期的錯誤: {e}")
        return None
    return paths

def read_cname_mapping(file_path):
    """從 CSV 檔案讀取颱風 ID 與中文名稱的對照表"""
    if not os.path.exists(file_path):
        print(f"警告：找不到颱風中文名稱對照表檔案 {file_path}，將只顯示颱風 ID。")
        return {}
    
    mapping = {}
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            try:
                next(reader)  # 跳過標頭
                for row in reader:
                    if len(row) >= 2:
                        typhoon_id, cname = row[0].strip(), row[1].strip()
                        if typhoon_id and cname:
                            mapping[typhoon_id] = cname
            except StopIteration:
                print(f"警告：中文名稱對照表檔案 {file_path} 是空的。")
    except Exception as e:
        print(f"讀取中文名稱對照表 {file_path} 時發生錯誤: {e}")
    
    return mapping

# --- 主程式 ---

def main(args):
    """主函式"""
    cname_mapping = read_cname_mapping(args.cname_db)
    if args.sort_by.lower() == 'composite' and not np.isclose(sum(args.weights), 1.0):
        print(f"警告：提供的權重總和 {sum(args.weights):.2f} 不為 1.0，這可能會影響綜合分數的可解釋性。")


    base_path = read_typhoon_path(args.base_csv)
    if base_path is None or len(base_path) < 2:
        print(f"錯誤：無法讀取基準颱風路徑 {args.base_csv} 或路徑點少於2。")
        return

    json_paths = read_json_paths(args.json_db)
    if json_paths is None: return

    # 新增：根據年份篩選颱風
    if args.start_year is not None or args.end_year is not None:
        original_count = len(json_paths)
        # 如果未指定，則設定一個寬鬆的邊界
        start_year = args.start_year if args.start_year is not None else 0
        end_year = args.end_year if args.end_year is not None else 9999

        filtered_paths = {}
        for typhoon_id, path in json_paths.items():
            try:
                # 颱風 ID 格式應為 'YYYYNAME'，前四位為年份
                year = int(typhoon_id[:4])
                if start_year <= year <= end_year:
                    filtered_paths[typhoon_id] = path
            except (ValueError, IndexError):
                # 忽略不符合 'YYYY...' 格式的 ID
                continue
        json_paths = filtered_paths
        print(f"\n已根據年份範圍 [{start_year if args.start_year is not None else '不限'} - {end_year if args.end_year is not None else '不限'}] 進行篩選，從 {original_count} 個颱風中選出 {len(json_paths)} 個進行比較。")

    base_features = extract_features(base_path)

    # --- 平行計算 ---
    # 準備要傳遞給工作函式的固定參數
    worker_func = partial(process_one_typhoon,
                          base_path=base_path,
                          base_features=base_features,
                          epsilon=args.epsilon,
                          clip_dist=args.clip_dist)

    items_to_process = json_paths.items()

    # 使用 multiprocessing.Pool 來平行處理
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        # 使用 imap 來取得一個結果迭代器，並用 tqdm 顯示進度
        results_iterator = pool.imap(worker_func, items_to_process)
        results = list(tqdm(results_iterator, total=len(items_to_process), desc="計算相似度", unit="個颱風"))

    # 過濾掉無效的結果 (例如路徑點太少或與自身比較的結果)
    results = [r for r in results if r is not None]

    if not results:
        print("在 JSON 檔案中沒有找到可比較的颱風路徑。")
        return

    # 如果選擇綜合分數排序，則進行計算
    if args.sort_by.lower() == 'composite':
        # 1. 提取所有值以找到 min/max 進行正規化
        all_dtw = np.array([r['dtw'] for r in results])
        all_lcss = np.array([r['lcss'] for r in results])
        all_hausdorff = np.array([r['hausdorff'] for r in results])
        all_feature = np.array([r['feature'] for r in results])

        # 處理除以零的情況（如果所有值都相同）
        range_dtw = all_dtw.max() - all_dtw.min() if all_dtw.max() > all_dtw.min() else 1.0
        range_lcss = all_lcss.max() - all_lcss.min() if all_lcss.max() > all_lcss.min() else 1.0
        range_hausdorff = all_hausdorff.max() - all_hausdorff.min() if all_hausdorff.max() > all_hausdorff.min() else 1.0
        range_feature = all_feature.max() - all_feature.min() if all_feature.max() > all_feature.min() else 1.0

        w_dtw, w_lcss, w_hausdorff, w_feature = args.weights

        # 2. 計算每個結果的綜合分數
        for r in results:
            # 正規化分數至 [0, 1] 區間，1 代表最好
            norm_dtw = (all_dtw.max() - r['dtw']) / range_dtw
            norm_hausdorff = (all_hausdorff.max() - r['hausdorff']) / range_hausdorff
            norm_feature = (all_feature.max() - r['feature']) / range_feature
            norm_lcss = (r['lcss'] - all_lcss.min()) / range_lcss

            # 3. 計算加權總和
            composite_score = (w_dtw * norm_dtw +
                               w_lcss * norm_lcss +
                               w_hausdorff * norm_hausdorff +
                               w_feature * norm_feature)
            r['composite'] = composite_score

    # 根據使用者選擇的排序方式進行排序
    sort_key = args.sort_by.lower()
    # LCSS 和綜合分數是越大越好
    reverse_order = (sort_key in ['lcss', 'composite'])
    
    # 對於 feature 和 hausdorff，值越小越相似
    sorted_results = sorted(results, key=lambda item: item[sort_key], reverse=reverse_order)

    print(f"\n與 {os.path.basename(args.base_csv)} 路徑最相似的前 {args.top_n} 個颱風 (排序依據: {sort_key.upper()})：")
    header_parts = [f"{'颱風名稱(ID)':<25}", f"{'DTW':<10}", f"{'LCSS':<10}", f"{'Hausdorff':<12}", f"{'Feature':<10}"]
    if sort_key == 'composite':
        header_parts.append(f"{'綜合分數':<10}")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))
    
    for res in sorted_results[:args.top_n]:
        typhoon_id = res['id']
        cname = cname_mapping.get(typhoon_id)
        display_name = f"{cname}({typhoon_id})" if cname else typhoon_id

        dtw_str = f"{res['dtw']:.2f}"
        lcss_str = f"{res['lcss']:.2f}"
        hausdorff_str = f"{res['hausdorff']:.2f}"
        feature_str = f"{res['feature']:.2f}"
        
        print_parts = [f"{display_name:<25}", f"{dtw_str:<10}", f"{lcss_str:<10}", f"{hausdorff_str:<12}", f"{feature_str:<10}"]
        if sort_key == 'composite':
            composite_str = f"{res['composite']:.3f}"
            print_parts.append(f"{composite_str:<10}")
        print(" | ".join(print_parts))
    
    base_filename = os.path.splitext(os.path.basename(args.base_csv))[0]
    output_dir = os.path.join('results', base_filename)

    # 如果需要，繪製地圖
    if args.plot:
        os.makedirs(output_dir, exist_ok=True)
        image_filename = f"{args.sort_by.lower()}.png"
        image_path = os.path.join(output_dir, image_filename)
        
        # 取得前 N 個颱風的路徑資料，並加上中文名稱
        top_n_paths = {}
        for res in sorted_results[:args.top_n]:
            typhoon_id = res['id']
            cname = cname_mapping.get(typhoon_id)
            display_name = f"{cname}({typhoon_id})" if cname else typhoon_id
            top_n_paths[display_name] = json_paths[res['id']]
        
        # 準備給繪圖函式的分析參數
        year_range_str = None
        if args.start_year is not None or args.end_year is not None:
            start = args.start_year if args.start_year is not None else '不限'
            end = args.end_year if args.end_year is not None else '不限'
            year_range_str = f"{start}-{end}"
            
        analysis_params = {
            'epsilon': args.epsilon,
            'clip_dist': args.clip_dist,
            'year_range': year_range_str
        }
        plot_paths_on_map(base_path, base_filename, top_n_paths, image_path, args.top_n, args.sort_by, analysis_params)

    # 如果需要，儲存 CSV
    if args.output:
        try:
            os.makedirs(output_dir, exist_ok=True)

            output_filename = f"{args.sort_by.lower()}.csv"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                csv_header = ['序位', '颱風ID', '颱風中文名', 'DTW距離', 'LCSS相似度', 'Hausdorff距離', 'Feature距離']
                if args.sort_by.lower() == 'composite':
                    csv_header.append('綜合分數')
                writer.writerow(csv_header)
                for i, res in enumerate(sorted_results[:args.top_n]):
                    typhoon_id = res['id']
                    cname = cname_mapping.get(typhoon_id, '')
                    csv_row = [i + 1, typhoon_id, cname, f"{res['dtw']:.2f}", f"{res['lcss']:.2f}", f"{res['hausdorff']:.2f}", f"{res['feature']:.2f}"]
                    if args.sort_by.lower() == 'composite':
                        csv_row.append(f"{res['composite']:.4f}")
                    writer.writerow(csv_row)
            print(f"\n結果已成功儲存至 {output_path}")
        except IOError as e:
            print(f"\n錯誤：無法寫入檔案 {output_path}。原因: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用多種演算法，比較一個基準颱風路徑與 JSON 資料庫中所有路徑的相似度。")
    parser.add_argument("base_csv", help="基準颱風路徑的 CSV 檔案路徑。")
    parser.add_argument("--json_db", default="data/mit_typhoon_paths.json", help="包含多個颱風路徑的 JSON 資料庫檔案路徑。")
    parser.add_argument("--cname_db", default="data/typhoon_cname.csv", help="颱風中文名稱對照表的 CSV 檔案路徑。")
    parser.add_argument("--top_n", type=int, default=10, help="顯示最相似的前 N 個結果。")
    parser.add_argument("--epsilon", type=float, default=100.0, help="LCSS 演算法的距離閾值(公里)。")
    parser.add_argument("--sort_by", type=str, default="dtw", choices=['dtw', 'lcss', 'hausdorff', 'feature', 'composite'], help="結果的排序依據。'composite'為綜合分數。")
    parser.add_argument("--weights", type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25], metavar=('W_DTW', 'W_LCSS', 'W_HAUS', 'W_FEAT'), help="計算綜合分數時，四種演算法的權重 (DTW, LCSS, Hausdorff, Feature)，總和應為1。")
    parser.add_argument("--clip_dist", type=float, default=None, help="在計算相似度前，將歷史路徑裁切至基準路徑周圍 N 公里範圍內。")
    parser.add_argument("-o", "--output", action="store_true", help="將結果輸出到 CSV 檔案。路徑會自動生成為 'results\\<基準檔名>\\<排序演算法>.csv'。")
    parser.add_argument("--plot", action="store_true", help="將基準路徑與最相似的前 N 個路徑繪製在地圖上並儲存為 PNG 圖片。")
    parser.add_argument("--start_year", type=int, default=None, help="篩選歷史颱風的起始年份 (包含)。")
    parser.add_argument("--end_year", type=int, default=None, help="篩選歷史颱風的結束年份 (包含)。")
    
    args = parser.parse_args()
    main(args)