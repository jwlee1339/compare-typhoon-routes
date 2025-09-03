# compare_paths.py
# 2025-09-02
# Description:
# 比較不同颱風路徑的相似度，使用動態時間規整 (DTW) 演算法
# 支援多種 CSV 格式的讀取
# version 1: 從檔案讀取路徑，計算並顯示相似度

import csv
import numpy as np
import os
import argparse

def read_typhoon_path(file_path):
    """從 CSV 檔案中讀取颱風路徑資料，並處理不同的格式"""
    path = []
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 {file_path}")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            # 檢查第一行來判斷格式
            first_row = next(reader)
            f.seek(0)
            # 重新建立 reader
            reader = csv.reader(f)
            # 跳過標頭 (如果有)
            if first_row[0].lower() in ['date', '"date"']:
                next(reader)

            for row in reader:
                try:
                    if len(row) == 4: # date,hr,lat,lon
                        path.append((float(row[2]), float(row[3])))
                    elif len(row) == 3: # datetime,lat,lon
                        path.append((float(row[1]), float(row[2])))
                    elif len(row) == 2: # lat,lon
                        path.append((float(row[0]), float(row[1])))
                    else:
                        # 嘗試從前兩欄讀取，適用於 2025PODUL.csv 的特殊格式
                        if len(row) > 1 and row[0].replace('.', '', 1).isdigit() and row[1].replace('.', '', 1).isdigit():
                           path.append((float(row[0]), float(row[1])))
                        # 嘗試從後兩欄讀取
                        elif len(row) > 2 and row[-2].replace('.', '', 1).isdigit() and row[-1].replace('.', '', 1).isdigit():
                            path.append((float(row[-2]), float(row[-1])))
                        else:
                             print(f"警告：在 {file_path} 中發現無法解析的資料行，已跳過。行內容: {row}")
                except (ValueError, IndexError):
                    print(f"警告：在 {file_path} 中發現無效的資料行，已跳過。行內容: {row}")
                    continue
        except StopIteration:
            print(f"警告：檔案 {file_path} 是空的。")
            return None
    return np.array(path) if path else None

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

def main(args):
    """主函式"""
    base_path_file = args.base_csv
    other_files = args.other_csvs

    base_path = read_typhoon_path(base_path_file)
    if base_path is None or len(base_path) == 0:
        print(f"錯誤：無法讀取基準颱風路徑 {base_path_file} 或路徑為空。")
        return

    results = {}
    for file in other_files:
        other_path = read_typhoon_path(file)
        if other_path is not None and len(other_path) > 0:
            distance = dtw_distance(base_path, other_path)
            results[file] = distance

    if not results:
        print("沒有可比較的颱風路徑。")
        return

    print(f"\n與 {base_path_file} 的路徑相似度比較：")
    print("{:<40} | {:<25}".format("颱風路徑檔案", "DTW 距離 (越小越相似)"))
    print("-" * 68)
    for file, distance in sorted(results.items(), key=lambda item: item[1]):
        print("{:<40} | {:<25.2f}".format(file, distance))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用 DTW 演算法，比較一個基準颱風路徑與多個其他路徑的相似度。")
    parser.add_argument("base_csv", help="基準颱風路徑的 CSV 檔案路徑。")
    parser.add_argument("other_csvs", nargs='+', help="一個或多個要進行比較的其他颱風路徑 CSV 檔案。")
    
    args = parser.parse_args()
    main(args)