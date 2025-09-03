# 颱風路徑相似度分析工具

這是一個功能強大的命令列工具，旨在分析與比較颱風路徑的相似度。它能夠讀取一個基準颱風路徑，並與一個龐大的歷史颱風資料庫進行比對，使用多種演算法計算相似度分數，並產生詳細的報告與視覺化地圖。

## 主要功能

*   **多演算法支援**: 整合 DTW、LCSS、Hausdorff 及特徵向量等多種演算法，從不同維度評估路徑相似性。
*   **綜合分數排名**: 可將多個演算法的分數正規化並加權，計算出一個綜合排名分數，提供更全面的評估。
*   **高效能計算**: 利用多核心平行處理（Multiprocessing）大幅加速大量歷史路徑的比對過程。
*   **彈性資料篩選**:
    *   可依據年份範圍篩選歷史颱風。
    *   可設定一個地理半徑，將歷史路徑裁切至基準路徑周圍，專注於重疊區域的比較。
*   **豐富的輸出格式**:
    *   在終端機顯示清晰的排名表格。
    *   將詳細結果匯出為 CSV 檔案。
    *   自動繪製並儲存最相似路徑與基準路徑的比較地圖。
*   **友善的顯示**: 支援讀取颱風中文名稱對照表，讓結果更直觀易讀。

## 安裝說明

1.  **環境要求**:
    *   Python 3.8 或更高版本。

2.  **安裝套件**:
    專案的核心依賴項已整理在 `requirements.minimal.txt` 中。請執行以下指令進行安裝：
    ```bash
    pip install -r requirements.minimal.txt
    ```

3.  **Cartopy 安裝注意事項**:
    `Cartopy` 套件依賴於系統底層函式庫，直接使用 `pip` 安裝可能會失敗。**強烈建議使用 `conda` 進行安裝**，以確保所有依賴項都能被正確處理：
    ```bash
    conda install -c conda-forge cartopy
    ```

## 使用方法

本工具的主要執行腳本為 `compare_json_paths.py`。

### 基本指令格式

```bash
python compare_json_paths.py <基準颱風CSV路徑> [選項]
```

### 常用範例

1.  **基本比較**
    *   比較 `2019LEKIMA.csv` 與資料庫中所有颱風，並依預設的 DTW 分數排序，顯示前 5 名。
    ```bash
    python compare_json_paths.py data/測試用/2019LEKIMA.csv --top_n 5
    ```

2.  **指定排序演算法並產生報告與地圖**
    *   改用 LCSS 演算法排序，並將結果儲存為 CSV 檔案 (`-o`) 和地圖 (`--plot`)。
    ```bash
    python compare_json_paths.py data/測試用/2019LEKIMA.csv --sort_by lcss --top_n 5 -o --plot
    ```

3.  **使用綜合分數排名 (自訂權重)**
    *   使用綜合分數 (`composite`) 排序，並透過 `--weights` 自訂權重 (順序: DTW, LCSS, Hausdorff, Feature)。此處設定 DTW 和 LCSS 的權重較高。
    ```bash
    python compare_json_paths.py data/測試用/2019LEKIMA.csv --sort_by composite --weights 0.4 0.4 0.1 0.1 -o --plot
    ```

4.  **篩選特定年份並進行路徑裁切**
    *   只比較 2010-2020 年間的颱風，並且在計算前，先將歷史路徑裁切至基準路徑周圍 250 公里範圍內。
    ```bash
    python compare_json_paths.py data/測試用/2019LEKIMA.csv --start_year 2010 --end_year 2020 --clip_dist 250
    ```
    * 常用: dtw演算法，在計算前，先將歷史路徑裁切至基準路徑周圍 50 公里範圍內。
    ```bash
    python compare_json_paths.py data\測試用\2025PODUL.csv --top_n 5 --sort_by dtw -o --plot --clip_dist 50
    ```

### 命令列參數說明

| 參數 | 說明 |
| :--- | :--- |
| `base_csv` | **(必須)** 基準颱風路徑的 CSV 檔案。 |
| `--sort_by` | 排序依據。可選: `dtw`, `lcss`, `hausdorff`, `feature`, `composite`。預設: `dtw`。 |
| `--weights` | 計算綜合分數時，四種演算法的權重 (DTW, LCSS, Hausdorff, Feature)。總和應為 1。 |
| `--top_n` | 顯示前 N 個結果。預設: `10`。 |
| `--start_year`, `--end_year` | 篩選歷史颱風的年份範圍 (包含)。 |
| `--clip_dist` | 路徑裁切的距離半徑 (公里)。預設不過濾。 |
| `-o`, `--output` | 將結果儲存為 CSV 檔案。 |
| `--plot` | 繪製並儲存最相似路徑的地圖。 |
| `--json_db` | 指定歷史颱風 JSON 資料庫路徑。 |
| `--cname_db` | 指定颱風中文名稱對照表路徑。 |

## 檔案結構

```
.
├── compare_json_paths.py   # 主要執行腳本
├── requirements.minimal.txt# 專案必要套件
├── README.md               # 本說明檔案
├── data/
│   ├── mit_typhoon_paths.json  # 歷史颱風路徑資料庫
│   ├── typhoon_cname.csv       # 颱風中文名稱對照表
│   └── 測試用/                 # 存放基準颱風 CSV 檔案的範例目錄
│       └── 2019LEKIMA.csv
├── src/
│   └── plot_paths_on_map.py    # 地圖繪製工具
└── results/                    # 輸出目錄 (自動生成)
    └── 2019LEKIMA/
        ├── dtw.csv
        ├── dtw.png
        └── ...
```

## 輸出說明

*   **終端機**: 程式執行後，會在終端機印出一個格式化的表格，顯示最相似的前 N 個颱風及其各項分數。
*   **CSV 檔案**: 如果使用 `-o` 選項，詳細結果會儲存於 `results/<基準颱風名>/<排序演算法>.csv`。
*   **地圖圖片**: 如果使用 `--plot` 選項，比較圖會儲存於 `results/<基準颱G風名>/<排序演算法>.png`。
