import re

# Read the restored original file
with open('C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/tex/Chapter4_restored.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Normalize line endings
content = content.replace('\r\n', '\n')

# 1. Replace the documentclass to support compiling without ctexart/zhnumber
ctex_class = r"\documentclass[a4paper, 12pt]{ctexart}"
cjk_class = """\\documentclass[a4paper, 12pt]{article}
\\usepackage{xeCJK}
\\setCJKmainfont{Microsoft JhengHei}"""

if ctex_class in content:
    content = content.replace(ctex_class, cjk_class)
else:
    print("Warning: ctexart class declaration not found.")

# 2. Add \doi command to custom commands section if not already present
doi_definition = "\\newcommand{\\doi}[1]{DOI: \\href{https://doi.org/#1}{\\texttt{#1}}}"
if "newcommand{\\doi}" not in content:
    # Insert it under another custom command
    content = content.replace(
        "\\newcommand{\\RI}{\\ensuremath{\\text{RI}}}",
        "\\newcommand{\\RI}{\\ensuremath{\\text{RI}}}\n" + doi_definition
    )

# 3. Replace all itemize and enumerate lists with narrative paragraphs
# List 1: Section 1.3
list1_pattern = r'\\begin\{enumerate\}\[label=\\textbf\{Phase \\arabic\*:\},\s*leftmargin=4em\]\s*\\item\s*\\textbf\{CMAP\s*建立與優化搜尋\}——在\s*Isaac\s*Lab\s*中進行大規模\s*GPU\s*並行物理模擬，搜尋每個腫瘤區塊的最佳底座位置\。\s*\\item\s*\\textbf\{資料庫彙整\}——將各區塊的優化結果整合為統一的\s*JSON\s*資料庫（\\texttt\{cmap\\_database\.json\}）\。\s*\\item\s*\\textbf\{查詢與啟動\}——透過\s*Tkinter\s*GUI\s*提供直覺的座標查詢介面，輸入腫瘤座標即可獲得最佳底座建議\。\s*\\item\s*\\textbf\{3D\s*視覺化驗證\}——在\s*Isaac\s*Sim\s*Standalone\s*環境中以\s*Lula\s*IK\s*求解器驗證針尖能否精準觸及腫瘤\。\s*\\end\{enumerate\}'
list1_replacement = (
    '本系統將上述問題拆解為四個階段，分別在不同的軟體平台上執行。'
    '在第一階段（Phase 1: CMAP 建立與優化搜尋）中，系統在 Isaac Lab 中進行大規模 GPU 並行物理模擬，搜尋每個腫瘤區塊的最佳底座位置。'
    '接著在第二階段（Phase 2: 資料庫彙整）中，將各區塊的優化結果整合為統一的 JSON 資料庫（\\texttt{cmap\\_database.json}）。'
    '在第三階段（Phase 3: 查詢與啟動）中，透過 Tkinter GUI 提供直覺的座標查詢介面，輸入腫瘤座標即可獲得最佳底座建議。'
    '最後在第四階段（Phase 4: 3D 視覺化驗證）中，在 Isaac Sim Standalone 環境中以 Lula IK 求解器驗證針尖能否精準觸及腫瘤。'
)
content = re.sub(list1_pattern, list1_replacement, content, flags=re.DOTALL)

# List 2: Section 2.1
list2_pattern = r'\\begin\{enumerate\}\s*\\item\s*\\textbf\{建立全域能力圖譜（Global\s*CMAP）\}：對機器人的整個三維工作空間進行離散化，將其劃分為體素（Voxels）\。在每個體素內，於嵌入球面上均勻取樣多種接近方向，並透過逆運動學（Inverse\s*Kinematics,\s*IK）求解判斷可達性，同時記錄操作度（Manipulability）等品質指標\。\s*\\item\s*\\textbf\{投射為任務導向子空間（RASSCMAP）\}：根據特定手術任務的約束條件（例如腹腔鏡的套管針入口點、脊椎手術的螺釘植入角度），將全域\s*CMAP\s*篩選為符合手術需求的子集\。\s*\\item\s*\\textbf\{多目標最佳化\}：使用基因演算法（Genetic\s*Algorithm,\s*GA）同時最佳化機器人底座位置與手術接入點（如套管針位置），平衡可達性、靈巧度與碰撞避免等多重目標\。\s*\\end\{enumerate\}'
list2_replacement = (
    'RASSCMAP 方法的核心步驟主要包含三個部分：首先是建立全域能力圖譜（Global CMAP），對機器人的整個三維工作空間進行離散化，將其劃分為體素（Voxels）。在每個體素內，於嵌入球面上均勻取樣多種接近方向，並透過逆運動學（Inverse Kinematics, IK）求解判斷可達性，同時記錄操作度（Manipulability）等品質指標。'
    '其次是投射為任務導向子空間（RASSCMAP），根據特定手術任務的約束條件（例如腹腔鏡的套管針入口點、脊椎手術的螺釘植入角度），將全域 CMAP 篩選為符合手術需求的子集。'
    '最後是多目標最佳化，使用基因演算法（Genetic Algorithm, GA）同時最佳化機器人底座位置與手術接入點（如套管針位置），平衡可達性、靈巧度與碰撞避免等多重目標。'
)
content = re.sub(list2_pattern, list2_replacement, content, flags=re.DOTALL)

# List 3: Section 2.2
list3_pattern = (
    r'\\begin\{itemize\}\s*'
    r'\\item\s*\\textbf\{可達性圖譜（Reachability\s*Map,\s*RM）\}：在每個體素位置檢測機器人是否能抵達該位置（及特定方向），記錄為二元值或有效\s*IK\s*解的數量\。\s*'
    r'\\item\s*\\textbf\{能力圖譜（Capability\s*Map,\s*CMAP）\}：RM\s*的延伸版本，在每個體素中不僅記錄可達性，更量化機器人在該位置的操作品質\。品質指標包含操作度（Manipulability）、靈巧度指數（Dexterity\s*Index）與關節餘裕分數（Joint\s*Margin\s*Score）\。\s*'
    r'\\item\s*\\textbf\{可達性指數（Reachability\s*Index,\s*RI）\}：定義為某位置可達方向的比例：\s*'
    r'\\begin\{equation\}\s*'
    r'RI\s*=\s*\\frac\{\\text\{可達方向數量\}\}\{\\text\{總離散方向數量\}\}\s*'
    r'\\end\{equation\}\s*'
    r'\\item\s*\\textbf\{逆可達性圖譜（Inverse\s*Reachability\s*Map,\s*IRM）\}：將\s*RM\s*反轉視角——不是問「機器人從底座能到達哪裡」，而是問「為了到達特定目標，底座應該放在哪裡」\。\s*'
    r'\\end\{itemize\}'
)
list3_replacement = (
    'Sundaram 等人的 CMAP 框架定義了數個核心評估指標。首先是可達性圖譜（Reachability Map, RM），它在每個體素位置檢測機器人是否能抵達該位置（及特定方向），記錄為二元值或有效 IK 解的數量。'
    '其次是能力圖譜（Capability Map, CMAP），作為 RM 的延伸版本，它在每個體素中不僅記錄可達性，更量化機器人在該位置的操作品質，品質指標包含操作度（Manipulability）、靈巧度指數（Dexterity Index）與關節餘裕分數（Joint Margin Score）。'
    '再來是可達性指數（Reachability Index, RI），定義為某位置可達方向的比例，即 RI = 可達方向數量 / 總離散方向數量。'
    '最後是逆可達性圖譜（Inverse Reachability Map, IRM），將 RM 反轉視角——不是問「機器人從底座能到達哪裡」，而是問「為了到達特定目標，底座應該放在哪裡」。'
)
content = re.sub(list3_pattern, list3_replacement, content, flags=re.DOTALL)

# List 4: Section 4.1
list4_pattern = (
    r'\\textbf\{輸出\}（寫入\s*\\texttt\{heatmap/blocks/\}\s*目錄）：\s*'
    r'\\begin\{itemize\}\s*'
    r'\\item\s*\\texttt\{optimal\\_block\\_\\{N\\}\\\.json\}：區塊元數據與最佳底座結果\s*'
    r'\\item\s*\\texttt\{backup\\_live\\_\\{run\\_tag\\}\\\.csv\}：逐任務逐環境的原始結果（支援中斷復原）\s*'
    r'\\item\s*\\texttt\{optimization\\_results\\_\\{run\\_tag\\}\\\.csv\}：各環境的彙整統計\s*'
    r'\\item\s*RASSCMAP\s*3D\s*散佈圖與條形圖（PNG）\s*'
    r'\\end\{itemize\}'
)
list4_replacement = (
    '在輸出方面，系統會將結果寫入 \\texttt{heatmap/blocks/} 目錄下，包含儲存區塊元數據與最佳底座結果的 \\texttt{optimal\\_block\\_\\{N\\}.json}、'
    '紀錄逐任務逐環境原始結果並支援中斷復原的 \\texttt{backup\\_live\\_\\{run\\_tag\\}.csv}、'
    '彙整統計各環境結果的 \\texttt{optimization\\_results\\_\\{run\\_tag\\}.csv}，以及生成的 RASSCMAP 3D 散佈圖與條形圖（PNG）。'
)
content = re.sub(list4_pattern, list4_replacement, content, flags=re.DOTALL)

# List 5: Section 4.5
list5_pattern = (
    r'\\begin\{itemize\}\s*'
    r'\\item\s*\\textbf\{Isaac\s*Lab\}：透過\s*Tensor\s*API\s*批量操作\s*PhysX\s*關節狀態.*?\\\\'
    r'\\item\s*\\textbf\{Isaac\s*Sim\s*Standalone\}：直接透過\s*USD\s*Xform\s*API\s*操作場景.*?\\\\'
    r'\\item\s*\\textbf\{Tkinter\}：Isaac\s*Sim\s*4\.5\.0\s*的內建\s*Python\s*環境.*?\\\\'
    r'\\end\{itemize\}'
)
list5_replacement = (
    '系統之所以橫跨 Isaac Lab 與 Isaac Sim Standalone 兩個執行環境，其核心原因在於兩者在物理狀態管理上的根本差異。'
    '其中，Isaac Lab 透過 Tensor API 批量操作 PhysX 關節狀態，適合 GPU 並行的大規模優化搜尋，然而其透過 \\texttt{write\\_root\\_pose\\_to\\_sim()} 修改的是 PhysX 內部狀態，不一定同步反映在 USD 場景層級，這在除錯 USD 視覺化行為時會造成困擾。'
    '相對地，Isaac Sim Standalone 直接透過 USD Xform API 操作場景，所見即所得，使用 \\texttt{LulaKinematicsSolver} 進行 IK 求解，不依賴 Isaac Lab 的場景抽象層，非常適合作為最終的視覺驗證工具。'
    '另外，由於 Isaac Sim 4.5.0 的內建 Python 環境不包含 \\texttt{\\_tkinter} 模組，因此查詢 GUI 必須在獨立的 Conda 環境中執行，並透過 \\texttt{subprocess.Popen} 呼叫 Isaac Sim 的 \\texttt{python.bat} 來啟動 3D 檢視器。'
)
content = re.sub(list5_pattern, list5_replacement, content, flags=re.DOTALL)

# List 6: Section 6.1
list6_pattern = (
    r'\\begin\{itemize\}\s*'
    r'\\item\s*\\textbf\{左側控制面板\}：提供腫瘤座標.*?\s*'
    r'\\item\s*\\textbf\{右側視覺化面板\}：嵌入\s*Matplotlib.*?\s*'
    r'\\end\{itemize\}'
)
list6_replacement = (
    'Tkinter 查詢介面分為左右兩個面板。左側控制面板提供腫瘤座標 (X, Y, Z) 的輸入欄位、「Query Optimal Base」查詢按鈕、結果顯示標籤（匹配區塊、最佳底座位置、f\\_g、成功率、AMI）以及「Open 3D CMAP Viewer」按鈕；'
    '右側視覺化面板則嵌入 Matplotlib 圖表（透過 FigureCanvasTkAgg 橋接），以 2D 俯視圖繪製線性軌道、區塊工作空間、腫瘤位置與最佳底座。'
)
content = re.sub(list6_pattern, list6_replacement, content, flags=re.DOTALL)

# List 7: Section 6.1.1
list7_pattern = (
    r'\\begin\{enumerate\}\s*'
    r'\\item\s*\\textbf\{第一階段（嚴格邊界匹配）\}：遍歷所有區塊.*?\s*'
    r'\\item\s*\\textbf\{第二階段（距離退避）\}：若座標不在任何區塊內，計算輸入座標到各區塊邊界框的夾限歐幾里得距離.*?\s*'
    r'\\end\{enumerate\}'
)
list7_replacement = (
    '查詢採用兩階段搜尋策略。第一階段為嚴格邊界匹配，遍歷所有區塊並檢查輸入座標是否落在其路徑點邊界框（wp\\_bounds）內，若有多個區塊符合，則以成功率、S\\_r\\^{}min、f\\_g、平均操作度的優先序選出最佳區塊。'
    '第二階段為距離退避，若座標不在任何區塊內，則計算輸入座標到各區塊邊界框的夾限歐幾里得距離：d = √[max(x\\_min - t\\_x, 0, t\\_x - x\\_max)² + max(y\\_min - t\\_y, 0, t\\_y - y\\_max)² + max(z\\_min - t\\_z, 0, t\\_z - z\\_max)²]，並選擇距離最近的區塊，在介面上顯示紅色警告文字。'
)
content = re.sub(list7_pattern, list7_replacement, content, flags=re.DOTALL)

# List 8: Section 6.2.1
list8_pattern = (
    r'\\begin\{itemize\}\s*'
    r'\\item\s*\\textbf\{RI\s*色階圖例\}.*?\\end\{itemize\}'
)
list8_replacement = (
    '檢視器使用 Isaac Sim 內建的 omni.ui 框架建構側邊控制面板，主要包含以下四類元素。第一是 RI 色階圖例，以 jet\\_r 色階顯示可達性指數的對應顏色（紅色代表 0 不可達，藍色代表 1 完全可達）。'
    '第二是環境按鈕列表，列出 15 個候選底座位置的按鈕，顯示底座 X 座標、成功率、f\\_g 等數值，點擊後可觸發平滑的底座切換動畫。'
    '第三是多個功能按鈕，包括用以找出離腫瘤最近的路徑點並以黃色高亮的「Highlight Nearest WP」、啟動 IK 求解以引導針尖移向腫瘤的「Follow Target (Lula IK)」，以及切換 CMAP 球體可見性的「Hide/Show CMAP Spheres」。'
    '第四是狀態標籤，用以即時顯示當前選定的環境、最近路徑點距離、IK 追蹤殘差等資訊。'
)
content = re.sub(list8_pattern, list8_replacement, content, flags=re.DOTALL)

# List 9: Section 6.2.2
list9_pattern = (
    r'\\begin\{itemize\}\s*'
    r'\\item\s*\$RI\s*=\s*1\.0\$.*?\\end\{itemize\}'
)
list9_replacement = (
    '系統在 USD 場景中建立 70 個球體 Prim（路徑 /World/RASSCMAP\\_Overlay/），每個球體的位置對應一個路徑點，顏色根據該路徑點在當前環境下的可達性指數 RI\\_k 以 jet\\_r 色階映射。'
    '具體而言，當 RI = 1.0（所有 5 種姿態皆可達）時映射為藍色；當 RI = 0.0（完全不可達）時映射為紅色；其餘中間值則以漸層色表示。'
    '切換環境時，系統會自動重新計算這 70 個球體的顏色，以直觀反映不同底座位置對各路徑點可達性的影響。'
)
content = re.sub(list9_pattern, list9_replacement, content, flags=re.DOTALL)

# List 10: Section 6.3.2
list10_pattern = (
    r'\\begin\{enumerate\}\s*'
    r'\\item\s*使用預設種子.*?\\end\{enumerate\}'
)
list10_replacement = (
    '為了確保求解出的關節組態位於安全的負分支（joint\\_1 < -0.1 rad），系統採用多種子搜尋策略。'
    '首先使用預設種子 [-75.0°, 37.3°, 64.7°, 78.4°, 9.8°, -88.9°] 進行首次求解。'
    '若求解出的 joint\\_1 為正值，則以 8 個額外的負 joint\\_1 種子重新求解。'
    '接著在所有候選解中，排除 joint\\_1 > -0.1 rad 的不安全解。'
    '若經過上述步驟仍無有效解，則嘗試僅位置（Position-Only）退避求解。'
)
content = re.sub(list10_pattern, list10_replacement, content, flags=re.DOTALL)

# List 11: Section 6.3.3
list11_pattern = (
    r'\\textbf\{動畫速度調整參數\}.*?\\end\{itemize\}'
)
list11_replacement = (
    '動畫速度調整參數位於 \\texttt{sim\\_rasscmap\\_interactive\\_viewer.py} 第 69--71 行，其中 \\texttt{BASE\\_TRANSITION\\_TOTAL\\_STEPS = 90} 代表底座滑動動畫'
    '的總幀數（在 60 FPS 下相當於 1.5 秒），而 \\texttt{ARM\\_FOLLOW\\_TOTAL\\_STEPS = 90} 代表手臂 Follow Target 動畫的總幀數（在 60 FPS 下相當於 1.5 秒）。'
)
content = re.sub(list11_pattern, list11_replacement, content, flags=re.DOTALL)

# List 12: Section 7.1.1
list12_pattern = (
    r'\\begin\{enumerate\}\s*'
    r'\\item\s*\\textbf\{Block\s*0\s*的特殊性\}.*?\\end\{enumerate\}'
)
list12_replacement = (
    '從搜尋結果中可以得出四項關鍵觀察。第一是 Block 0 的特殊性，Block 0 擁有所有區塊中最高的綜合成功率（93.14%），但 S\\_r\\^{}min = 0，'
    '代表至少有一個路徑點的所有 5 種姿態皆無法到達，因此其 f\\_g = 0，在排序優先序中被 S\\_r\\^{}min = 1 的區塊超越。'
    '第二是最佳底座位置的規律，Block 1--9 的最佳底座位置皆為 base\\_x = block\\_center\\_x + 0.1 m，顯示最佳底座位置傾向於略微偏移對應區塊中心約 0.1 m。'
    '第三是操作度的一致性，Block 1--9 的平均操作度 w\\_bar 值域在 [0.01625, 0.01651] 之間，波動僅為 1.6%，表明系統在不同空間區域的靈巧度表現非常穩定。'
    '第四是 Block 7 的最佳表現，在所有 S\\_r\\^{}min = 1 的區塊中，Block 7（base\\_x = 0.8 m）以 87.43% 的成功率高居首位。'
)
content = re.sub(list12_pattern, list12_replacement, content, flags=re.DOTALL)

# List 13: Section 7.3.2
list13_pattern = (
    r'\\begin\{enumerate\}\s*'
    r'\\item\s*更新\s*PhysX\s*Root.*?\\end\{enumerate\}'
)
list13_replacement = (
    '為此，我們在切換底座時增加了三個同步步驟，首先是更新 PhysX Root State（平移與四元數姿態），其次是執行 5 步物理穩定沉降，最後則是同步 Lula 求解器的底座姿態，從而使底座切換後的座標達到 100% 一致。'
)
content = re.sub(list13_pattern, list13_replacement, content, flags=re.DOTALL)

# List 14: Section 7.3.5
list14_pattern = (
    r'\\begin\{enumerate\}\s*'
    r'\\item\s*準備\s*8\s*個預設種子.*?\\end\{enumerate\}'
)
list14_replacement = (
    '為了解決此問題，我們實作了多種子負分支鎖定機制，其執行流程包括：第一，準備 8 個預設種子，每個種子的 Joint 1 值皆設為負數（例如 [-75°, -60°, -90°, -120°, -150°, -45°, -135°, -170°] 等）；'
    '第二，對每個種子各自求解一次 IK；第三，排除所有求解出 Joint 1 > -0.1 rad 的解；第四，從剩餘的有效解中選取末端殘差最小的解。'
    '此機制確保了 100% 的 Follow Target 結果皆為安全的 Elbow-Down 構型。'
)
content = re.sub(list14_pattern, list14_replacement, content, flags=re.DOTALL)

# List 15: Yoshikawa ellipsoids list
list15_pattern = (
    r'\\begin\{itemize\}\s*'
    r'\\item\s*當\s*\$w\s*\\to\s*0\$.*?\\end\{itemize\}'
)
list15_replacement = (
    '當 w → 0 時，速度橢球體沿某方向退化為零，表示機器人接近\\textbf{運動學奇異點}，喪失在某些笛卡爾方向上的運動能力；'
    '當 w 較大時，橢球體飽滿，機器人在各方向皆具有良好的運動與施力能力，即具有高\\textbf{靈巧度}。'
)
content = re.sub(list15_pattern, list15_replacement, content, flags=re.DOTALL)

# List 16: AMI variables list
list16_pattern = (
    r'\\begin\{itemize\}\s*'
    r'\\item\s*\$\\mathcal\{S\}\_k\s*\\subseteq\s*\\\{1,\\\s*\\ldots,\\\s*M\\\}\$.*?\\end\{itemize\}'
)
list16_replacement = (
    '其中 S\\_k ⊆ \\{1, ..., M\\} 代表在底座位置 k 下成功到達的路徑點集合（定位誤差 < ε，ε = 1.5 mm），而 q\\_k\\^{}(j) 代表成功到達路徑點 j ∈ S\\_k 時的關節組態。'
)
content = re.sub(list16_pattern, list16_replacement, content, flags=re.DOTALL)


# 4. Replace block equations with centered Unicode text
block_eqs = [
    # Equation 1: RI
    (
        r"""\begin{equation}
        RI = \frac{\text{可達方向數量}}{\text{總離散方向數量}}
    \end{equation}""",
        r"""\begin{center}
RI = 可達方向數量 / 總離散方向數量
\end{center}"""
    ),
    # Equation 2: Offset
    (
        r"""\begin{equation}
    \mathbf{p}_{\text{tm5}} = \mathbf{p}_{\text{base}} + \mathbf{T}, \quad
    \mathbf{T} = \mathbf{p}_{\text{tm5}}^{\text{default}} - \mathbf{p}_{\text{base}}^{\text{default}}
    \label{eq:offset}
\end{equation}""",
        r"""\begin{center}
p\_tm5 = p\_base + T，其中 T = p\_tm5\^{}default - p\_base\^{}default
\end{center}"""
    ),
    # Equation 3: NDI_UM
    (
        r"""\begin{equation}
    T_{\text{NDI\_UM}} = T_{\text{World\_NDI}}^{-1} \cdot T_{\text{World\_UM}}
    \label{eq:ndi_um}
\end{equation}""",
        r"""\begin{center}
T\_NDI\_UM = T\_World\_NDI⁻¹ × T\_World\_UM
\end{center}"""
    ),
    # Equation 4: block shift
    (
        r"""\begin{equation}
    \mathbf{w}_j^{(b)} = \mathbf{w}_j^{(0)} + \texttt{block\_center}(b), \quad
    \texttt{block\_center}(b) = (b \times 0.1,\; 0,\; 0)^T \text{ m}
    \label{eq:block_shift}
\end{equation}""",
        r"""\begin{center}
w\_j^(b) = w\_j^(0) + block\_center(b)，其中 block\_center(b) = (b × 0.1, 0, 0)ᵀ m
\end{center}"""
    ),
    # Equation 5: grid
    (
        r"""\begin{equation}
    \mathcal{G} = \{x_i \mid x_i \in \text{linspace}(0.1,\; 1.5,\; 15)\}, \quad y = 0.0 \text{ m}
    \label{eq:grid}
\end{equation}""",
        r"""\begin{center}
G = \{x\_i | x\_i ∈ linspace(0.1, 1.5, 15)\}, y = 0.0 m
\end{center}"""
    ),
    # Equation 6: Jacobian relation
    (
        r"""\begin{equation}
    \dot{\mathbf{x}} = \mathbf{J}(\mathbf{q})\, \dot{\mathbf{q}}
    \label{eq:jacobian}
\end{equation}""",
        r"""\begin{center}
ẋ = J(q) q̇
\end{center}"""
    ),
    # Equation 7: Yoshikawa
    (
        r"""\begin{equation}
    \boxed{w(\mathbf{q}) = \left| \det\bigl(\mathbf{J}(\mathbf{q})\bigr) \right|}
    \label{eq:manip}
\end{equation}""",
        r"""\begin{center}
w(q) = |det(J(q))|
\end{center}"""
    ),
    # Equation 8: Speed ellipsoid
    (
        r"""\begin{equation}
    \mathcal{E}(\mathbf{q}) = \left\{ \dot{\mathbf{x}} \in \mathbb{R}^6 \;\middle|\; \|\dot{\mathbf{q}}\| \leq 1,\; \dot{\mathbf{x}} = \mathbf{J}(\mathbf{q})\dot{\mathbf{q}} \right\}
\end{equation}""",
        r"""\begin{center}
E(q) = \{ẋ ∈ ℝ⁶ | ||q̇|| ≤ 1, ẋ = J(q)q̇\}
\end{center}"""
    ),
    # Equation 9: RI counter
    (
        r"""\begin{equation}
    RI_j = \frac{\text{路徑點 } j \text{ 成功到達的姿態數量}}{5}
    \label{eq:ri}
\end{equation}""",
        r"""\begin{center}
RI\_j = (路徑點 j 成功到達的姿態數量) / 5
\end{center}"""
    ),
    # Equation 10: AMI
    (
        r"""\begin{equation}
    \boxed{\AMI_j = \frac{1}{|\mathcal{S}_j|} \sum_{i \in \mathcal{S}_j} w\!\left(\mathbf{q}_j^{(i)}\right) = \frac{1}{|\mathcal{S}_j|} \sum_{i \in \mathcal{S}_j} \left|\det\!\left(\mathbf{J}\!\left(\mathbf{q}_j^{(i)}\right)\right)\right|}
    \label{eq:ami}
\end{equation}""",
        r"""\begin{center}
AMI\_j = (1 / |S\_j|) ∑\_(i ∈ S\_j) w(q\_j^(i)) = (1 / |S\_j|) ∑\_(i ∈ S\_j) |det(J(q\_j^(i)))|
\end{center}"""
    ),
    # Equation 11: fg_score
    (
        r"""\begin{equation}
    \boxed{f_g = S_r^{\min} \cdot \prod_{k=1}^{m} \AMI_k \cdot c_s}
    \label{eq:fg}
\end{equation}""",
        r"""\begin{center}
f\_g = S\_r\^{}min × ∏\_(k=1)ᵐ AMI\_k × c\_s
\end{center}"""
    ),
    # Equation 12: selection priority
    (
        r"""\begin{equation}
    \mathbf{p}^* = \arg\max_{\mathbf{p}_k \in \mathcal{G}} \left( \text{success\_rate},\; S_r^{\min},\; f_g,\; \overline{w} \right)
    \label{eq:selection}
\end{equation}""",
        r"""\begin{center}
p* = arg max\_(p\_k ∈ G) (success\_rate, S\_r\^{}min, f\_g, w\_bar)
\end{center}"""
    ),
    # Equation 13: DLS IK
    (
        r"""\begin{equation}
    \Delta\mathbf{q} = \mathbf{J}^T \left(\mathbf{J}\mathbf{J}^T + \lambda^2 \mathbf{I}\right)^{-1} \Delta\mathbf{x}
    \label{eq:dls}
\end{equation}""",
        r"""\begin{center}
Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ Δx
\end{center}"""
    ),
    # Equation 14: PI controller
    (
        r"""\begin{equation}
    \Delta\mathbf{x}_{\text{boosted}} = \Delta\mathbf{x} + \mathbf{I}_{\text{cart}}, \quad
    \mathbf{I}_{\text{cart}} = \text{clamp}\!\left(\sum_{t} 0.02 \cdot \Delta\mathbf{x}_t,\ -0.05,\ 0.05\right)
\end{equation}""",
        r"""\begin{center}
Δx\_boosted = Δx + I\_cart，其中 I\_cart = clamp(∑\_t 0.02 × Δx\_t, -0.05, 0.05)
\end{center}"""
    ),
    # Equation 15: joint clamp
    (
        r"""\begin{equation}
    q_i^{\text{target}} = \text{clamp}(q_i^{\text{target}},\ q_i^{\min},\ q_i^{\max})
\end{equation}""",
        r"""\begin{center}
q\_i\^{}target = clamp(q\_i\^{}target, q\_i\^{}min, q\_i\^{}max)
\end{center}"""
    ),
    # Equation 16: distance backoff
    (
        r"""\begin{equation}
        d = \sqrt{\max(x_{\min} - t_x, 0, t_x - x_{\max})^2 + \max(y_{\min} - t_y, 0, t_y - y_{\max})^2 + \max(z_{\min} - t_z, 0, t_z - z_{\max})^2}
    \end{equation}""",
        r"""\begin{center}
d = √[max(x\_min - t\_x, 0, t\_x - x\_max)² + max(y\_min - t\_y, 0, t\_y - y\_max)² + max(z\_min - t\_z, 0, t\_z - z\_max)²]
\end{center}"""
    ),
    # Equation 17: ease in out
    (
        r"""\begin{equation}
    \mathbf{p}(t) = \mathbf{p}_{\text{start}} + (\mathbf{p}_{\text{end}} - \mathbf{p}_{\text{start}}) \cdot \frac{1 - \cos(\pi \cdot t)}{2}, \quad t = \frac{\text{frame}}{\text{total\_frames}}
    \label{eq:ease_in_out}
\end{equation}""",
        r"""\begin{center}
p(t) = p\_start + (p\_end - p\_start) × (1 - cos(π × t)) / 2，其中 t = frame / total\_frames
\end{center}"""
    ),
    # Equation 18: flange compensation
    (
        r"""\begin{equation}
    T_{\text{flange}}^{\text{target}} = T_{\text{tumor}}^{\text{world}} \cdot (T_{\text{flange} \to \text{tip}}^{\text{local}})^{-1}
    \label{eq:flange_comp}
\end{equation}""",
        r"""\begin{center}
T\_flange\^{}target = T\_tumor\^{}world × (T\_(flange → tip)\^{}local)⁻¹
\end{center}"""
    ),
    # Equation 19: arm interpolation
    (
        r"""\begin{equation}
    \theta_i(t) = q_{i,\text{start}} + (q_{i,\text{target}} - q_{i,\text{start}}) \cdot \frac{1 - \cos(\pi \cdot t)}{2}, \quad t = \frac{\text{frame}}{N_{\text{total}}}
    \label{eq:arm_interp}
\end{equation}""",
        r"""\begin{center}
θ\_i(t) = q\_(i,start) + (q\_(i,target) - q\_(i,start)) × (1 - cos(π × t)) / 2，其中 t = frame / N\_total
\end{center}"""
    ),
    # Equation 20: arm interpolation bug 3
    (
        r"""\begin{equation}
    \theta_i(t) = q_{i,\text{start}} + (q_{i,\text{target}} - q_{i,\text{start}}) \cdot \frac{1 - \cos(\pi t)}{2}
\end{equation}""",
        r"""\begin{center}
θ\_i(t) = q\_(i,start) + (q\_(i,target) - q\_(i,start)) × (1 - cos(π × t)) / 2
\end{center}"""
    )
]

for src, dest in block_eqs:
    if src in content:
        content = content.replace(src, dest)
    else:
        src_norm = src.replace('\r\n', '\n')
        content_norm = content.replace('\r\n', '\n')
        if src_norm in content_norm:
            content = content_norm.replace(src_norm, dest)
        else:
            print(f"Warning: Exact block equation match failed.")

# 5. Clean up inline equations $ ... $ and convert them to unicode text
def clean_math(match):
    val = match.group(1).strip()
    
    # Specific standard symbol replacements inside math mode
    repls = {
        r'\mathbf{p}_{\text{tm5}}': 'p_tm5',
        r'\mathbf{p}_{\text{base}}': 'p_base',
        r'\mathbf{T}': 'T',
        r'\mathbf{p}_{\text{tm5}}^{\text{default}}': 'p_tm5^default',
        r'\mathbf{p}_{\text{base}}^{\text{default}}': 'p_base^default',
        r'\mathbf{w}_j^{(0)}': 'w_j^(0)',
        r'\mathbf{w}_j^{(b)}': 'w_j^(b)',
        r'\mathbf{w}_j^{(k)}': 'w_j^(k)',
        r'\mathbf{p}_{\text{needle\_tip}}^{(k)}': 'p_needle_tip^(k)',
        r'\mathbf{q}_{\text{target}}': 'q_target',
        r'\mathbf{q}': 'q',
        r'\mathbf{J}': 'J',
        r'\mathbf{I}': 'I',
        r'\mathbf{x}': 'x',
        r'\mathbf{v}': 'v',
        r'\mathbf{p}_k': 'p_k',
        r'\mathbf{p}^*': 'p*',
        r'\mathbf{q}_k^{(j)}': 'q_k^(j)',
        r'\mathcal{G}': 'G',
        r'\mathcal{S}_k': 'S_k',
        r'\mathcal{S}_j': 'S_j',
        r'\mathcal{E}': 'E',
        r'\Delta\mathbf{q}': 'Δq',
        r'\Delta\mathbf{x}': 'Δx',
        r'\Delta\mathbf{q}^{(k)}': 'Δq^(k)',
        r'\Delta\mathbf{x}^{(k)}': 'Δx^(k)',
        r'\Delta_b': 'Δ_b',
        r'\Theta': 'Θ',
        r'\det': 'det',
        r'\lambda': 'λ',
        r'\epsilon': 'ε',
        r'\sigma_i': 'σ_i',
        r'\theta': 'θ',
        r'\pi': 'π',
        r'\omega': 'ω',
        r'\approx': '≈',
        r'\in': '∈',
        r'\times': '×',
        r'\to': '→',
        r'\mid': '|',
        r'\cdot': '×',
        r'\sum': '∑',
        r'\prod': '∏',
        r'\bar{w}': 'w_bar',
        r'\overline{w}': 'w_bar',
        r'\overline{w}_k': 'w_bar_k',
        r'^{\min}': '^min',
        r'^{\max}': '^max',
        r'^{\text{target}}': '^target',
        r'^{\text{default}}': '^default',
        r'\fg': 'f_g',
        r'\srmin': 'S_r^min',
        r'\AMI': 'AMI',
        r'\RI': 'RI',
        # remove bold and text fonts
        r'\mathbf': '',
        r'\mathrm': '',
        r'\mathcal': '',
        r'\texttt': '',
        r'\text': '',
        r'\boxed': '',
        r'\rightarrow': '→',
        r'\leftarrow': '←',
        r'\textcolor{blue}': '',
        r'\textcolor{red}': '',
        r'\subseteq': '⊆',
        r'\dots': '...',
        r'\ldots': '...',
        # spacing
        r'\;': ' ',
        r'\,': ' ',
        r'\!': '',
    }
    
    # Run key replacements
    for k, v in repls.items():
        val = val.replace(k, v)
        
    # Standard cleanup of braces
    val = val.replace('{', '').replace('}', '')
    
    # CRITICAL FIX: Strip all LaTeX formatting backslashes FIRST
    val = val.replace('\\', '')
    
    # CRITICAL FIX: Escape underscores and carets AFTER backslash stripping is done
    val = val.replace('_', '\\_')
    val = val.replace('^', '\\^{}')
    
    return val

# Replace all $...$ with their clean representations
content = re.sub(r'\$([^\$]+)\$', clean_math, content)

# 6. Apply direct CJK macro replacements in text body (like \fg, \srmin etc. outside math mode)
content = content.replace('\\fg', 'f\\_g')
content = content.replace('\\srmin', 'S\\_r\\^{}min')
content = content.replace('\\AMI', 'AMI')
content = content.replace('\\RI', 'RI')

# Write the final text
with open('C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/tex/Chapter4.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Conversion and Escaping Completed Successfully")
