with open('C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/tex/Chapter4.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Normalize line endings
content = content.replace('\r\n', '\n')

# 1. Replace the Yoshikawa ellipsoids list (line 407-410)
list_yoshikawa = """\\begin{itemize}
    \\item 當 w → 0 時，橢球體沿某方向退化為零，表示機器人接近\\textbf{運動學奇異點}，喪失在某些笛卡爾方向上的運動能力。
    \\item 當 w 較大時，橢球體飽滿，機器人在各方向皆具有良好的運動與施力能力——即高\\textbf{靈巧度}。
\\end{itemize}"""

replacement_yoshikawa = """當 w → 0 時，橢球體沿某方向退化為零，表示機器人接近\\textbf{運動學奇異點}，喪失在某些笛卡爾方向上的運動能力；當 w 較大時，橢球體飽滿，機器人在各方向皆具有良好的運動與施力能力，即具有高\\textbf{靈巧度}。"""

if list_yoshikawa in content:
    content = content.replace(list_yoshikawa, replacement_yoshikawa)
else:
    # Try with minor space/indent differences
    import re
    content = re.sub(
        r'\\begin\{itemize\}\s*\\item\s*當 w → 0 時，橢球體沿某方向退化為零.*?\\end\{itemize\}',
        replacement_yoshikawa,
        content,
        flags=re.DOTALL
    )

# 2. Replace the AMI variables list (line 419-422)
list_ami = """\\begin{itemize}
    \\item S_k subseteq 1, ldots, M：在底座位置 k 下成功到達的路徑點集合（定位誤差 < ε，ε = 1.5 mm）。
    \\item q_k^{}(j)：成功到達路徑點 j ∈ S_k 時的關節組態。
\\end{itemize}"""

replacement_ami = """其中 S\\_k ⊆ \\{1, ..., M\\} 代表在底座位置 k 下成功到達的路徑點集合（定位誤差 < ε，ε = 1.5 mm），而 q\\_k\\^{}(j) 代表成功到達路徑點 j ∈ S\\_k 時的關節組態。"""

if list_ami in content:
    content = content.replace(list_ami, replacement_ami)
else:
    import re
    content = re.sub(
        r'\\begin\{itemize\}\s*\\item\s*S_k subseteq 1, ldots, M.*?\\end\{itemize\}',
        replacement_ami,
        content,
        flags=re.DOTALL
    )

# 3. Clean up the omni.ui control panel leftovers (lines 665-669)
old_omni_ui = """檢視器使用 Isaac Sim 內建的 \\texttt{omni.ui} 框架建構側邊控制面板，包含以下元素：

檢視器使用 Isaac Sim 內建的 omni.ui 框架建構側邊控制面板，主要包含以下四類元素。第一是 RI 色階圖例，以 jet\\_r 色階顯示可達性指數的對應顏色（紅色代表 0 不可達，藍色代表 1 完全可達）。第二是環境按鈕列表，列出 15 個候選底座位置的按鈕，顯示底座 X 座標、成功率、f\\_g 等數值，點擊後可觸發平滑的底座切換動畫。第三是多個功能按鈕，包括用以找出離腫瘤最近的路徑點並以黃色高亮的「Highlight Nearest WP」、啟動 IK 求解以引導針尖移向腫瘤的「Follow Target (Lula IK)」，以及切換 CMAP 球體可見性的「Hide/Show CMAP Spheres」。第四是狀態標籤，用以即時顯示當前選定的環境、最近路徑點距離、IK 追蹤殘差等資訊。
    \\item \\textbf{狀態標籤}：即時顯示當前選定的環境、最近路徑點距離、IK 追蹤殘差等資訊。
\\end{itemize}"""

new_omni_ui = """檢視器使用 Isaac Sim 內建的 omni.ui 框架建構側邊控制面板，主要包含以下四類元素。第一是 RI 色階圖例，以 jet\\_r 色階顯示可達性指數的對應顏色（紅色代表 0 不可達，藍色代表 1 完全可達）。第二是環境按鈕列表，列出 15 個候選底座位置的按鈕，顯示底座 X 座標、成功率、f\\_g 等數值，點擊後可觸發平滑的底座切換動畫。第三是多個功能按鈕，包括用以找出離腫瘤最近的路徑點並以黃色高亮的「Highlight Nearest WP」、啟動 IK 求解以引導針尖移向腫瘤的「Follow Target (Lula IK)」，以及切換 CMAP 球體可見性的「Hide/Show CMAP Spheres」。第四是狀態標籤，用以即時顯示當前選定的環境、最近路徑點距離、IK 追蹤殘差等資訊。"""

if old_omni_ui in content:
    content = content.replace(old_omni_ui, new_omni_ui)
else:
    # Use regular expression for more flexibility
    import re
    content = re.sub(
        r'檢視器使用 Isaac Sim 內建的 \\\\texttt\{omni\.ui\}.*?狀態標籤.*?\\end\{itemize\}',
        new_omni_ui,
        content,
        flags=re.DOTALL
    )

# 4. Replace the sphere overlay colors list (line 675-679)
list_colors = """\\begin{itemize}
    \\item RI = 1.0（所有 5 種姿態皆可達）rightarrow \\textcolor{blue}{藍色}
    \\item RI = 0.0（完全不可達）rightarrow \\textcolor{red}{紅色}
    \\item 中間值以漸層表示
\\end{itemize}"""

replacement_colors = """具體而言，當 RI = 1.0（所有 5 種姿態皆可達）時映射為藍色；當 RI = 0.0（完全不可達）時映射為紅色；其餘中間值則以漸層色表示。"""

if list_colors in content:
    content = content.replace(list_colors, replacement_colors)
else:
    import re
    content = re.sub(
        r'\\begin\{itemize\}\s*\\item\s*RI = 1\.0.*?\\end\{itemize\}',
        replacement_colors,
        content,
        flags=re.DOTALL
    )

# Write output
with open('C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/tex/Chapter4.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Pass 2 Completed Successfully")
