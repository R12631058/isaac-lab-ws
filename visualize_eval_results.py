"""
可視化評估結果

讀取 eval_results.json 並繪製性能圖表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_results(json_path: str = "eval_results.json"):
    """讀取並可視化評估結果"""
    
    # 讀取 JSON 結果
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 提取數據
    x_positions = [r['x_position'] for r in results]
    mean_returns = [r['mean_return'] for r in results]
    std_returns = [r['std_return'] for r in results]
    success_rates = [r['success_rate'] * 100 for r in results]  # 轉換為百分比
    mean_lengths = [r['mean_episode_length'] for r in results]
    
    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'模型評估結果 - Checkpoint: {Path(data["checkpoint"]).name}', 
                 fontsize=14, fontweight='bold')
    
    # 1. 平均回報 (帶誤差條)
    ax1 = axes[0, 0]
    ax1.errorbar(x_positions, mean_returns, yerr=std_returns, 
                 marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
                 color='#2E86AB', ecolor='#A23B72', label='Mean ± Std')
    ax1.set_xlabel('Robot Arm Base X Position (m)', fontsize=11)
    ax1.set_ylabel('Mean Return', fontsize=11)
    ax1.set_title('平均回報 vs Base 位置', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加數值標籤
    for x, y in zip(x_positions, mean_returns):
        ax1.annotate(f'{y:.1f}', 
                    xy=(x, y), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center', 
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # 2. 成功率
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(x_positions)), success_rates, 
                   color='#06A77D', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Position Index', fontsize=11)
    ax2.set_ylabel('Success Rate (%)', fontsize=11)
    ax2.set_title('成功率', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(x_positions)))
    ax2.set_xticklabels([f'{x:.4f}' for x in x_positions], rotation=45, ha='right')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加百分比標籤
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.0f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. 標準差變化
    ax3 = axes[1, 0]
    ax3.plot(x_positions, std_returns, 
             marker='s', linewidth=2, markersize=8, 
             color='#F18F01', label='Std Dev')
    ax3.set_xlabel('Robot Arm Base X Position (m)', fontsize=11)
    ax3.set_ylabel('Standard Deviation', fontsize=11)
    ax3.set_title('回報標準差 vs Base 位置', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 數據摘要表格
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 創建表格數據
    table_data = []
    table_data.append(['X Position', 'Mean Return', 'Success Rate', 'Std Dev'])
    for r in results:
        table_data.append([
            f"{r['x_position']:.4f}",
            f"{r['mean_return']:.2f}",
            f"{r['success_rate']*100:.0f}%",
            f"{r['std_return']:.2f}"
        ])
    
    table = ax4.table(cellText=table_data, 
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 設置標題行樣式
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 設置交替行顏色
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            for j in range(4):
                table[(i, j)].set_facecolor('#E8E8E8')
    
    ax4.set_title('評估數據摘要', fontsize=12, fontweight='bold', pad=20)
    
    # 調整佈局
    plt.tight_layout()
    
    # 保存圖片
    output_path = Path('eval_results_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 圖表已保存至: {output_path}")
    
    # 顯示圖表
    plt.show()
    
    # 打印統計摘要
    print(f"\n{'='*60}")
    print("評估統計摘要")
    print(f"{'='*60}")
    print(f"測試位置數: {len(results)}")
    print(f"每位置 Episodes: {results[0]['num_episodes']}")
    print(f"X 位置範圍: {min(x_positions):.4f} 到 {max(x_positions):.4f}")
    print(f"平均回報範圍: {min(mean_returns):.2f} 到 {max(mean_returns):.2f}")
    print(f"總體平均回報: {np.mean(mean_returns):.2f} ± {np.std(mean_returns):.2f}")
    print(f"總體成功率: {np.mean(success_rates):.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="可視化評估結果")
    parser.add_argument("--json", type=str, default="eval_results.json", 
                       help="評估結果 JSON 文件路徑")
    
    args = parser.parse_args()
    
    visualize_results(args.json)
