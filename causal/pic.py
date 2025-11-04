import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results(file_paths):
    """加载多个结果文件"""
    results = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            results.append(json.load(f))
    return results

def plot_comparison(results, labels, save_path='comparison.png'):
    """
    绘制对比图
    results: list of result dicts
    labels: list of labels for each result
    """
    metrics = ['parse_success_rate', 'valid_rate', 'novelty_rate', 'recovery_rate']
    metric_names = ['解析成功率', '有效率', '新颖率', '恢复率']
    
    # 提取数据
    means = []
    stds = []
    for result in results:
        mean_vals = [result['statistics'][m]['mean'] for m in metrics]
        std_vals = [result['statistics'][m]['std'] for m in metrics]
        means.append(mean_vals)
        stds.append(std_vals)
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: 柱状图对比
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, (mean, label) in enumerate(zip(means, labels)):
        offset = (i - 1) * width
        axes[0].bar(x + offset, mean, width, label=label, 
                   color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    axes[0].set_xlabel('评估指标', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('分数', fontsize=12, fontweight='bold')
    axes[0].set_title('Qwen-14B因果推理优化对比', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_names)
    axes[0].legend(loc='upper left', framealpha=0.9)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_ylim([0, 1.1])
    
    # 子图2: 折线图展示改进趋势
    for i, (mean, label) in enumerate(zip(means, labels)):
        axes[1].plot(metric_names, mean, marker='o', linewidth=2.5, 
                    markersize=8, label=label, color=colors[i])
    
    axes[1].set_xlabel('评估指标', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('分数', fontsize=12, fontweight='bold')
    axes[1].set_title('优化趋势分析', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {save_path}")
    plt.show()

def plot_detailed_metrics(results, labels, save_path='detailed_metrics.png'):
    """绘制详细指标对比（包含误差条）"""
    metrics = ['parse_success_rate', 'valid_rate', 'novelty_rate', 'recovery_rate']
    metric_names = ['parse_success_rate', 'valid_rate', 'novelty_rate', 'recovery_rate']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = ['#3498db', '#2ecc71', "#b72616"]
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        x = np.arange(len(results))
        means = [r['statistics'][metric]['mean'] for r in results]
        stds = [r['statistics'][metric]['std'] for r in results]
        
        axes[idx].bar(x, means, color=colors[:len(results)], 
                     alpha=0.7, edgecolor='black', linewidth=1)
        axes[idx].errorbar(x, means, yerr=stds, fmt='none', 
                          ecolor='black', capsize=5, linewidth=2)
        
        axes[idx].set_title(name, fontsize=18, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(labels, fontsize=16, rotation=15)
        axes[idx].set_ylim([0, 1.1])
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for i, (m, s) in enumerate(zip(means, stds)):
            axes[idx].text(i, m + s + 0.05, f'{m:.3f}', 
                          ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Qwen14B Causal Task', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"详细图表已保存至: {save_path}")
    plt.show()

def print_comparison_table(results, labels):
    """打印对比表格"""
    metrics = ['parse_success_rate', 'valid_rate', 'novelty_rate', 'recovery_rate']
    metric_names = ['解析成功率', '有效率', '新颖率', '恢复率']
    
    print("\n" + "="*80)
    print("实验结果对比表")
    print("="*80)
    print(f"{'指标':<15}", end='')
    for label in labels:
        print(f"{label:<20}", end='')
    print()
    print("-"*80)
    
    for metric, name in zip(metrics, metric_names):
        print(f"{name:<15}", end='')
        for result in results:
            mean = result['statistics'][metric]['mean']
            std = result['statistics'][metric]['std']
            print(f"{mean:.4f}±{std:.4f}       ", end='')
        print()
    
    print("="*80)
    
    # 打印改进幅度
    if len(results) > 1:
        print("\n改进幅度分析:")
        print("-"*80)
        for metric, name in zip(metrics, metric_names):
            baseline = results[0]['statistics'][metric]['mean']
            final = results[-1]['statistics'][metric]['mean']
            improvement = ((final - baseline) / baseline) * 100
            print(f"{name}: {improvement:+.2f}%")
        print("="*80)

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    file_paths = [
        'results\datasets_s_qwen3-14b.json',      # 原始模型
        'results\datasets_s_qwen3-14bwith_better_prompt.json',  # Prompt优化
        'results\datasets_s_finetuned_qwen.json'    # LoRA微调
    ]
    
    labels = ['Baseline Model', 'Prompt Optimization', 'LoRA Fine-tuning']
    
    # 加载结果
    results = load_results(file_paths)
    
    # 绘制对比图
    plot_comparison(results, labels, 'comparison.png')
    
    # 绘制详细指标图
    plot_detailed_metrics(results, labels, 'detailed_metrics.png')
    
    # 打印对比表格
    print_comparison_table(results, labels)
