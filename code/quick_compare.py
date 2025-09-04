#!/usr/bin/env python3
"""
简化版推理结果比较工具
快速比较两个推理结果文件并输出关键统计信息
"""

import os
import sys
from collections import Counter, defaultdict

def load_results(file_path):
    """加载结果文件"""
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    filename = parts[0]
                    prediction = int(parts[1])
                    results[filename] = prediction
    return results

def quick_compare(file1_path, file2_path):
    """快速比较两个结果文件"""
    print(f"比较文件:")
    print(f"  文件1: {file1_path}")
    print(f"  文件2: {file2_path}")
    print("-" * 60)
    
    # 加载结果
    results1 = load_results(file1_path)
    results2 = load_results(file2_path)
    
    # 基本统计
    print(f"基本统计:")
    print(f"  文件1样本数: {len(results1)}")
    print(f"  文件2样本数: {len(results2)}")
    
    # 找到共同文件
    common_files = set(results1.keys()) & set(results2.keys())
    file1_only = set(results1.keys()) - set(results2.keys())
    file2_only = set(results2.keys()) - set(results1.keys())
    
    print(f"  共同文件数: {len(common_files)}")
    print(f"  仅在文件1中: {len(file1_only)}")
    print(f"  仅在文件2中: {len(file2_only)}")
    
    # 分析差异
    differences = {}
    for filename in common_files:
        if results1[filename] != results2[filename]:
            differences[filename] = {
                'pred1': results1[filename],
                'pred2': results2[filename],
                'diff': results2[filename] - results1[filename]
            }
    
    print(f"  预测不同的文件: {len(differences)}")
    if common_files:
        consistency = (len(common_files) - len(differences)) / len(common_files) * 100
        print(f"  一致性: {consistency:.2f}%")
    
    print("\n" + "=" * 60)
    
    # 类别分布对比
    counter1 = Counter(results1.values())
    counter2 = Counter(results2.values())
    
    print("类别分布对比:")
    print("类别\t文件1\t文件2\t差异\t百分比变化")
    print("-" * 50)
    
    all_classes = sorted(set(counter1.keys()) | set(counter2.keys()))
    for cls in all_classes:
        count1 = counter1.get(cls, 0)
        count2 = counter2.get(cls, 0)
        diff = count2 - count1
        
        if count1 > 0:
            pct_change = (diff / count1) * 100
            print(f"{cls}\t{count1}\t{count2}\t{diff:+d}\t{pct_change:+.1f}%")
        else:
            print(f"{cls}\t{count1}\t{count2}\t{diff:+d}\t+∞%")
    
    # 差异分析
    if differences:
        print(f"\n差异详细分析 (共 {len(differences)} 个不同预测):")
        print("-" * 60)
        
        # 差异值统计
        diff_counter = Counter()
        transition_counter = defaultdict(int)
        
        for filename, diff_info in differences.items():
            diff_val = diff_info['diff']
            diff_counter[diff_val] += 1
            
            from_pred = diff_info['pred1']
            to_pred = diff_info['pred2']
            transition_counter[(from_pred, to_pred)] += 1
        
        print("差异值分布:")
        for diff_val in sorted(diff_counter.keys()):
            count = diff_counter[diff_val]
            pct = count / len(differences) * 100
            print(f"  差异 {diff_val:+d}: {count} 个文件 ({pct:.1f}%)")
        
        print("\n最常见的预测变化:")
        most_common_transitions = sorted(transition_counter.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
        for (from_pred, to_pred), count in most_common_transitions:
            pct = count / len(differences) * 100
            print(f"  {from_pred} → {to_pred}: {count} 次 ({pct:.1f}%)")
        
        # 显示一些具体的差异示例
        print(f"\n差异示例 (前10个):")
        print("文件名\t\t文件1预测\t文件2预测\t差异")
        print("-" * 60)
        for i, (filename, diff_info) in enumerate(sorted(differences.items())[:10]):
            print(f"{filename}\t\t{diff_info['pred1']}\t\t{diff_info['pred2']}\t\t{diff_info['diff']:+d}")
        
        if len(differences) > 10:
            print(f"... 还有 {len(differences) - 10} 个差异")

def main():
    if len(sys.argv) != 3:
        print("用法: python quick_compare.py <结果文件1> <结果文件2>")
        print("例如: python quick_compare.py result1.txt result2.txt")
        sys.exit(1)
    
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    
    # 检查文件是否存在
    if not os.path.exists(file1_path):
        print(f"错误: 文件 {file1_path} 不存在")
        sys.exit(1)
    
    if not os.path.exists(file2_path):
        print(f"错误: 文件 {file2_path} 不存在")
        sys.exit(1)
    
    quick_compare(file1_path, file2_path)

if __name__ == "__main__":
    main()
