#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import random
import argparse
from collections import Counter, defaultdict

def read_categories(file_path):
    """カテゴリ定義ファイルを読み込む関数"""
    categories = {}
    category_order = []  # カテゴリ名の順序を保持するリスト
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                category_name = parts[0]
                # パターンはタブ区切りの2番目以降の部分をすべて含める
                pattern = '\t'.join(parts[1:]) if len(parts) > 2 else parts[1]
                categories[category_name] = pattern
                # カテゴリ名の順序を記録
                if category_name not in category_order:
                    category_order.append(category_name)
    
    return categories, category_order

def process_log_entries(log_file, categories):
    """ログファイルを読み込み、カテゴリごとに集計し、ログ内容も保存する関数"""
    category_counter = Counter()
    category_logs = defaultdict(list)
    
    with open(log_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                category_name = parts[0]
                log_content = '\t'.join(parts[1:]) if len(parts) > 2 else parts[1]
                
                if category_name in categories:
                    category_counter[category_name] += 1
                    category_logs[category_name].append(log_content)
    
    return category_counter, category_logs

def display_report(category_counts, category_logs, categories, category_order, show_all=False):
    """
    レポートを表示する関数
    show_all=True の場合は全サンプルを表示、False の場合は最大10件をランダムに表示
    """
    # 合計の表示（先頭に表示）
    total = sum(category_counts.values())
    print(f"合計: {total}件")
    print()
    print()
    print("-----------------")
    
    # カテゴリ別の集計結果とパターン、サンプルを表示
    # cate.csvの順序に従って表示
    for category in category_order:
        # カテゴリがログに存在する場合のみ表示
        if category in category_counts:
            count = category_counts[category]
            print(f"Category: {category} {count}件")
            print(f"Pattern : {categories[category]}")
            print("Samples :")
            
            # カテゴリに対応するログからサンプルを選択
            log_samples = category_logs[category]
            
            if show_all:
                # 全サンプルを表示
                samples_to_show = log_samples
            else:
                # ランダムに最大10件を選択
                samples_to_show = min(10, len(log_samples))
                samples_to_show = random.sample(log_samples, samples_to_show)
            
            # サンプルを表示（番号付き）
            for i, sample in enumerate(samples_to_show, 1):
                # パターン部分を削除して表示する
                clean_sample = sample.split(categories[category])[0].rstrip()
                print(f" {i:02d}. {clean_sample}")
            print()
            print()
            print("-----------------")

def main():
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='ログファイルのカテゴリ別集計とサンプル表示')
    parser.add_argument('--report-summary', action='store_true', 
                        help='カテゴリごとのサマリーレポートを表示（最大10件のランダムサンプル）')
    parser.add_argument('--report-all', action='store_true', 
                        help='カテゴリごとの全サンプルを表示')
    args = parser.parse_args()
    
    # レポートが指定されていない場合は何も表示しない
    if not (args.report_summary or args.report_all):
        print("レポート表示するには --report-summary または --report-all オプションを指定してください。")
        return
    
    # カテゴリ定義ファイルを読み込む
    categories, category_order = read_categories('cate.csv')
    
    # ログファイルを読み込んでカテゴリごとに集計とログ内容の保存
    category_counts, category_logs = process_log_entries('defined.log', categories)
    
    # レポート表示
    if args.report_all:
        display_report(category_counts, category_logs, categories, category_order, show_all=True)
    elif args.report_summary:
        display_report(category_counts, category_logs, categories, category_order, show_all=False)

if __name__ == "__main__":
    main()

# ex)
# ./04_categoman_no3.py --report-all
# ./04_categoman_no3.py --report-summary
#
