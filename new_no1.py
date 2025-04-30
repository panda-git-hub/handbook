#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter

# パターン抽出用の正規表現
mac_pattern = re.compile(r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b')
ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
date_pattern1 = re.compile(r'(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2}\s\d{2}:\d{2}:\d{2}(?:\s\d{4})?')
date_pattern2 = re.compile(r'\d{4}[/-]\d{2}[/-]\d{2}\s\d{2}:\d{2}:\d{2}')
date_pattern3 = re.compile(r'\d{2}[/-]\w{3}[/-]\d{4}:\d{2}:\d{2}:\d{2}')
num_pattern = re.compile(r'\b\d+\b')
url_path_pattern = re.compile(r'/[a-zA-Z0-9_/.-]+')
hostname_pattern = re.compile(r'\b[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+\b')

def preprocess_log(log_line):
    """ログを前処理してパターン抽出する"""
    processed = log_line
    
    # 日時表記の置換
    processed = date_pattern1.sub("DATETIME", processed)
    processed = date_pattern2.sub("DATETIME", processed)
    processed = date_pattern3.sub("DATETIME", processed)
    
    # IPアドレスの置換
    processed = ip_pattern.sub("IP_ADDR", processed)

    # MACアドレスの置換
    processed = mac_pattern.sub("MACADDR", processed)
    
    # URLパスの置換
    processed = url_path_pattern.sub("/PATH", processed)
    
    # ホスト名の置換
    processed = hostname_pattern.sub("HOSTNAME", processed)
    
    # 数値の置換 (他のパターン変換後に行う)
    processed = num_pattern.sub("NUM", processed)
    
    return processed

def find_category_candidates():
    """教師なしでカテゴリ候補を生成する"""
    # ログファイルの読み込みと前処理
    processed_logs = []
    original_logs = []
    
    with open('analy.log', 'r', encoding='utf-8') as log_file:
        for log_line in log_file:
            log_line = log_line.strip()
            if not log_line:
                continue
                
            processed_line = preprocess_log(log_line)
            processed_logs.append(processed_line)
            original_logs.append(log_line)
    
    if not processed_logs:
        print("No logs found in analy.log")
        return
    
    # 既存のカテゴリ定義があれば読み込む
    existing_patterns = set()
    if os.path.exists('cate.csv') and os.path.getsize('cate.csv') > 0:
        try:
            categories = pd.read_csv('cate.csv', sep='\t', names=['category', 'pattern'])
            existing_patterns = set(categories['pattern'].tolist())
        except:
            print("Warning: Could not read existing categories from cate.csv")
    
    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(processed_logs)
    
    # DBSCANクラスタリング
    # eps: クラスタを形成する最大距離、min_samples: クラスタを形成する最小サンプル数
    clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(tfidf_matrix)
    
    # クラスタリング結果の取得
    labels = clustering.labels_
    
    # クラスタごとにグループ化
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:  # ノイズポイント（孤立点）
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((original_logs[i], processed_logs[i]))
    
    # 各クラスタの代表パターンを抽出して表示
    print(f"Found {len(clusters)} potential log pattern categories:")
    print("=" * 80)
    
    for cluster_id, logs in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
        # 最も頻出するパターンを抽出
        pattern_counter = Counter(log[1] for log in logs)
        representative_pattern = pattern_counter.most_common(1)[0][0]
        
        # 既存のパターンと一致するものはスキップ
        if representative_pattern in existing_patterns:
            continue
        
        # クラスタの情報を表示
        print(f"Cluster {cluster_id+1} (Count: {len(logs)})")
        print(f"Representative pattern: {representative_pattern}")
        print("Example logs:")
        # 最大10個のサンプルログを表示（番号を揃えて表示）
        for i, (original, processed) in enumerate(logs[:10]):
            print(f"  {i+1:2d}. {original}")
        
        # カテゴリ候補の提案
        # パターンから特徴的な単語を抽出して候補名を生成
        words = representative_pattern.split()
        category_name_candidate = "_".join([w for w in words[:2] if w not in 
                                         {"DATETIME", "IP_ADDR", "NUM", "/PATH", "HOSTNAME"}])
        if not category_name_candidate:
            category_name_candidate = "log_type_" + str(cluster_id+1)
        
        print(f"Suggested category name: {category_name_candidate}")
        
        print("\n\n")
        print("-" * 80)

def main():
    find_category_candidates()
    
    print("\nTo add these categories to cate.csv, you can:")
    print("1. Manually edit cate.csv")
    print("2. Use the original script without --classify-only to interactively add categories")

if __name__ == "__main__":
    main()

