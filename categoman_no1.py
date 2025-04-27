#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# コマンドライン引数の処理
if len(sys.argv) > 1 and sys.argv[1] == "--classify-only":
    classify_only = True
else:
    classify_only = False

# パターン抽出用の正規表現
ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
date_pattern1 = re.compile(r'\d{4}[/-]\d{2}[/-]\d{2}\s\d{2}:\d{2}:\d{2}')
date_pattern2 = re.compile(r'\d{2}[/-]\w{3}[/-]\d{4}:\d{2}:\d{2}:\d{2}')
num_pattern = re.compile(r'\b\d+\b')
url_path_pattern = re.compile(r'/[a-zA-Z0-9_/.-]+')
hostname_pattern = re.compile(r'\b[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+\b')

def preprocess_log(log_line):
    """ログを前処理してパターン抽出する"""
    processed = log_line
    
    # 日時表記の置換
    processed = date_pattern1.sub("DATETIME", processed)
    processed = date_pattern2.sub("DATETIME", processed)
    
    # IPアドレスの置換
    processed = ip_pattern.sub("IP_ADDR", processed)
    
    # URLパスの置換
    processed = url_path_pattern.sub("/PATH", processed)
    
    # ホスト名の置換
    processed = hostname_pattern.sub("HOSTNAME", processed)
    
    # 数値の置換 (他のパターン変換後に行う)
    processed = num_pattern.sub("NUM", processed)
    
    return processed

def load_categories():
    """カテゴリ定義をロードする"""
    if not os.path.exists('cate.csv') or os.path.getsize('cate.csv') == 0:
        return pd.DataFrame(columns=['category', 'pattern'])
    
    return pd.read_csv('cate.csv', sep='\t', names=['category', 'pattern'])

def save_category(category, pattern):
    """新しいカテゴリ定義を保存する"""
    with open('cate.csv', 'a', encoding='utf-8') as f:
        f.write(f"{category}\t{pattern}\n")

def classify_log(log_line, processed_line, categories):
    """ログをカテゴリに分類する"""
    if categories.empty:
        return "undefined"
    
    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # 1-gramと2-gramの特徴を使用
    patterns = categories['pattern'].tolist()
    patterns.append(processed_line)
    
    # ベクトル化できない場合はundefinedを返す
    try:
        tfidf_matrix = vectorizer.fit_transform(patterns)
    except:
        return "undefined"
    
    # 最後の行（現在のログ）と他のパターンとの類似度を計算
    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
    
    # 最も類似度の高いカテゴリを選択（閾値を0.8に上げて厳しくする）
    if len(similarities) > 0 and np.max(similarities) > 0.5:  # より厳しい類似度の閾値
        max_idx = np.argmax(similarities)
        return categories.iloc[max_idx]['category']
    else:
        return "undefined"

def main():
    # カテゴリ定義のロード
    categories = load_categories()
    
    # 処理結果を保存するリスト
    results = []
    undefined_logs = []
    
    # ログファイルの処理
    with open('analy.log', 'r', encoding='utf-8') as log_file:
        for line_num, log_line in enumerate(log_file):
            log_line = log_line.strip()
            if not log_line:
                continue
                
            processed_line = preprocess_log(log_line)
            category = classify_log(log_line, processed_line, categories)
            
            result = f"{category}\t{log_line}\t{processed_line}"
            results.append((category, log_line, processed_line))
            
            if not classify_only:
                print(result)
            elif category != "undefined":
                print(result)
            
            if category == "undefined":
                undefined_logs.append((line_num, log_line, processed_line))
    
    # 分類のみのモードなら終了
    if classify_only:
        return
    
    # undefinedのログに対する処理
    if undefined_logs:
        print("\n-- undefined logs --")
        
        for idx, (line_num, log_line, processed_line) in enumerate(undefined_logs):
            print(f"\n[{idx+1}/{len(undefined_logs)}] Line {line_num+1}: {log_line}")
            print(f"Processed: {processed_line}")
            
            while True:
                choice = input("1.setting  2.skip  3.quit: ")
                
                if choice == "1":
                    category = input("Enter category name: ").strip()
                    if category:
                        save_category(category, processed_line)
                        print(f"Saved category '{category}' for this log pattern.")
                        break
                    else:
                        print("Category name cannot be empty.")
                elif choice == "2":
                    print("Skipped.")
                    break
                elif choice == "3":
                    print("Quit processing undefined logs.")
                    return
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()

#
# python3 check.py
# python3 check.py --classify-only
