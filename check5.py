import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re
import argparse
import logging
from collections import Counter

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_logs(file_path):
    """ログファイルを読み込む"""
    logger.info(f"ログファイル {file_path} を読み込んでいます")
    with open(file_path, 'r', encoding='utf-8') as f:
        logs = [line.strip() for line in f if line.strip()]
    logger.info(f"{len(logs)} 行のログを読み込みました")
    return logs

def extract_log_structure(log_line):
    """ログからフォーマット構造を抽出（パターン化）"""
    # 数値を一般化
    pattern = re.sub(r'\b\d+\b', 'NUM', log_line)
    # IPアドレスを一般化
    pattern = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP_ADDR', pattern)
    # メールアドレスを一般化
    pattern = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL', pattern)
    # 日付形式を一般化 (複数フォーマット対応)
    pattern = re.sub(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:,\d{3})?', 'DATETIME', pattern)
    pattern = re.sub(r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}', 'DATETIME', pattern)
    pattern = re.sub(r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4}', 'DATETIME', pattern)
    # HEXを一般化
    pattern = re.sub(r'0x[0-9a-fA-F]+', 'HEX', pattern)
    # UUIDを一般化
    pattern = re.sub(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', 'UUID', pattern)
    # URLパスを一般化
    pattern = re.sub(r'/[a-zA-Z0-9_\-/.]+', '/PATH', pattern)
    
    return pattern

def preprocess_logs(logs):
    """ログの前処理とパターン抽出"""
    logger.info("ログを前処理してパターンを抽出しています")
    patterns = [extract_log_structure(log) for log in logs]
    return patterns

def vectorize_logs(patterns):
    """TF-IDFでログパターンをベクトル化"""
    logger.info("ログパターンをベクトル化しています")
    vectorizer = TfidfVectorizer(
        analyzer='char', 
        ngram_range=(2, 5),  # 文字レベルのn-gramを使用
        max_features=1000,   # 特徴量の数を制限
        lowercase=True
    )
    X = vectorizer.fit_transform(patterns)
    logger.info(f"特徴ベクトルの形状: {X.shape}")
    return X, vectorizer

def determine_optimal_clusters(X, max_clusters=15):
    """シルエットスコアを使用して最適なクラスタ数を推定"""
    logger.info("最適なクラスタ数を推定しています")
    silhouette_scores = []
    K = range(2, min(max_clusters, X.shape[0] // 2))
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
        logger.info(f"クラスタ数 {k}: シルエットスコア {score:.4f}")
    
    best_k = K[np.argmax(silhouette_scores)]
    logger.info(f"最適なクラスタ数: {best_k}")
    return best_k

def cluster_logs(X, n_clusters=None):
    """K-meansでログをクラスタリング"""
    if n_clusters is None:
        n_clusters = determine_optimal_clusters(X)
    
    logger.info(f"{n_clusters}クラスタでK-meansクラスタリングを実行しています")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # クラスタごとのサンプル数を表示
    counts = Counter(labels)
    for cluster_id, count in sorted(counts.items()):
        logger.info(f"クラスタ {cluster_id}: {count} サンプル")
    
    return labels

def display_cluster_examples(logs, patterns, labels, n_examples=3):
    """各クラスタのサンプルログを表示"""
    logger.info("各クラスタのサンプルログを表示します")
    unique_labels = sorted(set(labels))
    
    cluster_examples = {}
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) > 0:
            sample_indices = indices[:min(n_examples, len(indices))]
            cluster_examples[label] = {
                "count": len(indices),
                "pattern": patterns[indices[0]],
                "examples": [logs[i] for i in sample_indices]
            }
    
    return cluster_examples

def print_cluster_statistics(labels):
    """クラスタのサイズと分布の統計情報を表示"""
    counts = Counter(labels)
    n_clusters = len(counts)
    total_logs = sum(counts.values())
    
    print("\n===== クラスタ統計情報 =====")
    print(f"検出されたクラスタ数: {n_clusters}")
    print(f"ログの総数: {total_logs}")
    
    print("\nクラスタサイズの分布:")
    for cluster_id, count in sorted(counts.items()):
        percentage = (count / total_logs) * 100
        bar_length = int(percentage / 2)
        bar = "#" * bar_length
        print(f"クラスタ {cluster_id}: {count} ログ ({percentage:.1f}%) {bar}")

def save_results(logs, patterns, labels, output_file):
    """結果をCSVファイルに保存"""
    logger.info(f"結果を {output_file} に保存しています")
    
    results = []
    for i, (log, pattern, label) in enumerate(zip(logs, patterns, labels)):
        results.append({
            "log_id": i,
            "cluster": label,
            "log": log,
            "pattern": pattern
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logger.info(f"結果を {output_file} に保存しました")

def main(log_file, output_file="log_clusters.csv", n_clusters=None, n_examples=3, save_results_flag=True):
    """ログ分類の主要処理フロー"""
    # ログの読み込み
    logs = load_logs(log_file)
    
    # 前処理とパターン抽出
    patterns = preprocess_logs(logs)
    
    # ベクトル化
    X, vectorizer = vectorize_logs(patterns)
    
    # クラスタリング
    labels = cluster_logs(X, n_clusters)
    
    # クラスタの統計情報を表示
    print_cluster_statistics(labels)
    
    # 各クラスタのサンプル表示
    cluster_examples = display_cluster_examples(logs, patterns, labels, n_examples)
    
    # クラスタの詳細を表示
    print("\n===== クラスタ分析結果 =====")
    for label, data in sorted(cluster_examples.items()):
        print(f"\nクラスタ {label} ({data['count']} ログ):")
        print(f"パターン: {data['pattern']}")
        print("サンプル:")
        for i, example in enumerate(data['examples'], 1):
            print(f"  {i}. {example}")
    
    # 結果の保存（オプション）
    if save_results_flag:
        save_results(logs, patterns, labels, output_file)
        logger.info(f"結果を {output_file} に保存しました")
        print(f"\n分析結果は {output_file} に保存されました")
    else:
        logger.info("結果の保存はスキップされました")
        print("\n結果の保存はスキップされました")
    
    logger.info("解析完了!")
    return cluster_examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ログフォーマットの自動分類ツール")
    parser.add_argument("log_file", help="分析するログファイルのパス")
    parser.add_argument("--output", "-o", default="log_clusters.csv", help="結果出力先CSVファイル")
    parser.add_argument("--clusters", "-c", type=int, default=None, help="クラスタ数（指定しない場合は自動推定）")
    parser.add_argument("--examples", "-e", type=int, default=3, help="各クラスタから表示するサンプル数")
    parser.add_argument("--no-save", action="store_true", help="結果をCSVファイルに保存しない")
    
    args = parser.parse_args()
    main(args.log_file, args.output, args.clusters, args.examples, not args.no_save)


# ex
# python3 check.py logs.txt --clusters 5 --examples 5 --no-save
#
