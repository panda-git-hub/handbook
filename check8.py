import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
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

def load_fixed_patterns(pattern_file):
    """固定patternファイルを読み込む"""
    logger.info(f"固定patternファイル {pattern_file} を読み込んでいます")
    fixed_patterns = {}
    try:
        with open(pattern_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        cluster_name, pattern = parts
                        # cluster名からcluster番号を抽出
                        match = re.search(r'cluster (\d+)', cluster_name)
                        if match:
                            cluster_id = int(match.group(1))
                            fixed_patterns[pattern] = cluster_id
                            logger.info(f"固定pattern読み込み: cluster {cluster_id} -> {pattern[:50]}...")
        logger.info(f"{len(fixed_patterns)} 個の固定patternを読み込みました")
    except FileNotFoundError:
        logger.warning(f"固定patternファイル {pattern_file} が見つかりません。固定patternなしで処理を続行します。")
    return fixed_patterns

def extract_log_structure(log_line):
    """ログからフォーマット構造を抽出（pattern化）"""
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
    """ログの前処理とpattern抽出"""
    logger.info("ログを前処理してpatternを抽出しています")
    patterns = [extract_log_structure(log) for log in logs]
    return patterns

def match_fixed_patterns(logs, patterns, fixed_patterns):
    """固定patternにマッチするログを分類"""
    logger.info("固定patternとのマッチングを実行しています")
    
    # 初期化: 全てのログを未割り当て(-2)としてマーク
    labels = np.full(len(logs), -2, dtype=int)
    
    # 固定patternが指定されていない場合は空の配列を返す
    if not fixed_patterns:
        logger.info("固定patternがありません。すべてのログは動的clusterリングの対象になります。")
        return labels
    
    # 各ログを固定patternと照合
    matches_count = 0
    for i, (log, pattern) in enumerate(zip(logs, patterns)):
        for fixed_pattern, cluster_id in fixed_patterns.items():
            if pattern == fixed_pattern:
                labels[i] = cluster_id
                matches_count += 1
                break
    
    logger.info(f"{matches_count} ログが固定patternに一致しました")
    remaining = len(logs) - matches_count
    logger.info(f"{remaining} ログが動的clusterリングの対象になります")
    
    return labels

def vectorize_logs(patterns, indices=None):
    """TF-IDFでログpatternをベクトル化"""
    if indices is not None:
        selected_patterns = [patterns[i] for i in indices]
        logger.info(f"選択された {len(selected_patterns)} ログpatternをベクトル化しています")
    else:
        selected_patterns = patterns
        logger.info(f"すべての {len(selected_patterns)} ログpatternをベクトル化しています")
        
    vectorizer = TfidfVectorizer(
        analyzer='char', 
        ngram_range=(2, 5),  # 文字レベルのn-gramを使用
        max_features=1000,   # 特徴量の数を制限
        lowercase=True
    )
    X = vectorizer.fit_transform(selected_patterns)
    logger.info(f"特徴ベクトルの形状: {X.shape}")
    return X, vectorizer

def determine_optimal_clusters(X, max_clusters=15):
    """シルエットスコアを使用して最適なcluster数を推定"""
    logger.info("最適なcluster数を推定しています")
    silhouette_scores = []
    K = range(2, min(max_clusters, X.shape[0] // 2))
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
        logger.info(f"cluster数 {k}: シルエットスコア {score:.4f}")
    
    best_k = K[np.argmax(silhouette_scores)]
    logger.info(f"最適なcluster数: {best_k}")
    return best_k

def find_outlier_logs(X, kmeans, percentile_threshold=95):
    """cluster中心から遠いログを検出して「未分類」としてマーク"""
    logger.info(f"未分類ログを検出しています (閾値: {percentile_threshold}パーセンタイル)")
    
    # 各データポイントとそのcluster中心との距離を計算
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # 各ログのcluster中心からの距離を計算
    distances = []
    for i, label in enumerate(labels):
        point = X[i].toarray().flatten()
        center = cluster_centers[label]
        dist = np.linalg.norm(point - center)
        distances.append(dist)
    
    # 閾値を計算（パーセンタイル）
    distance_threshold = np.percentile(distances, percentile_threshold)
    logger.info(f"距離閾値: {distance_threshold:.4f}")
    
    # 距離が閾値を超えるログを「未分類」（-1）としてマーク
    new_labels = labels.copy()
    for i, dist in enumerate(distances):
        if dist > distance_threshold:
            new_labels[i] = -1
    
    # 未分類ログの数を表示
    uncategorized_count = np.sum(new_labels == -1)
    logger.info(f"{uncategorized_count} ログが未分類としてマークされました ({uncategorized_count/len(new_labels)*100:.1f}%)")
    
    return new_labels

def cluster_logs(X, indices, all_labels, fixed_cluster_ids, n_clusters=None, detect_outliers=True, outlier_percentile=95):
    """K-meansでログをclusterリングし、必要に応じて外れ値を検出"""
    if n_clusters is None:
        n_clusters = determine_optimal_clusters(X)
    
    # 既存の固定clusterIDと重複しないように調整
    used_cluster_ids = set(fixed_cluster_ids)
    
    # 固定clusterIDがある場合とない場合で処理を分ける
    if used_cluster_ids:
        max_used_id = max(used_cluster_ids)
        available_cluster_ids = [i for i in range(max_used_id + n_clusters + 1) if i not in used_cluster_ids]
    else:
        # 固定clusterIDがない場合は0から順番にclusterIDを割り当て
        available_cluster_ids = list(range(n_clusters))
    
    if len(available_cluster_ids) < n_clusters:
        logger.warning("固定clusterIDと動的clusterIDの間で競合が発生しています。固定IDが優先されます。")
    
    adjusted_n_clusters = min(n_clusters, len(available_cluster_ids))
    logger.info(f"{adjusted_n_clusters}clusterでK-meansclusterリングを実行しています")
    
    kmeans = KMeans(n_clusters=adjusted_n_clusters, random_state=42, n_init=10)
    dynamic_labels = kmeans.fit_predict(X)
    
    # 外れ値検出が有効な場合
    if detect_outliers:
        dynamic_labels = find_outlier_logs(X, kmeans, outlier_percentile)
    
    # 動的clusterIDを利用可能なIDにマッピング
    id_mapping = {}
    for i in range(adjusted_n_clusters):
        if i < len(available_cluster_ids):
            id_mapping[i] = available_cluster_ids[i]
        else:
            logger.warning(f"利用可能なclusterIDが不足しています。cluster {i} は未分類として扱われます。")
            id_mapping[i] = -1
    
    # -1（未分類）はそのまま保持
    id_mapping[-1] = -1
    
    # ラベル結果をマッピングして全体のラベル配列に統合
    for i, idx in enumerate(indices):
        label = dynamic_labels[i]
        all_labels[idx] = id_mapping.get(label, -1)
    
    return all_labels, kmeans

def display_cluster_examples(logs, patterns, labels, n_examples=3):
    """各clusterのsampleログを表示"""
    logger.info("各clusterのsampleログを表示します")
    # -1（未分類）を除いた一意のラベルを取得し、-1を最後に追加
    unique_labels = sorted([l for l in set(labels) if l != -1])
    if -1 in labels:
        unique_labels.append(-1)
    
    cluster_examples = {}
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) > 0:
            sample_indices = indices[:min(n_examples, len(indices))]
            cluster_name = "未分類" if label == -1 else f"cluster {label}"
            cluster_examples[label] = {
                "name": cluster_name,
                "count": len(indices),
                "pattern": patterns[indices[0]] if indices.size > 0 else "未検出",
                "examples": [logs[i] for i in sample_indices]
            }
    
    return cluster_examples

def print_cluster_statistics(labels):
    """clusterのサイズと分布の統計情報を表示"""
    counts = Counter(labels)
    # -1（未分類）を除外してcluster数を計算
    n_clusters = len([k for k in counts.keys() if k != -1])
    total_logs = sum(counts.values())
    
    print("\n===== cluster統計情報 =====")
    print(f"検出されたcluster数: {n_clusters}")
    print(f"ログの総数: {total_logs}")
    
    print("\nclusterサイズの分布:")
    # 未分類を最後に表示するためのソート関数
    def sort_key(item):
        cluster_id, _ = item
        return (cluster_id == -1, cluster_id)  # 未分類を最後にソート
    
    for cluster_id, count in sorted(counts.items(), key=sort_key):
        percentage = (count / total_logs) * 100
        bar_length = int(percentage / 2)
        bar = "#" * bar_length
        cluster_name = "未分類" if cluster_id == -1 else f"cluster {cluster_id}"
        print(f"{cluster_name}: {count} ログ ({percentage:.1f}%) {bar}")

def save_results(logs, patterns, labels, output_file):
    """結果をCSVファイルに保存"""
    logger.info(f"結果を {output_file} に保存しています")
    
    results = []
    for i, (log, pattern, label) in enumerate(zip(logs, patterns, labels)):
        cluster_name = "未分類" if label == -1 else f"cluster {label}"
        results.append({
            "log_id": i,
            "cluster": label,
            "cluster_name": cluster_name,
            "log": log,
            "pattern": pattern
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logger.info(f"結果を {output_file} に保存しました")

def main(log_file, output_file="log_clusters.csv", pattern_file=None, n_clusters=None, n_examples=3, 
         save_results_flag=True, detect_outliers=True, outlier_percentile=95):
    """ログ分類の主要処理フロー"""
    # ログの読み込み
    logs = load_logs(log_file)
    
    # 前処理とpattern抽出
    patterns = preprocess_logs(logs)
    
    # 固定patternの読み込み（指定されている場合）
    fixed_patterns = {}
    if pattern_file:
        fixed_patterns = load_fixed_patterns(pattern_file)
    
    # 固定patternによる初期clusterリング
    labels = match_fixed_patterns(logs, patterns, fixed_patterns)
    
    # 固定patternに一致しないログのインデックスを取得
    unmatched_indices = np.where(labels == -2)[0]
    
    # 使用済みのclusterIDを記録
    fixed_cluster_ids = set([cluster_id for cluster_id in labels if cluster_id >= 0])
    
    # 未割り当てのログが存在する場合は動的clusterリングを実行
    if len(unmatched_indices) > 0:
        logger.info(f"{len(unmatched_indices)} ログを動的にclusterリングします")
        # 未割り当てログのベクトル化
        X, vectorizer = vectorize_logs([patterns[i] for i in unmatched_indices])
        
        # 動的clusterリングを実行
        labels, kmeans = cluster_logs(X, unmatched_indices, labels, fixed_cluster_ids, 
                                      n_clusters, detect_outliers, outlier_percentile)
    else:
        logger.info("すべてのログが固定patternに一致しました。動的clusterリングはスキップします。")
    
    # clusterの統計情報を表示
    print_cluster_statistics(labels)
    
    # 各clusterのsample表示
    cluster_examples = display_cluster_examples(logs, patterns, labels, n_examples)
    
    # clusterの詳細を表示（未分類を最後に）
    print("\n===== cluster分析結果 =====")
    
    # ソート関数: 未分類を最後に表示
    def sort_key(item):
        label, _ = item
        return (label == -1, label)
    
    for label, data in sorted(cluster_examples.items(), key=sort_key):
        cluster_name = data['name']
        print(f"\n{cluster_name} ({data['count']} ログ):")
        print(f"pattern: {data['pattern']}")
        print("sample:")
        for i, example in enumerate(data['examples'], 1):
            print(f"  {i}. {example}")
        print()
        print()
        print()
        print()
    
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
    parser.add_argument("--pattern", "-p", default=None, help="固定patternを含むファイルのパス")
    parser.add_argument("--clusters", "-c", type=int, default=None, help="cluster数（指定しない場合は自動推定）")
    parser.add_argument("--examples", "-e", type=int, default=3, help="各clusterから表示するsample数")
    parser.add_argument("--no-save", action="store_true", help="結果をCSVファイルに保存しない")
    parser.add_argument("--no-outliers", action="store_true", help="未分類ログの検出を無効にする")
    parser.add_argument("--outlier-percentile", type=float, default=95, 
                        help="未分類ログ検出のパーセンタイル閾値 (0-100)")
   
    args = parser.parse_args()
    main(args.log_file, args.output, args.pattern, args.clusters, args.examples, 
         not args.no_save, not args.no_outliers, args.outlier_percentile)

#
# python3 check7.py log.txt <options>
#
# --pattern <FILENAME>
# --output <FILENAME>
# --no-save
# --clusters <NUM>
# --examples <NUM>
# --outlier-percentile <NUM>
# --no-outliers
#



