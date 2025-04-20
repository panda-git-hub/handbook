import csv
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_categories(filename='cate.csv'):
    """カテゴリ定義ファイルを読み込む"""
    categories = []
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    categories.append((row[0], row[1]))
    return categories

def save_category(category, log_content, filename='cate.csv'):
    """新しいカテゴリをファイルに保存する"""
    with open(filename, 'a', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([category, log_content])

def extract_log_content(log_line):
    """ログ行からログ内容部分を抽出する"""
    parts = log_line.split(' : ', 1)
    if len(parts) > 1:
        return parts[1].strip()
    return ""

def classify_log(log_line, categories):
    """ログを分類する"""
    if not categories:
        return "undefined"
    
    log_content = extract_log_content(log_line)
    
    # 完全一致をまず確認
    for category, content in categories:
        if log_content == content:
            return category
    
    # 類似度の計算
    if log_content:
        category_contents = [content for _, content in categories]
        
        # TF-IDFベクトル化
        vectorizer = TfidfVectorizer()
        # 文書数が1の場合エラーが出るので対策
        if len(category_contents) > 0:
            tfidf_matrix = vectorizer.fit_transform(category_contents + [log_content])
            
            # コサイン類似度の計算
            similarity_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
            
            # 最も類似度の高いカテゴリを選択（閾値: 0.5）
            if len(similarity_scores) > 0:
                max_idx = similarity_scores.argmax()
                #if similarity_scores[max_idx] > 0.5:
                if similarity_scores[max_idx] > 0.3:
                    return categories[max_idx][0]
    
    return "undefined"

def main():
    # カテゴリ定義の読み込み
    categories = load_categories()
    
    # ログファイルの読み込みと分類
    log_lines = []
    with open('analy.log', 'r', encoding='utf-8') as f:
        log_lines = f.readlines()
    
    # 分類結果の表示
    classified_logs = []
    for log_line in log_lines:
        log_line = log_line.strip()
        category = classify_log(log_line, categories)
        print(f"{category} --> {log_line}")
        classified_logs.append((log_line, category))
    
    # undefined のログに対してカテゴリ設定を促す
    undefined_logs = [(log, extract_log_content(log)) for log, category in classified_logs 
                      if category == "undefined"]
    
    if undefined_logs:
        print("\n未定義のログが見つかりました。カテゴリを設定してください。")
        
        for log, log_content in undefined_logs:
            print(f"\n{log}")
            print("1.setting  2.skip  3.quit")
            choice = input("選択してください: ")
            
            if choice == "1":
                category = input("カテゴリ名を入力してください: ")
                save_category(category, log_content)
                print(f"カテゴリ '{category}' を保存しました。")
            elif choice == "3":
                print("終了します。")
                break
            else:
                print("スキップします。")

if __name__ == "__main__":
    main()
