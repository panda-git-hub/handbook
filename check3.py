import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_categories(file_path):
    """カテゴリ定義ファイルを読み込む関数"""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, delimiter='\t', header=None, names=['log', 'category'])
            print(f"{len(df)}件のカテゴリ定義を読み込みました")
            return df
        else:
            print(f"ファイル {file_path} が見つかりません。新規作成します。")
            return pd.DataFrame(columns=['log', 'category'])
    except Exception as e:
        print(f"カテゴリ定義ファイルの読み込み中にエラーが発生しました: {e}")
        return pd.DataFrame(columns=['log', 'category'])

def load_logs(file_path):
    """ログファイルを読み込む関数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logs = [line.strip() for line in f if line.strip()]
        print(f"{len(logs)}件のログを読み込みました")
        return logs
    except Exception as e:
        print(f"ログファイルの読み込み中にエラーが発生しました: {e}")
        return []

def train_model(categories_df):
    """カテゴリデータからモデルを訓練する関数"""
    # カテゴリが1つの場合の特別処理
    if len(categories_df) == 1 or len(categories_df['category'].unique()) == 1:
        print("カテゴリが1つだけ定義されています。類似度に基づく分類を行います。")
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
        X = vectorizer.fit_transform(categories_df['log'])
        # 1つしかないカテゴリ名を取得
        single_category = categories_df['category'].iloc[0]
        # モデルの代わりにカテゴリ名と特徴ベクトルを返す
        return {"type": "single", "category": single_category, "vectors": X}, vectorizer
        
    elif len(categories_df) < 2:
        print("訓練データが不足しています。少なくとも1つのカテゴリが必要です。")
        return None, None
    
    # 複数カテゴリがある場合は通常の分類モデルを訓練
    # 特徴量抽出器を作成
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
    
    # 特徴量とラベルを作成
    X = vectorizer.fit_transform(categories_df['log'])
    y = categories_df['category']
    
    # モデルの作成と訓練
    model = MultinomialNB()
    
    # データが少ない場合はトレーニングデータのみを使用
    if len(categories_df) < 10:
        model.fit(X, y)
        print("モデルを訓練しました（テストデータなし）")
        return {"type": "multi", "model": model}, vectorizer
    
    # データが十分にある場合はテストデータも使用
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # モデル評価
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"モデルの精度: {accuracy:.2f}")
    
    return {"type": "multi", "model": model}, vectorizer

def classify_logs(logs, model_data, vectorizer, categories_df):
    """ログを分類する関数"""
    result = {}
    
    # モデルが無い場合は全て未分類とする
    if model_data is None or vectorizer is None:
        print("モデルがトレーニングされていないため、全てのログを未分類とします。")
        return {log: "未分類" for log in logs}, logs
    
    unknown_logs = []
    
    # 単一カテゴリの場合：特定のパターンのみを同じカテゴリに分類する
    if model_data["type"] == "single":
        single_category = model_data["category"]
        category_vectors = model_data["vectors"]
        
        # トレーニングログから主要パターンを抽出
        training_patterns = []
        for _, row in categories_df.iterrows():
            # ログの主要部分を抽出
            parts = row['log'].split(' [')
            if len(parts) > 1:
                log_type = parts[1].split(']')[0]  # ERROR, INFO などの種類
                msg_parts = row['log'].split('] ')[1:]
                if msg_parts:
                    main_msg = msg_parts[0].split(':')[0]  # メインメッセージ部分
                    training_patterns.append((log_type, main_msg))
        
        for log in logs:
            # 完全一致するログを確認
            exact_match = categories_df[categories_df['log'] == log]
            if not exact_match.empty:
                result[log] = exact_match.iloc[0]['category']
                continue
            
            # ログからパターンを抽出
            is_similar = False
            parts = log.split(' [')
            if len(parts) > 1:
                log_type = parts[1].split(']')[0]  # ERROR, INFO などの種類
                msg_parts = log.split('] ')[1:]
                if msg_parts:
                    main_msg = msg_parts[0].split(':')[0]  # メインメッセージ部分
                    
                    # トレーニングデータのパターンとマッチするか確認
                    for train_type, train_msg in training_patterns:
                        if log_type == train_type and main_msg == train_msg:
                            result[log] = single_category
                            is_similar = True
                            break
            
            if is_similar:
                continue
                
            # 類似度に基づく分類（より厳格な閾値）
            log_vector = vectorizer.transform([log])
            similarities = cosine_similarity(log_vector, category_vectors)
            max_similarity = np.max(similarities)
            
            if max_similarity > 0.8:  # 閾値を高く設定
                result[log] = single_category
            else:
                unknown_logs.append(log)
                result[log] = "未分類"
    else:
        # 複数カテゴリの場合
        model = model_data["model"]
        
        # トレーニングログから主要パターンを抽出
        pattern_dict = {}
        for _, row in categories_df.iterrows():
            parts = row['log'].split(' [')
            if len(parts) > 1:
                log_type = parts[1].split(']')[0]
                msg_parts = row['log'].split('] ')[1:]
                if msg_parts:
                    main_msg = msg_parts[0].split(':')[0]
                    pattern_dict[(log_type, main_msg)] = row['category']
        
        for log in logs:
            # 完全一致するログを確認
            exact_match = categories_df[categories_df['log'] == log]
            if not exact_match.empty:
                result[log] = exact_match.iloc[0]['category']
                continue
            
            # パターンマッチを試みる
            matched = False
            parts = log.split(' [')
            if len(parts) > 1:
                log_type = parts[1].split(']')[0]
                msg_parts = log.split('] ')[1:]
                if msg_parts:
                    main_msg = msg_parts[0].split(':')[0]
                    if (log_type, main_msg) in pattern_dict:
                        result[log] = pattern_dict[(log_type, main_msg)]
                        matched = True
            
            if matched:
                continue
            
            # モデルによる予測
            log_vector = vectorizer.transform([log])
            predicted_category = model.predict(log_vector)[0]
            
            # 予測確率を取得
            prediction_proba = model.predict_proba(log_vector)[0]
            max_proba = np.max(prediction_proba)
            
            # 確信度が高い場合のみ分類
            if max_proba > 0.7:  # 閾値を高く設定
                result[log] = predicted_category
            else:
                unknown_logs.append(log)
                result[log] = "未分類"
    
    return result, unknown_logs

def define_new_categories(unknown_logs, categories_df):
    """未分類のログに対して新しいカテゴリを定義する関数"""
    if not unknown_logs:
        print("未分類のログはありません。")
        return categories_df, False
    
    print(f"\n{len(unknown_logs)}件の未分類ログがあります。")
    
    for i, log in enumerate(unknown_logs):
        print(f"\n未分類ログ {i+1}/{len(unknown_logs)}: {log}")
        
        # 既存のカテゴリを表示
        existing_categories = categories_df['category'].unique()
        if len(existing_categories) > 0:
            print("\n既存のカテゴリ:")
            for j, cat in enumerate(existing_categories):
                print(f"{j+1}. {cat}")
            
            # ユーザーにカテゴリを選択または新規作成してもらう
            print("\n選択肢:")
            print("1-N: 既存のカテゴリを選択")
            print("N: 新しいカテゴリを作成")
            print("S: スキップ")
            print("Q: 終了")  # 終了オプションを追加
        else:
            print("\n既存のカテゴリがありません。")
            print("\n選択肢:")
            print("N: 新しいカテゴリを作成")
            print("S: スキップ")
            print("Q: 終了")  # 終了オプションを追加
        
        choice = input("選択してください: ").strip()
        
        # 終了オプションのチェック
        if choice.upper() == 'Q':
            print("カテゴリ定義を終了します。")
            # 現在のカテゴリ定義を返して終了
            return categories_df, True
        
        if choice.upper() == 'S':
            continue
        
        try:
            # 既存カテゴリの選択（カテゴリが存在する場合）
            if len(existing_categories) > 0:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(existing_categories):
                    selected_category = existing_categories[choice_idx]
                    categories_df = pd.concat([categories_df, pd.DataFrame({'log': [log], 'category': [selected_category]})], ignore_index=True)
                    print(f"ログを '{selected_category}' カテゴリに追加しました。")
                    continue
        except ValueError:
            pass  # 数値変換できない場合は新カテゴリ作成に進む
        
        # 新カテゴリの作成
        new_category = input("新しいカテゴリ名を入力してください: ").strip()
        if new_category:
            categories_df = pd.concat([categories_df, pd.DataFrame({'log': [log], 'category': [new_category]})], ignore_index=True)
            print(f"ログを新しいカテゴリ '{new_category}' に追加しました。")
    
    # 通常終了（すべてのログを処理）
    return categories_df, False

def save_categories(categories_df, file_path):
    """カテゴリ定義をファイルに保存する関数"""
    try:
        categories_df.to_csv(file_path, sep='\t', header=False, index=False)
        print(f"カテゴリ定義を {file_path} に保存しました。")
    except Exception as e:
        print(f"カテゴリ定義の保存中にエラーが発生しました: {e}")

def main():
    """メイン関数"""
    # ファイルパスの設定
    categories_file = "cate.csv"
    logs_file = "analy.log"
    
    # カテゴリ定義の読み込み
    categories_df = load_categories(categories_file)
    
    # ログファイルの読み込み
    logs = load_logs(logs_file)
    if not logs:
        print("分析するログがありません。")
        return
    
    # カテゴリが存在しない場合は、全てのログを未分類として処理
    if len(categories_df) < 1:
        print("カテゴリ定義がありません。全てのログを未分類とします。")
        model_data, vectorizer = None, None
    else:
        # モデルのトレーニング（カテゴリが1つでもトレーニング可能）
        model_data, vectorizer = train_model(categories_df)
    
    # ログの分類
    classification_result, unknown_logs = classify_logs(logs, model_data, vectorizer, categories_df)
    
    # 分類結果の表示
    print("\n分類結果:")
    for log, category in classification_result.items():
        print(f"ログ: {log} => カテゴリ: {category}")
    
    # 未分類ログへの対応
    if unknown_logs:
        proceed = input("\n未分類のログに対して新しいカテゴリを定義しますか？ (Y/N): ").strip().upper()
        if proceed == 'Y':
            # define_new_categoriesから終了フラグも受け取る
            categories_df, should_exit = define_new_categories(unknown_logs, categories_df)
            
            # ユーザーが終了を選択した場合
            if should_exit:
                save_categories(categories_df, categories_file)
                print("\nプログラムを終了します。")
                return
            
            save_categories(categories_df, categories_file)
            
            # 新しいカテゴリでモデルを再トレーニング（カテゴリが少なくとも1つある場合）
            if len(categories_df) >= 1:
                print("\n新しいカテゴリでモデルを再トレーニングします。")
                model_data, vectorizer = train_model(categories_df)
                
                # 再分類
                if model_data is not None and vectorizer is not None:
                    classification_result, new_unknown_logs = classify_logs(logs, model_data, vectorizer, categories_df)
                    
                    print("\n再分類結果:")
                    for log, category in classification_result.items():
                        print(f"ログ: {log} => カテゴリ: {category}")
    
    print("\nログ分類が完了しました。")

if __name__ == "__main__":
    main()
