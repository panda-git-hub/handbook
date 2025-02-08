## 正常系フロー

```mermaid
sequenceDiagram
  autonumber
  Title: 正常処理
  participant A as Ansible
  participant B as Target node
  A->>A: プレイブック実行
  A->>+B: sshログイン
  B->>B: コマンドを実行
  B->>-A: コマンドの標準出力を取得
  A->>A: 保存した結果をファイルに出力
  Note right of A: テキストファイルで保存
  A->>A: レポートシェルを実行
 ```


## 設定登録

```mermaid
sequenceDiagram
  autonumber
  participant A as 監視ホスト
  participant B as rsyslog
  participant C as zabbix<br>agent
  participant D as zabbix<br>アイテム
  participant E as zabbix<br>トリガー
  participant F as zabbix<br>アクション
  participant G as zabbix<br>外部プログラム
  participant H as 監視システム<br>xxxxx
  participant I as 監視システム<br>xxxxx
  A->>B: ログ送信
  B->>B: ログフォーマットの指定
  B->>B: ログ保存
  D->>C: 対象ログの指定
  C->>B: 対象ログの確認
  C->>D: 対象ログの取得
  D->>E: トリガー条件の評価
  E->>F: アクションの実行
  F->>G: プログラムの実行
```
