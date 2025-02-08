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
  participant G as voyager<br>SS
  participant H as voyager<br>OC
  A->>B: ログ送信
  B->>B: ログフォーマットの指定
  B->>B: ログ保存
```
