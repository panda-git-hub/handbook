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
  B->>B: ログフォーマット指定
  B->>B: ログ保存
  D->>C: 対象ログ指定
  note over A,I: フィルタ条件: Linkdownを含む
  C->>B: 対象ログ確認
  C->>D: 対象ログ取得
  D->>E: トリガー評価
  E->>F: アクション実行
  F->>G: プログラム実行
  G->>H: トラップ連携
  H->>I: アラーム生成
  I->>I: アラーム表示
```
