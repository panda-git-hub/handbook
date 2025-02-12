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


