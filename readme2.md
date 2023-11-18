## フロー

```mermaid
sequenceDiagram
  autonumber
  Title: ワンタッチ
  participant A as 連携システム
  participant B as Fuse（JAZZ)
  participant C as DecisionManager（JAZZ)
  participant D as AnsibleTower（JAZZ)
  participant E as 担当者
  participant F as 自動化対象ホスト

  A->>B: オーダー発行
  B->>B: オーダー精査
  B->>C: オーダー転送
  C->>C: ルール判定
  C->>B: 判定結果を送信
  B->>E: ワンコードを発行
  E->>D: ワンコードを入力
  D->>F: 自動化を実行
  D->>E: 結果を表示
