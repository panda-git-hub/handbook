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

  A->>A: プロセスダウン
