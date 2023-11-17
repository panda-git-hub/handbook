## フロー

```mermaid
sequenceDiagram
  autonumber
  Title: ワンタッチ
  participant A as SRV
  participant B as NW(m)
  participant C as NW(b)
  participant D as FM
  participant E as OC
  participant F as FU
  participant G as DM
  participant H as AT(WF1)
  participant I as AT(WF2)
  participant J as Human

  A->>A: プロセスダウン
  A->>D: アラート発生
  D->>D: アラーム生成
  D->>E: アラーム表示
  E->>J: アラーム目視
  D->>F: アラーム転送
  F->>G: ルール判定依頼
  G->>F: ルール判定
  F->>H: ワークフロー実行
  H->>D: ワンタッチ通知
  D->>E: 通知表示
  E->>J: 通知目視
  H->>A: 状態確認
  H->>D: ワンタッチ情報通知
  D->>E: ワンタッチ情報表示
  E->>J: ワンタッチ情報目視
  J->>I: ワンタッチ実行
  I->>A: 一次対応実施
  I->>D: 一次対応結果通知
  D->>E: 一次対応結果表示
  E->>J: 結果確認
