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
  participant H as AT
  participant I as Human

  A->>A: プロセスダウン
  A->>D: アラート発生
  D->>D: アラーム生成
  D->>E: アラーム表示
　E->>I: アラーム目視
  D->>F: アラーム転送
