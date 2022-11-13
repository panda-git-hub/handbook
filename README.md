# 概要

|項目|内容|
|:---|:---|
|用途|ゼロタッチオペレーション、プロセス再起動|
|対象|Voyager、syslogサーバ、sysedgeプロセス|

<br><br>


# フロー

```mermaid
graph TD

start["スタート"]
fin["エンド"]

AA["オーダー受信"]
AB{"判定"}

BA["通知（ゼロタッチ実施）"]
BB{"判定"}

CA["プロセス確認"]
CB{"判定"}

DA["プロセス起動"]
DB{"判定"}

EA["プロセス確認"]
EB{"判定"}

FA["通知（ゼロタッチ成功）"]
FB{"判定"}


start --> AA --> AB
AB --"OK"--> BA --> BB
BB --"OK"--> CA --> CB
CB --"OK"--> DA --> DB
DB --"OK"--> EA --> EB
EB --"OK"--> FA --> FB
FB --"OK"--> fin



