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
finA["エンド（成功）"]
finB["エンド（失敗）"]

AA["オーダー受信"]
AB{"判定"}
AC["失敗"]

BA["通知（ゼロタッチ実施）"]
BB{"判定"}
BC["失敗"]

CA["プロセス確認"]
CB{"判定"}
CC["失敗"]

DA["プロセス起動"]
DB{"判定"}
DC["失敗"]

EA["プロセス確認"]
EB{"判定"}
EC["失敗"]

FA["通知（ゼロタッチ成功）"]
FB{"判定"}
FC["失敗"]


start --> AA --> AB
AB --"OK"--> BA --> BB
AB --"NG"--> AC --> finB
BB --"OK"--> CA --> CB
BB --"NG"--> BC --> finB
CB --"OK"--> DA --> DB
CB --"NG"--> CC --> finB
DB --"OK"--> EA --> EB
DB --"NG"--> DC --> finB
EB --"OK"--> FA --> FB
EB --"NG"--> EC --> finB
FB --"OK"--> finA
FB --"NG"--> FC --> finB



