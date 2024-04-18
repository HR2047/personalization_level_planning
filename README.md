# personalization_level_planning
個人化レベルのメリハリを考慮した観光ルートプランニング  
(Tailored Trip: Advanced Route Planning with Personalization Levels in Focus)

## 環境

- gurobi
- numpy
- pandas

## 仕様

#### 入力
- start_point (`int`): 出発する地点のAOI ID
- end_point (`int`): 目的地のAOI ID
- time_budget (`int`): 観光の合計時間予算，時間単位

#### 例

- 入力
``` python
# Kawaramati(88) to Kyoto tower(81), Timebudget:4h
gen_route(start_point=88, end_point=81, timebudget=4)
```

- 出力
``` python
訪問エリア数 5
Timebudget 4 h
合計旅行時間 3.990526227222222
旅程のスコア 18.453802222658997
河原町(0)→トロッコ嵐山駅(2)→二尊院(1)→祇王寺(1)→神護寺(1)→京都タワー
個人化レベル(設定値)　　 0.35607336811924306 0.383388261460441 0.26053837042031597
個人化レベル(生成ルート) 0.2 0.6 0.2
```
