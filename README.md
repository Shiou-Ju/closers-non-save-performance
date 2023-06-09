# Do baseball closers perform worse in non-save situations?
在非救援情況下登版，救援投手是否表現比較差？


## Conclusion
1. The performance is indeed slightly worse, Their performance is indeed slightly worse, registering about a 4% difference. However, this doesn't absolutely mean that closers should not be used in non-save situations. 

2. Since 2017, there's been a noticeable trend of closers increasingly being used in non-save situations, coupled with a decrease in their usage during save situations. 

3. A review of BAbip, BA, and OBP indicates an improvement in the performance of closers in non-save situations post-2017 as compared to prior years. This seems to indicate that closers have come to understand that their real value lies in crisis resolution, leading to a paradigm shift from past practices.

#### 結論
1. 表現確實較差，約有 4% 的差距，但並不會因此說絕對不能把終結者使用在非救援的情境。
2. 自 2017 以來，終結者有在非救援情況增加出賽的趨勢，並且有減少於救援情況使用的比率。
3. 從 BAbip, BA, OBP 來看，非救援情況的終結者的表現在 2017 年以後，有比過往來得好的趨勢。這似乎可以說明終結者體認到自己實際的價值在於解決危機，產生與過往不同的典範轉移。

## Problem Awareness
1. The prevailing trend involves deploying closers on the field during critical moments. Could this trend potentially influence the analytical results?

2. If this trend is indeed real, which year should serve as the benchmark or turning point?

3. It's plausible that this trend, along with the evolving ethos of baseball, has prompted pitchers to realize that their true value lies in crisis resolution during critical moments. Therefore, do they put in extra effort to excel, even in non-save situations?

#### 問題意識
1. 現在的趨勢是大家會在關鍵時刻把救援投手放上場，這個趨勢是否會影響到分析結果？
2. 如果確實有這個趨勢，那麼應該用哪一年作為分水嶺呢？
3. 或許是這個趨勢，以及棒球風氣的改變，讓投手自己也有意識到:「在關鍵時刻能化解危機，才是真正的價值」。所以他們也會在非救援情況下，盡可能表現？


### 在處理完每個投手 + 年度後，應該要確認比較的基準

什麼叫做差？

救援投手在非救援情況下，表現比較差。

但所謂的差，是跟聯盟平均比嗎？

參考以下文章

http://www.highheatstats.com/2017/06/leadoff-hitters-slugging/

裡面就是拿第一棒跟全聯盟來比。


### TODO: 需要驗證假設：這些終結者的平均救援次數在逐年下降
暫無數據


***

## 該選擇哪一年作為數據的起始點？

## 初步參考文獻
1. ERA 在不同年代的變化
   1. https://thesportjournal.org/article/eras-of-era/
2. 2006 年以後的資料，因為這之後才有 PITCHf/x 資料
   1. https://en.wikipedia.org/wiki/PITCHf/x

### 可能需要的欄位有：
1. 「登版局數」: 在哪一局開始投
2. 「登版時筆數」：只有領先三分以內才有救援成功機率
3. 「投球局數」：投了幾局，如果投超過三局，並且比賽結束，也會算是救援成功
4. 「勝敗」：如果在救援的情況上場，敗投或勝投，都算是救援失敗。
5. 「救援成功」：這算是一項獨立的記錄。
6. 「失分」：失分超過或等於三分，就不算是救援成功。
7. 「投球數」：投了幾球。

### 不知道會不會有的欄位：
1. 「連續出賽場次」
2. 「連續投球局數」
3. 「連續投球數」

***

### 我將以上需求加上以下文章整理後，請 GPT 4 為我選擇一個起始點。 

https://www.espn.com/mlb/columns/story?columnist=schwarz_alan&id=1835745

#### GPT4 回答
根據您想要進行的救援投手砸鍋分析，以下是我為您挑選出的三個時間點，以及相應的理由：

- 1990 年 - 在這一年，USA Today 開始與 STATS Inc. 合作，對棒球賽事的盒式記分表進行了重大改進。該表格包括了所有投手的四壞球、三振、留下的跑者，以及投球局數和新統計數據「保持」（Holds），這對您分析救援投手非常有幫助。此外，1990 年代之後的數據質量和完整性相對較高。

- 1984 年 - 由於 Pete Palmer 和 John Thorn 的《The Hidden Game of Baseball》的影響，紐約時報開始在每周發布一個新的統計數據：上壘率加長打率（OPS）。儘管 OPS 與您的分析目標不完全相符，但從這一年開始，棒球統計學的關注度逐漸上升，可作為您分析的起始點。

- 1977 年 - 在這一年，Bill James 首次出版了他的《Baseball Abstract》，開創了棒球統計學（Sabermetrics）的先河。從這一年開始，棒球統計學的發展和應用逐漸受到重視。雖然這一年的數據可能不如後來的年份完整，但它仍然是一個具有歷史意義的時間點，可以作為您分析的起始點。

### 最終決定選擇 1990 年
但由於 baseball-reference 的資料，我們分析了 2008 - 2022 的資料。

***

## 檔案說明

### brief-look-pybaseball-apis.ipynb
調查 pybaseball 的 API。
### non-save-situation-blow-save.ipynb
分析非救援情況下，救援投手的表現。


