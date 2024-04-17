# NCKU-AICUP2023-TEAM-3575
## 主旨
AI CUP 2023 教育部全國大專校院人工智慧競賽 真相只有一個: 事實文字檢索與查核競賽

內容主要是在做NLP繁體中文的假訊息偵測，要辨別一句話是真的(Supports)、 假的(Refutes)、資訊不足(Not Enough Information )， 通過主辦方所提供的訓練集和測試集進而對wiki網路上的資料篩選出證據性足夠的句子幫助辨識資料真實性，先對本身句子和Wiki文章算 Cosine Similarity(IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese )  取相關度高的Top10 Predicted Page ， 再通過Predicted Page中的各句子，使用BERT(hfl/chinese-bert-wwm-ext )模型尋找出真正對要預測的句子有用的Top 5 Evidence 句子，最後透過這些證據句再使用BERT進行分類問題，判斷該句話是三分類的哪一類結果。

## 運行流程

### PART 1. Document retrieval 
 將主辦方所提供的訓練集我們檔名稱作 "public_train_all.jsonl" 輸入
HanLP 模型將 Claim 切成多個詞語，接著再交給 SimCSE 模型將所有切出的詞
語和原句子去做 "CosineSimilarity" 計算相似度分數，然後進行排序由高到低，再把詞語前十高的當作該句子之預選 predicted pages，最後得到 
"train_doc10_all_method.jsonl" 這個檔案。
緊接著也會將測試資料集和上述做一樣的事情，將提供的測試集名稱為 
"public_private_combine_test_data.jsonl" 檔案丟入模型得到 
"combine_test_doc10_all.jsonl檔案，part1到此結束，會得到train和test每筆句子的的top10 Predicted pages。

### PART 2. Sentence retrieval 
首先會先將 "train_doc10_all_method.jsonl" 檔案 split 成 Train 和 
Validation 資料比例分別是8:2，接著會把 Train 資料集的每個 predicted_
pages 丟入 wiki 生成的 mapping 函式去和 ground truth 比較，如果相等就
將該句當作正相關資料去訓練，剩下再從 mapping 函式通 negative ratio 去
random 選取資料當作負相關資料，將資料丟進 chinese-bert-wwm-ext 做訓
練，取 F1-score 最好的 checkpoint 檔，此訓練模型的功用為預測資料集取
top_N evidences，接著通過訓練出的模型預測 Train、Validation、Test 
data 的 top5 evidences，再來將這些資料的 evidences 通過我們設計的 
evidence reranking 學習機制，得到"train_doc10sent5_all_rerank_ext.jsonl"、"dev_doc10sent5_all_rerank_ext.jsonl "、"combine_test_doc10sent5
_rerank_ext.jsonl" 三個檔案，part2結束，會得到rerank後每筆句子的top5 evidences 。

### PART 3. Claim verification
我們使用prompt-based learning，並讓模型訓練mask predict，並計算[Mask] 
的 crossentropy loss，將 "train_doc10sent5_all_rerank_ext.jsonl "、 
"dev_doc10sent5_all_rerank_ext.jsonl" 檔案丟入chinese-bert-wwm-ext做
訓練，但值得注意的是，在此步驟我們只會將每句話中predicted_evidence的
前三個證據丟入模型訓練，這是因為part2做了rerank後取越前面分數高的證據
能讓模型更有效率的進行訓練，訓練結束取val-acc最好的checkpoint檔，用此
模型去對測試集"combine_test_doc10sent5_rerank_ext.jsonl" 檔案進行mask 
predict，得到最後的submission.jsonl檔案，獲得最終每筆資料的三分類結果，程式結束。
## 運行環境
### 使用 Anaconda 建立環境
```bash
conda create -n Contest python=3.8
```
### python 套件
```bash
pip install -r requirements.txt
```
## Checkpoint
|        Class         | Description                          
| :------------------: | :----------------------------------- 
|     `Part2`          | [BERT label predict](https://drive.google.com/file/d/14g1haykEVEspO62SMqWVHZg6Ua-UY7gG/view?usp=drive_link)<br> [Evidence Reranking](https://drive.google.com/file/d/1w_xZEenaYmJdaBJzvu8RhszejcRJjcwO/view?usp=drive_link) 
|     `Part3`          | [Mask predict](https://drive.google.com/file/d/1mT_6Y2Bplq0ullOx8NgMXlAdSUXXM0-G/view?usp=drive_link)

## 各 part 分數
### part1
origin
|                        | Top1    |  Top2    | Top3    | Top4    | Top5  |
| :------------------: | :------------------ |------------------ |------------------ |------------------ |------------------ |
|          `Precision`   |  0.247 |0.238 |0.214 |0.197 |0.178|
|           `Recall`     |  0.224 |0.433 | 0.584 | 0.713 |0.807 |

Adaptive Cosine Similarity
|                        | Top1    |  Top2    | Top3    | Top4    | Top5  | Top6  | Top7  | Top8  | Top9  | Top10 | 
| :------------------: | :------------------ |------------------ |------------------ |------------------ |------------------ |------------------ |------------------ |------------------ |------------------ |------------------ |
|          `Precision`   | 0.698   |	0.428   |	0.309 |0.241	|0.197	|0.167	|0.145	|0.128	|0.114	| 0.104 |
|           `Recall`     | 0.653   |	0.791	| 0.847	  |0.875	|0.892	|0.905	|0.913	|0.920	|0.926	| 0.931 |

### part2
`Train data`
|              Precision          | Top1    |  Top2    | Top3    | Top4    | Top5  |
| :------------------: | :------------------ |------------------ |------------------ |------------------ |------------------ |
|          `origin`   | 0.783	|0.577	|0.441	|0.353	|0.293|
|           `rerank`     | 0.903	|0.632	|0.461	|0.358	|0.293|

|              Recall          | Top1    |  Top2    | Top3    | Top4    | Top5  |
| :------------------: | :------------------ |------------------ |------------------ |------------------ |------------------ |
|          `origin`   | 0.586	| 0.774	| 0.846| 	0.876| 	0.891| 
|           `rerank`     | 0.689	|0.851	|0.885	|0.891|	0.891|

### part3
`Train data Accuracy:`
|                       | origin    |  Prompt    | Rerank & Prompt    |
| :------------------: | :------------------ |------------------ |------------------ |
|          `Val_acc`   | 0.4345	|0.7146	|0.7223	|
|           `loss`     | 1.2353	|0.8876	|0.9030	|

## 檔案說明
我們將 public_train.jsonl 跟 public_train_0522.jsonl 合併成 `public_train_all.jsonl`
以及public_test.jsonl 跟 private_test_data.jsonl 合併成 `public_private_combine_test_data.jsonl`
<br>而有兩個檔案太大[train_doc5_all_page_other_predictor.jsonl](https://drive.google.com/file/d/17171E3S6HRH9tFwtnhItrS-Vhms7Ce9O/view?usp=drive_link)以及[combine_test_doc5_all_page_with_sentences.jsonl](https://drive.google.com/file/d/1Jrs2aQVZX4JVqGHpfMGzmUIq7BSPkYuj/view?usp=drive_link)請自行下載
* **`hanlp_con_results.pkl:`** 為 public_train_all.jsonl 的 hanlp_results

* **`hanlp_con_combine_test_results_imp.pkl:`** 為 public_private_combine_test_data.jsonl 的 hanlp_results
* **`train_doc5_all_page_other_predictor.jsonl:`** 為包含所有 wikipedia search 的結果以及其對應的內容的 train data
* **`combine_test_doc5_all_page_with_sentences.jsonl:`** 為包含所有 wikipedia search 的結果以及其對應的內容的 test data
* **`train_doc10_all_method.jsonl:`** 為Part1 public_train_all.jsonl 的經過 Adaptive Cosine Similarity取相似度前10的predicted page
* **`combine_test_doc10_all.jsonl:`** 為Part1 public_private_combine_test_data.jsonl 的經過 Adaptive Cosine Similarity取相似度前10的predicted page
* **`train_doc10sent5_all_ext.jsonl:`** 為Part2 將 train_doc10_all_method.jsonl 經過 BERT label predict 後取前五個 evidence的結果
* **`train_doc10sent5_all_rerank_ext.jsonl:`** 為 train_doc10sent5_all_ext.jsonl 經過 Evidence Reranking 的結果
* **`combine_test_doc10sent5_ext.jsonl:`** 為Part2 將 combine_test_doc10_all.jsonl 經過 BERT label predict 後取前五個 evidence的結果
* **`combine_test_doc10sent5_rerank_ext.jsonl:`** 為 combine_test_doc10sent5_ext.jsonl 經過 Evidence Reranking 的結果

## 檔案格式

train_doc5_all_page_other_predictor.jsonl:
```python3
{
    'id': '...',
    'label': '...',
    'claim': '...',
    'evidence': '...', 
    'hanlp_results': '...',
    'all_page': '...', //全部的wiki page
    'all_sentences': '...', //wiki page 對應的相關內容
}
```
combine_test_doc5_all_page_with_sentences.jsonl:
```python3
{
    'id': '...',
    'claim': '...',
    'hanlp_results': '...',
    'all_page': '...', //全部的wiki page
    'all_sentences': '...', //wiki page 對應的相關內容
}
```
train_doc10_all_method.jsonl:
```python3
{
    'id': '...',
    'label': '...',
    'claim': '...',
    'evidence': '...',
    'predicted_pages': '...',
}
```
combine_test_doc10_all.jsonl:
```python3
{
    'id': '...',
    'claim': '...',
    'predicted_pages': '...',
}
```
train_doc10sent5_all_ext.jsonl:
```python3
{
    'id': '...',
    'label': '...',
    'claim': '...',
    'evidence': '...',
    'predicted_pages': '...',
    "predicted_evidence": '...'
}
```
combine_test_doc10sent5_ext.jsonl:
```python3
{
    'id': '...',
    'claim': '...',
    'predicted_pages': '...',
    "predicted_evidence": '...'
}
```
train_doc10sent5_all_rerank_ext.jsonl:
```python3
{
    'id': '...',
    'label': '...',
    'claim': '...',
    'evidence': '...',
    'predicted_pages': '...',
    "predicted_evidence": '...' // 這裡是 reranking 後的結果
```
combine_test_doc10sent5_rerank_ext.jsonl:
```python3
{
    'id': '...',
    'claim': '...',
    'predicted_pages': '...',
    "predicted_evidence": '...' // 這裡是 reranking 後的結果
}
```
