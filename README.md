# TabNet: Attentive Interpretable Tabular Learning

## What is TabNet
- DNN과 DT-based 모델의 장점을 계승한 딥러닝 모델
- 기존의 기계 학습 모델보다 좋은 성능을 보인다고 하기에는 어려움이 있지만, 성능보다는 딥러닝의 장점이 필요하거나 앙상블을 구성할때 사용하는 것을 추천

### Introduce
- Sparse feature selection을 통해 핵심 feature에 집중하고 interpretability 확보
- 다수의 정형 데이터셋에서 안정적으로 높은 성능을 보이며 self-supervised learning으로 레이블이 모자란 데이터 셋에서 성능을 크게 향상시킬 수 있었다
- 특징
  1. 전처리가 필요 없고 최적화에 경사하강법을 사용하는 구조로 end-to-end 학습에 유연하게 적용 가능
  2. Sequential attention을 사용하여 feature selection의 이유를 추적할 수 있게 하여 interpretability 확보
  3. 다른 도메인의 회귀와 분류 데이터 셋에서도 매우 높은 성능을 보이고, 두 종류(local, global)의 interpretability 제공
  4. 정형 데이터셋에서 처음 비정형 사전학습(unsupervised pre-training)이 성능을 크게 향상

### Architecture
- 개요
  - TabNet은 Sparse feature selection을 제안해 학습 효율을 높이고 interpretability를 확보하여 feature selection mask를 학습이 가능하도록 변수로 지정하고, sparse selection이 가능하도록 활성함수로 sparsemax를 사용
    - Sparsemax는 softmax와 다르게 출력값에 0, 1이 포함되므로 지나치게 중요성이 낮은 feature에 대한 learning capacity 낭비를 방지
  ![image](https://user-images.githubusercontent.com/59715960/234817143-c58d5125-1f07-49a5-af9d-1805c03a20ea.png)
  - 위의 그림과 같이 여러 단계(step)의 decision step을 거치면서 각 단계마다 feature selection mask를 구하여 각 단계 별 핵심 feature 파악이 가능하고, 각 단계 별 mask를 합하여 개별 입력 데이터에 대해서도 feature의 중요도를 파악 가능하다
    - instance-wise feature selection 달성 및 local interpretability 확보
    - 각 mask를 전체 데이터에 대해 합하여 중요도 확인 가능 -> global interpretability
  
- Encoder
<img src=https://user-images.githubusercontent.com/59715960/234817915-8102e9be-7526-4f6c-8a11-807eb9ec40c5.png width="600" height="300"/>

  - Feature transformer에서 인코딩을 수행하여 Attentive transformer가 인코딩의 결과로 feature slection에 해당하는 mask를 생성
  - 위의 과정을 여러 단계의 decision step을 거치면서 반복하고 이전 단계의 feature는 다음 단계의 mask를 생성하는데 활용
  
- Feature Selection -> Attentive transformer
<img src=https://user-images.githubusercontent.com/59715960/234818565-e42b49d3-d304-4035-97d8-ee997404d9be.png width="350" height="350"/>

  - Attentive transformer는 learnable mask를 생성
    - 핵심 feature에 대해 soft selection을 수행
    <img width="240" alt="스크린샷 2023-04-27 오후 6 25 54" src="https://user-images.githubusercontent.com/59715960/234820135-5d9a0f30-1180-456a-a597-a169f5b97f2d.png">

  - softmax 대신 sparsemax를 활성함수로 사용하여 각 decision step마다 중요도가 떨어지는 feature에 learning capacity가 낭비되지 않도록 구성
  - prior scale term(P)을 통해 이전 decision step의 feature 재선택 비율 조절
  <img width="383" alt="스크린샷 2023-04-28 오전 11 36 23" src="https://user-images.githubusercontent.com/59715960/235040858-6703a9bc-b422-4043-b54f-966aa3ee18d9.png">
  - mask는 sparsemax를 활성함수로 사용하므로 mask vector의 합은 1
    - 일반적인 classification에서 softmax로 각 클래스로 예측될 확률을 짐작하듯이 feature의 중요도 파악 가능
  - Prior scale term(P)으로 영향력이 지나치게 큰 feature가 여러 decision step에서 지나치게 중복 선택될 가능성을 조절
  <img width="274" alt="스크린샷 2023-04-28 오후 12 26 09" src="https://user-images.githubusercontent.com/59715960/235047286-6ad41377-0123-493d-a724-f2cfac6f5cef.png">
    - Relaxation parameter(r, Gamma)을 통해 재선택 가능성을 조절
    - r = 1일 때, feature는 오진 1번의 decision step에서만 선택되도록 강제되며, r가 증가함에 따라 여러 decision step에서 선택될 수 있도록 조정
  - 선택된 features의 sparsity 조절을 위해 entropy form 제안
<img width="499" alt="스크린샷 2023-04-28 오후 12 28 48" src="https://user-images.githubusercontent.com/59715960/235047596-e5168d70-77dc-4ad6-ab37-89702d07bbb3.png">

- Feature Processing (Feature transformer)
- <img src=https://user-images.githubusercontent.com/59715960/235047689-12f10d78-70bb-40d2-9235-8d01501221e1.png width="400" height="200"/>

  - Mask에 의해 선택된 feature는 마스크에 feature vector를 곱한 M[i] * f 로 표현
  - 앞의 2개 블럭은 전체 decision step에서 가중치를 공유 (shared weights)를 함
  - 각 블럭은 활성함수로 GLU를 적용하고, sqrt(0.5)로 정규화(normalization)된 skip connection을 적용
  - 빠른 학습을 위해 큰 배치 사이즈를 선택하고 Ghost Batch Normalization을 적용
  
- Interpretability
<img src=https://user-images.githubusercontent.com/59715960/235048302-64b58d87-aabb-4ac0-a349-17ff95f7c836.png width="400" height="300"/>

  - 각 decision step의 마스크 M[i]와 각 마스크가 합쳐져(aggregated) 샘플의 feature importance로 해석 가능한 M_agg의 시각화
  
## Requirements
- python version : 3.6, 3.7, 3.8, 3.9, 3.10
- os issue : Mac OS에서는 GPU 미지원

## How to Use
### Pseudo Code
1. Import Libraries
<pre>
<code>
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import torch
# Regression
from pytorch_tabnet.tab_model import TabNetRegresson
from pytorch_tabnet.augmentations import RegressionSMOTE
# Classification
from pytorch_tabnet.tab_model import TabNetClassification
from pytorch_tabnet.augmentations import ClassificationSMOTE

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error # Regression
from sklearn.metrics import roc_auc_score # Classification
</code>
</pre>

2. Data Load
<pre>
<code>
train = pd.read_csv(...) # train data path
test = pd.read_csv(...) # test data path
</code>
</pre>

3. Simple Pre-Processing
<pre>
<code>
def SPP(df):
    n_unique = df.nunique() # 각 Column에서의 Unique한 Value의 수
    types = df.dtypes
    threshold = 10 # Categorical Feature가 가질 Unique한 Value의 가지 수 제한
    
    cat_columns = [] # Categorical 컬럼을 담을 리스트
    cat_dims = {} # Categorical 컬럼과 Unique한 Value를 담을 딕셔너리
    
    for col in tqdm(df.columns):
        print(col, df[col].nunique())
        if types[col] == 'object' or n_unique[col] < threshold:
            l_enc = LabelEncoder()
            df[col] = df[col].fillna("NULL") # 결측치를 "NULL"이라는 문자열로 치환
            df[col] = l_enc.fit_transform(df[col].values)
            cat_columns.append(col)
            cat_dims[col] = len(l_enc.classes_)
        else:
            df.fillna(df[col].mean(), inplace=True)
    return cat_columns, cat_dims, df
    
cat_columns, cat_dims, train = SPP(train)
test = SPP(test)[2]
</code>
</pre>

4. Define Categorical Features for Categorical Embeddings
<pre>
<code>
target = "" # Target Feature
unused_feat = [] # 학습 시 제외할 column (ex. ID)

features = [ col for col in train.columns if col not in unused_feat+[target]] 
cat_idxs = [ i for i, f in enumerate(features) if f in cat_columns]
cat_dims = [ cat_dims[f] for i, f in enumerate(features) if f in cat_columns]
cat_emb_dim = [5, 4, 3, 6, 2, 2, 1] # Random하게 지정?
</code>
</pre>

5. Split Dataset
<pre>
<code>
X_train, X_test, y_train, y_test = train_test_split(
    train.iloc[:,:-1],train.iloc[:,-1],
    test_size=0.3,
    random_state=530,
    shuffle=True,
    stratify=train.iloc[:,-1]
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,y_train,
    test_size=0.3,
    random_state=530,
    shuffle=True,
    stratify=y_train
)

# 학습용
X_train = X_train.to_numpy()
y_train = y_train.to_numpy().reshape(-1, 1)

# 확인용
X_valid = X_valid.to_numpy()
y_valid = y_valid.to_numpy().reshape(-1, 1)

# 검증용
X_test = X_test.to_numpy()
y_test = y_test.to_numpy().reshape(-1, 1)
</code>
</pre>

6. Training
<pre>
<code>
# Regresssion
clf = TabNetRegressor(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)
# Classification
clf = TabNetClassification(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)

max_epochs = 100 if not os.getenv("CI", False) else 2
aug = RegressionSMOTE(p=0.2)

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['rmsle', 'mae', 'rmse', 'mse'], # Regression
    # eval_metric=['auc'], # Regression
    max_epochs=max_epochs,
    patience=50,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    augmentations=aug, #aug
)

preds = clf.predict(X_test)

y_true = y_test

test_score = mean_squared_error(y_pred=preds, y_true=y_true)

print(f"BEST VALID SCORE FOR {dataset_name} : {clf.best_cost}")
print(f"FINAL TEST SCORE FOR {dataset_name} : {test_score}")
</code>
</pre>

7. Local Explainability and Masks Visualize
<pre>
<code>
explain_matrix, masks = clf.explain(X_test)

fig, axs = plt.subplots(1, 3, figsize=(20,20))

for i in range(3):
    axs[i].imshow(masks[i][:50])
    axs[i].set_title(f"mask {i}")
</code>
</pre>

8. Global Explainability Visualize
<pre>
<code>
clf.feature_importances_
</code>
</pre>
