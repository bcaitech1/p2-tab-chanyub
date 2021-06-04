# 정형데이터
> best AUC : 0.8557
![image](https://user-images.githubusercontent.com/54899906/120735668-fa602980-c525-11eb-82f9-e14bfb5eba56.png)
---
## 대회 개요
2009년 12월 ~ 2011년 11월의 데이터로 2011년 12월의 총 구매액이 300을 넘을 확률을 예측하는 Task
- 데이터
  - order_id : 주문 번호. 데이터에서 같은 주문번호는 동일 주문을 나타냄
  - product_id : 상품 번호
  - description : 상품 설명
  - quantity : 상품 주문 수량
  - order_date : 주문 일자
  - price : 상품 가격
  - customer_id : 고객 번호
  - country : 고객 거주 국가
  - total : 총 구매액(quantity X price)
  총, 780502개의 row(주문기록), 9개의 column(feature 갯수)
![image](https://user-images.githubusercontent.com/54899906/120739239-2b435d00-c52c-11eb-856a-fd2df35591e3.png)

- 평가 방법
  - AUC
---
## 최종 모델
- catboost
![image](https://user-images.githubusercontent.com/54899906/120744129-51b9c600-c535-11eb-8481-3afce22ce072.png)

---
## EDA
- EDA.ipynb로 저장해둠
- outlier가 소수 존재
---
## 가설과 검증
- 성공 목록
  - 2011년 12월의 고객 구매액을 예측하는 task이므로 order_date에서 시,분,초까지는 필요가 없을 것이다. -> **order_date를 '%y-%m-%d'형태로 변경**
  - **str로 주어진 country 컬럼을 라벨 인코딩** -> 성능 향상!
  - product_id를 통째로 라벨 인코딩하기에는 결과값이 너무 많아지고 product_id에도 어느정도 규칙성이 있어보인다. -> **product_id의 앞 3자리만 라벨 인코딩** -> 성능 향상!
  - **새로운 feature 추가 1: EU 국가인지 아닌지** -> 성능 매우 조금 향상!
    
    EDA했을때, 배송비와 관련있어보이는 post라는 값을 살펴보니 독일이 1위, 영국은 3위임.
    
    영국은 거래량이 제일 많은데 1위가 아닌 것도 이상했고 쇼핑몰이 영국에 있어 영국 내 배송은 30파운드 이상이면 무료랬는데 3위인 것도 이상했음.
    
    post를 부과하는 기준이 무엇일지 고민하다가 대체로 유럽국가가 많이 부과되는 것 같아 eu 국가끼리는 무역을 하는데에 이점이 있지 않을까 하고 추가함
  - **새로운 feature 추가 2: product_id 중 post라는 값을 따로 feature로 추출**

    product_id 중 post라는 값이 발견되었고 EDA를 해본 결과 한번에 많은 물품을 시키는 고객에게 부과되는 배송비라고 판단되었음.
    
    post와 300불이 넘게 주문하는 경우의 상관관계가 있을 것이라고 판단하고 따로 feature로 빼줌
- 실패 목록
  - feature 속에서 **고객의 씀씀이를 수학적으로 대변할 수 있는 feature를 새로 만들어내고자 했다.**
    ![image](https://user-images.githubusercontent.com/54899906/120737117-75c2da80-c528-11eb-8fa6-bdb0ef51a310.png)
    order_id 별 total의 평균을 다시한번 고객을 기준으로 평균 내주는 방식으로 "이 고객은 한번 주문할 때, 평균적으로 이만큼의 돈을 사용한다."는 의미를 가지는 feature를 만들어내려고 했다.
    
    그러나 부스트캠프 수업이 진행되면서 order_id를 기준으로 price의 cumsum을 구함으로써 더 쉽고 정확하게 나의 아이디어를 구현한 것을 발견함
    
    -> 처음에는 절망했지만.. 지금은 더 좋은 방법을 배웠고 나의 아이디어에 대한 자신감이 생겼으며 수학적인 고찰을 할 수 있는 기회였다고 생각 중

---
## 아쉬운 점
- 나처럼 새로운 feature를 생성하는데에 집중하는 다른 부스트캠퍼와 DM으로 따로 연락을 주고받았는데, 이번 task를 시계열 데이터로 접근하였고 매우 좋은 결과를 얻어내는 것을 보았음
  나도 시계열 데이터로 접근하고 조금 건드려볼껄.. 하는 생각이 듦
