import pandas as pd

# 데이터셋 불러오기
df = pd.read_excel('pharmacy_data.xlsx')

df.drop(columns=['Unnamed: 0'], inplace=True)
df.drop(columns=['weekday'], inplace=True)
df.drop(columns=['ahumidity'], inplace=True)
df.drop(columns=['SS'], inplace=True)
df.drop(columns=['SR'], inplace=True)
df.drop(columns=['maxtemp'], inplace=True)
df.drop(columns=['atemp'], inplace=True)
df.drop(columns=['dtemprange'], inplace=True)
df.drop(columns=['meanwindspeed'], inplace=True)
df.drop(columns=['maxwindsnd'], inplace=True)
df.drop(columns=['maxinwindspeed'], inplace=True)
df.drop(columns=['maxinwindsnd'], inplace=True)

#datetime 타입으로 바꾸기
df['date']=pd.to_datetime(df['date'])  #datetime형식으로 바뀜
df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
df['weekday']=df['date'].dt.weekday
df.drop(columns=['date'], inplace=True)

#데이터 나누기
# 결측치가 없는 데이터만 사용
X=df.drop(['count'],axis=1)
Y=df['count']

# 데이터 나누기
from sklearn.model_selection import train_test_split
#교차검증 조건
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0) #X_test : 피처

#피쳐 스케일링 : 범위표준화
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train  = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##### 모델로 예측
### 랜덤 포레스트
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV #GridSearchCV 에서 CV는 교차검증
import warnings
warnings.filterwarnings('ignore')
import numpy as np

rf_model = RandomForestRegressor()
rf_params={'random_state':[42],'n_estimators':[100],'max_depth':[5]}  #'n_estimators':[100,120,140],'max_depth':[5,10,15,20]
gridsearch_rf_model =GridSearchCV(estimator=rf_model,
                                param_grid=rf_params,
                                scoring='accuracy',
                                cv=5)     # 80%로 나누어지 train 데어터를 다시 cv=5 로 나눔

#그리드서치 실행
log_train = np.log(y_train)
gridsearch_rf_model.fit(X_train,log_train)   #학습 시키는 과정
print('최적 하이퍼파라미터:',gridsearch_rf_model.best_params_)


##랜덤포레스트II
rf_model = RandomForestRegressor(random_state=42)  #1217
# 모델 학습
rf_model.fit(X_train, y_train)
# 예측값 생성
y_test = rf_model.predict(X_test)


# ##XGBoost
# from xgboost import XGBRegressor
# xgb_model = XGBRegressor(random_state=1217)
# # 모델 학습
# xgb_model.fit(X_train, y_train)
# # 예측값 생성
# y_test = xgb_model.predict(X_test)


# ##LightGBM
# import lightgbm as lgb
# lgbm = lgb.LGBMRegressor(random_state=1217)
# # 모델 학습
# lgbm.fit(X_train, y_train)
# # 예측값 생성
# y_test = lgbm.predict(X_test)


####교차검증
import numpy as np
from sklearn.model_selection import cross_val_score

rf_model = RandomForestRegressor(random_state=1217)

# cv=5인 교차 검증
scores = cross_val_score(rf_model, X_train, y_train, cv=5, n_jobs=-1,
                         scoring = 'neg_mean_squared_error')

# 성능 확인
print('cross_val_score \n{}'.format(np.sqrt(-scores)))
print('cross_val_score.mean \n{:.3f}'.format(np.sqrt(-scores.mean())))


#gridsearch에서는 score로 평가할수 없음
from sklearn.metrics import mean_squared_log_error, r2_score   #metrics 평가지표

#예측
preds= gridsearch_rf_model.best_estimator_.predict(X_test) #예측값과 비교

# y_true = np.exp(log_train)
y_pred = np.exp(preds)

MSLE = mean_squared_log_error(y_test,y_pred)
R2=r2_score(np.log(y_test),preds)

print(MSLE)
print(R2)


from datetime import date
today = date.today()

print('오늘 조제건수를 예측하기 위해서, 다음 8개 인자를 넣어주세요')
# 참고 사이트 https://search.naver.com/search.naver?where=nexearch&sm=top_sly.hst&fbm=0&acr=1&ie=utf8&query=%EB%82%A0%EC%94%A8+%EC%82%AC%EB%8B%B9%EB%8F%99
val_year=today.year
val_month=today.month
val_weekday= input("3.요일 입력하세요. 월:0,화:1,수:2,목:3,금:4,토:5 >>>")         
val_SC=input("4.예상 일조량을 입력하세요(맑음:0,구름조금:25,구름많음:50,흐림:75)>>>")
val_mintemp=input("5.예상 최저온도를 입력>>>")
val_rainfall=input("6.예상 강수량을 입력>>>   ")          
val_maxwindspeed=input("7.예상 최대풍속을 입력>>>") 
val_minhumidity= input("8.예상 최저습도 입력>>> ")   

new_data_point =[val_minhumidity,val_SC,val_rainfall,val_mintemp,val_maxwindspeed,val_year,val_month,val_weekday]
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=1217)
# 모델 학습
rf_model.fit(X_train, y_train)
# 예측값 생성
# y_test = rf_model.predict(X_test)

# 랜덤 포레스트 모델에 적용하여 예측값 b 계산
predicted_b = rf_model.predict([new_data_point])

print(f"예측값 조제건수: {predicted_b[0]}")
