from pathlib import Path
import json, warnings
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor,RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from features import prepare_features,CATEGORICAL,NUMERIC
warnings.filterwarnings("ignore")
def prep():
 return ColumnTransformer([("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),CATEGORICAL),("num",SimpleImputer(strategy="median"),NUMERIC)])
def main():
 p=Path("data/traffic.csv")
 if not p.exists(): raise SystemExit("Dataset missing. Run python download_data.py first.")
 d=pd.read_csv(p); d.columns=[str(c).strip().lower() for c in d.columns]
 d["date_time"]=pd.to_datetime(d["date_time"],errors="coerce"); d["traffic_volume"]=pd.to_numeric(d["traffic_volume"],errors="coerce")
 d=d.dropna(subset=["date_time","traffic_volume"]).sort_values("date_time").reset_index(drop=True)
 X=prepare_features(d); y=d.traffic_volume.astype(float); cut=int(.8*len(d)); Xt,Xv=X.iloc[:cut],X.iloc[cut:]; yt,yv=y.iloc[:cut],y.iloc[cut:]
 lo,hi=yt.quantile(.33),yt.quantile(.67); band=lambda a:np.where(a<=lo,"Low",np.where(a<=hi,"Medium","High")); ct,cv=band(yt),band(yv)
 regs={"Random Forest":RandomForestRegressor(n_estimators=180,min_samples_leaf=2,n_jobs=-1,random_state=42),"Extra Trees":ExtraTreesRegressor(n_estimators=180,min_samples_leaf=2,n_jobs=-1,random_state=42),"Hist Gradient Boosting":HistGradientBoostingRegressor(max_iter=160,random_state=42)}
 clfs={"Random Forest":RandomForestClassifier(n_estimators=180,class_weight="balanced",n_jobs=-1,random_state=42),"Extra Trees":ExtraTreesClassifier(n_estimators=180,class_weight="balanced",n_jobs=-1,random_state=42),"Hist Gradient Boosting":HistGradientBoostingClassifier(max_iter=160,random_state=42)}
 try:
  from xgboost import XGBRegressor,XGBClassifier
  regs["XGBoost"]=XGBRegressor(n_estimators=220,max_depth=7,learning_rate=.06,subsample=.85,colsample_bytree=.85,objective="reg:squarederror",n_jobs=-1,random_state=42)
  clfsXGBClassifier(n_estimators=220,max_depth=7,learning_rate=.06,subsample=.85,colsample_bytree=.85,objective="multi:softprob",eval_metric="mlogloss",n_jobs=-1,random_state=42)
 except Exception: print("XGBoost not available; using three sklearn algorithms.")
 out=Path("models"); out.mkdir(exist_ok=True); rr={}; cc={}; best_r=(None,1e99,None); best_c=(None,-1,None)
 for name,est in regs.items():
  model=Pipeline([("prep",prep()),("model",est)]); model.fit(Xt,yt); pred=model.predict(Xv)
  s={"MAE":round(float(mean_absolute_error(yv,pred)),2),"RMSE":round(float(np.sqrt(mean_squared_error(yv,pred))),2),"R2":round(float(r2_score(yv,pred)),4)}; rr[name]=s; print(name,"Regressor",s)
  if s["MAE"]<best_r[1]:best_r=(name,s["MAE"],model)
 for name,est in clfs.items():
  model=Pipeline([("prep",prep()),("model",est)]); model.fit(Xt,ct); pred=model.predict(Xv)
  s={"accuracy":round(float(accuracy_score(cv,pred)),4),"precision_macro":round(float(precision_score(cv,pred,average="macro",zero_division=0)),4),"recall_macro":round(float(recall_score(cv,pred,average="macro",zero_division=0)),4),"f1_macro":round(float(f1_score(cv,pred,average="macro",zero_division=0)),4),"confusion_matrix":confusion_matrix(cv,pred,labels=["Low","Medium","High"]).tolist()}; cc[name]=s; print(name,"Classifier",s)
  if s["f1_macro"]>best_c[1]:best_c=(name,s["f1_macro"],model)
 joblib.dump(best_r[2],out/"best_regressor.joblib"); joblib.dump(best_c[2],out/"best_classifier.joblib")
 m={"dataset_rows":len(d),"train_rows":cut,"test_rows":len(d)-cut,"thresholds":{"low_upper":float(lo),"medium_upper":float(hi)},"regression":rr,"classification":cc,"best_regressor":best_r[0],"best_classifier":best_c[0]}
 (out/"metrics.json").write_text(json.dumps(m,indent=2)); print("Best:",best_r[0],"Regressor |",best_c[0],"Classifier")
if __name__=="__main__":main()
