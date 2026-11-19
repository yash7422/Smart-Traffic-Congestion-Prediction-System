import pandas as pd
CATEGORICAL=["holiday","weather_main"]
NUMERIC=["temp","rain_1h","snow_1h","clouds_all","hour","dayofweek","month","is_weekend","is_rush_hour"]
def prepare_features(df):
    d=df.copy(); d.columns=[str(c).strip().lower() for c in d.columns]
    dt=pd.to_datetime(d["date_time"],errors="coerce")
    d["hour"]=dt.dt.hour; d["dayofweek"]=dt.dt.dayofweek; d["month"]=dt.dt.month
    d["is_weekend"]=(dt.dt.dayofweek>=5).astype(int)
    d["is_rush_hour"]=dt.dt.hour.isin([7,8,9,16,17,18,19]).astype(int)
    for c in CATEGORICAL: d[c]=d[c].fillna("None").astype(str) if c in d else "None"
    for c in ["temp","rain_1h","snow_1h","clouds_all"]:
        if c not in d: d[c]=0
        d[c]=pd.to_numeric(d[c],errors="coerce")
    return d[CATEGORICAL+NUMERIC].fillna(0)
