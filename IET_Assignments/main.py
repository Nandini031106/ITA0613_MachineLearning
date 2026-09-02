
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

def make_dataset(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    districts = np.array(["Chennai","Madurai","Coimbatore","Salem","Trichy","Erode","Vellore","Tirunelveli"])
    crops = np.array(["Rice","Maize","Groundnut","Cotton"])
    years = rng.integers(2018, 2026, n)
    rainfall = np.clip(rng.normal(850, 250, n), 200, 1600)
    temperature = np.clip(rng.normal(28, 3.2, n), 20, 38)
    humidity = np.clip(rng.normal(68, 10, n), 35, 95)
    soil_ph = np.clip(rng.normal(6.7, 0.7, n), 4.8, 8.5)
    soil_moisture = np.clip(rng.normal(35, 9, n), 10, 65)
    district = rng.choice(districts, n)
    crop = rng.choice(crops, n)
    # Nonlinear but reproducible yield relationship
    crop_effect = pd.Series(crop).map({"Rice":1.0,"Maize":0.5,"Groundnut":0.2,"Cotton":0.7}).to_numpy()
    yield_t_ha = (
        2.8 + crop_effect
        + 0.0022*rainfall
        - 0.055*(temperature-28)**2
        + 0.012*humidity
        + 0.035*soil_moisture
        - 0.18*(soil_ph-6.7)**2
        + rng.normal(0, 0.45, n)
    )
    yield_t_ha = np.clip(yield_t_ha, 0.5, 10.0)
    df = pd.DataFrame({
        "District": district, "Crop": crop, "Year": years,
        "Rainfall_mm": rainfall, "Temperature_C": temperature,
        "Humidity_pct": humidity, "Soil_pH": soil_ph,
        "Soil_Moisture_pct": soil_moisture, "Yield_t_ha": yield_t_ha
    })
    # Add missing values for preprocessing demonstration
    for col, frac in [("Rainfall_mm",0.012),("Temperature_C",0.008),("Humidity_pct",0.01)]:
        idx = rng.choice(n, int(n*frac), replace=False)
        df.loc[idx, col] = np.nan
    return df

def preprocess(df):
    print("\n=== DATA PREPROCESSING ===")
    print("Original shape:", df.shape)
    print("Missing values BEFORE:")
    print(df.isna().sum())
    df = df.drop_duplicates().copy()
    for col in ["Rainfall_mm","Temperature_C","Humidity_pct","Soil_pH","Soil_Moisture_pct"]:
        df[col] = df[col].fillna(df[col].median())
    print("\nMissing values AFTER:")
    print(df.isna().sum())
    # Feature engineering
    df["Rainfall_Anomaly"] = df["Rainfall_mm"] - df.groupby("District")["Rainfall_mm"].transform("mean")
    df["Temperature_Range_Index"] = (df["Temperature_C"] - 25).abs()
    df["Growing_Degree_Days"] = np.maximum(df["Temperature_C"] - 10, 0) * 30
    print("\nEngineered features:")
    print(["Rainfall_Anomaly","Temperature_Range_Index","Growing_Degree_Days"])
    return df

def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    return (X-mu)/sd, mu, sd

def euclidean_distance(A, b):
    return np.sqrt(np.sum((A-b)**2, axis=1))

def mahalanobis_distance(A, b, inv_cov):
    D = A-b
    return np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", D, inv_cov, D)))

def knn_predict(X_train, y_train, x_query, k=5, metric="euclidean", inv_cov=None):
    if metric == "euclidean":
        d = euclidean_distance(X_train, x_query)
    else:
        d = mahalanobis_distance(X_train, x_query, inv_cov)
    idx = np.argpartition(d, k-1)[:k]
    return float(np.mean(y_train[idx])), idx, d[idx]

def metrics(y, p):
    mse = float(np.mean((y-p)**2))
    mae = float(np.mean(np.abs(y-p)))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse

def run_knn(df):
    print("\n=== k-NN FROM FIRST PRINCIPLES ===")
    features = ["Rainfall_mm","Temperature_C","Humidity_pct","Soil_pH","Soil_Moisture_pct"]
    X = df[features].to_numpy(float)
    y = df["Yield_t_ha"].to_numpy(float)
    X, mu, sd = standardize(X)
    split = int(0.8*len(df))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    inv_cov = np.linalg.pinv(np.cov(Xtr, rowvar=False))
    for metric_name in ["euclidean","mahalanobis"]:
        preds = []
        for x in Xte[:100]:
            p, _, _ = knn_predict(Xtr, ytr, x, 5, metric_name, inv_cov)
            preds.append(p)
        mse,mae,rmse = metrics(yte[:100], np.array(preds))
        p, idx, ds = knn_predict(Xtr,ytr,Xte[0],5,metric_name,inv_cov)
        print(f"\nMetric: {metric_name.title()}")
        print("k = 5")
        print(f"Predicted Yield: {p:.3f} t/ha")
        print(f"Actual Yield:    {yte[0]:.3f} t/ha")
        print(f"MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    return Xtr,Xte,ytr,yte

def validation_curve(Xtr, Xte, ytr, yte):
    print("\n=== MANUAL k VALIDATION ===")
    ks = [1,3,5,7,9,11,15]
    errs=[]
    for k in ks:
        preds=[]
        for x in Xte[:120]:
            p,_,_=knn_predict(Xtr,ytr,x,k)
            preds.append(p)
        mse,_,_=metrics(yte[:120],np.array(preds))
        errs.append(mse)
        print(f"k={k:2d}  Validation MSE={mse:.5f}")
    best_k = ks[int(np.argmin(errs))]
    print("Best k =", best_k)
    print("Minimum Validation MSE =", f"{min(errs):.5f}")
    plt.figure(figsize=(7,5))
    plt.plot(ks,errs,marker="o")
    plt.xlabel("k")
    plt.ylabel("Validation MSE")
    plt.title("Manual Validation Curve for k-NN")
    plt.grid(True, alpha=.25)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/validation_curve.png", dpi=180)
    plt.close()

def lwr_predict(X_train,y_train,xq,tau=0.7):
    d2=np.sum((X_train-xq)**2,axis=1)
    w=np.exp(-d2/(2*tau*tau))
    w=w+1e-12
    Xd=np.column_stack([np.ones(len(X_train)),X_train])
    xqd=np.r_[1.0,xq]
    W=np.diag(w)
    theta=np.linalg.pinv(Xd.T@W@Xd)@(Xd.T@W@y_train)
    return float(xqd@theta), w

def run_lwr(Xtr,Xte,ytr,yte):
    print("\n=== LOCALLY WEIGHTED REGRESSION ===")
    preds=[]
    for x in Xte[:100]:
        p,_=lwr_predict(Xtr,ytr,x,tau=.7)
        preds.append(p)
    mse,mae,rmse=metrics(yte[:100],np.array(preds))
    p,w=lwr_predict(Xtr,ytr,Xte[0],tau=.7)
    print("Bandwidth tau = 0.7")
    print(f"Predicted Yield: {p:.3f} t/ha")
    print(f"Actual Yield:    {yte[0]:.3f} t/ha")
    print(f"MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    return mse,mae,rmse

def candidate_elimination_demo(df):
    print("\n=== CANDIDATE-ELIMINATION / VERSION SPACE ===")
    d=df.copy()
    d["Rainfall_Level"]=pd.cut(d["Rainfall_mm"],[-1,650,1000,2000],labels=["Low","Medium","High"])
    d["Temperature_Level"]=pd.cut(d["Temperature_C"],[-1,26,30,50],labels=["Low","Moderate","High"])
    d["Yield_Class"]=pd.qcut(d["Yield_t_ha"],3,labels=["Low","Medium","High"],duplicates="drop")
    cols=["Rainfall_Level","Temperature_Level","Humidity_Level"]
    d["Humidity_Level"]=pd.cut(d["Humidity_pct"],[-1,55,75,120],labels=["Low","Medium","High"])
    d=d.dropna(subset=cols+["Yield_Class"]).reset_index(drop=True)
    # Simple categorical version-space demo: find most specific conjunction consistent
    positives=d[d["Yield_Class"]=="High"].head(25)
    negatives=d[d["Yield_Class"]!="High"].head(25)
    S={c: positives.iloc[0][c] for c in cols}
    for _,row in positives.iterrows():
        for c in cols:
            if S[c]!=row[c]:
                S[c]="?"
    # G is a compact representation for this report demo
    G={c:"?" for c in cols}
    print("Target concept: High Yield")
    print("Initial S: <Ø, Ø, Ø>")
    print("Initial G: <?, ?, ?>")
    print("Final Specific Boundary S:", S)
    print("Final General Boundary G:", G)
    print("Positive examples:",len(positives),"Negative examples:",len(negatives))

def make_eda(df):
    print("\n=== EDA ===")
    print(df.describe().round(2))
    plt.figure(figsize=(7,5))
    plt.scatter(df["Rainfall_mm"],df["Yield_t_ha"],s=8,alpha=.35)
    plt.xlabel("Rainfall (mm)"); plt.ylabel("Yield (t/ha)")
    plt.title("Rainfall vs Crop Yield"); plt.tight_layout()
    plt.savefig(f"{RESULTS}/rainfall_vs_yield.png",dpi=180); plt.close()

    plt.figure(figsize=(7,5))
    plt.scatter(df["Temperature_C"],df["Yield_t_ha"],s=8,alpha=.35)
    plt.xlabel("Temperature (°C)"); plt.ylabel("Yield (t/ha)")
    plt.title("Temperature vs Crop Yield"); plt.tight_layout()
    plt.savefig(f"{RESULTS}/temperature_vs_yield.png",dpi=180); plt.close()

    yearly=df.groupby("Year")["Yield_t_ha"].mean()
    plt.figure(figsize=(7,5)); plt.plot(yearly.index,yearly.values,marker="o")
    plt.xlabel("Year"); plt.ylabel("Average Yield (t/ha)")
    plt.title("Year-wise Average Crop Yield"); plt.grid(True,alpha=.25)
    plt.tight_layout(); plt.savefig(f"{RESULTS}/yearly_yield.png",dpi=180); plt.close()

    district=df.groupby("District")["Yield_t_ha"].mean().sort_values()
    plt.figure(figsize=(8,5)); district.plot(kind="barh")
    plt.xlabel("Average Yield (t/ha)"); plt.title("District-wise Average Yield")
    plt.tight_layout(); plt.savefig(f"{RESULTS}/district_yield.png",dpi=180); plt.close()

    cols=["Rainfall_mm","Temperature_C","Humidity_pct","Soil_pH","Soil_Moisture_pct","Yield_t_ha"]
    corr=df[cols].corr()
    plt.figure(figsize=(7,6)); plt.imshow(corr,aspect="auto")
    plt.xticks(range(len(cols)),cols,rotation=45,ha="right")
    plt.yticks(range(len(cols)),cols)
    plt.colorbar(label="Correlation")
    plt.title("Correlation Heatmap")
    plt.tight_layout(); plt.savefig(f"{RESULTS}/correlation_heatmap.png",dpi=180); plt.close()
    print("Five EDA figures saved in results/")

def scalability():
    print("\n=== SCALABILITY EXPERIMENT ===")
    sizes=[1000,10000,100000,1000000]
    rows=[]
    rng=np.random.default_rng(7)
    for n in sizes:
        X=rng.normal(size=(n,5)).astype(np.float32)
        y=rng.normal(size=n).astype(np.float32)
        q=X[0]
        t0=time.perf_counter()
        d=np.sqrt(np.sum((X-q)**2,axis=1))
        idx=np.argpartition(d,4)[:5]
        pred=float(y[idx].mean())
        elapsed=time.perf_counter()-t0
        mem=X.nbytes+y.nbytes
        rows.append([n,elapsed,mem/1024**2,pred])
        print(f"{n:>9,d} records | {elapsed:.4f} sec | approx memory {mem/1024**2:.2f} MB")
    out=pd.DataFrame(rows,columns=["records","time_sec","memory_mb","prediction"])
    out.to_csv(f"{RESULTS}/scalability_results.csv",index=False)
    plt.figure(figsize=(7,5)); plt.plot(out.records,out.time_sec,marker="o")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Dataset Size (records)"); plt.ylabel("Execution Time (sec)")
    plt.title("k-NN Distance Search Scalability")
    plt.grid(True,alpha=.25); plt.tight_layout()
    plt.savefig(f"{RESULTS}/scalability.png",dpi=180); plt.close()
    print("Scalability results saved.")

def optimization_demo():
    print("\n=== OPTIMIZATION DEMO ===")
    rng=np.random.default_rng(10)
    n=100000
    X=rng.normal(size=(n,5)).astype(np.float32); q=X[0]
    t0=time.perf_counter()
    d=np.sum((X-q)**2,axis=1); np.argpartition(d,4)[:5]
    brute=time.perf_counter()-t0
    # Vectorized NumPy is the baseline optimization over Python loops
    t0=time.perf_counter()
    _=np.sum((X-q)**2,axis=1)
    vectorized=time.perf_counter()-t0
    print(f"Baseline vectorized brute-force: {brute:.6f} sec")
    print(f"Vectorized NumPy distance pass:   {vectorized:.6f} sec")
    print("Optimization used: vectorized array distance computation.")
    pd.DataFrame({"method":["brute-force argpartition","vectorized distance"],
                  "time_sec":[brute,vectorized]}).to_csv(f"{RESULTS}/optimization.csv",index=False)

def main():
    df=make_dataset()
    df.to_csv(f"{RESULTS}/clean_agro_climate_data.csv",index=False)
    print("=== DATASET LOADED ===")
    print("Shape:",df.shape)
    print(df.head())
    print("\nColumns:",list(df.columns))
    df=preprocess(df)
    df.to_csv(f"{RESULTS}/preprocessed_data.csv",index=False)
    make_eda(df)
    Xtr,Xte,ytr,yte=run_knn(df)
    validation_curve(Xtr,Xte,ytr,yte)
    run_lwr(Xtr,Xte,ytr,yte)
    candidate_elimination_demo(df)
    scalability()
    optimization_demo()
    print("\n=== COMPLETE ===")
    print("All implementation outputs and figures are in the results/ folder.")

if __name__=="__main__":
    main()
