import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df=pd.read_csv("hour.csv")
df.head()

df["dteday"]=pd.to_datetime(df["dteday"])
df=df.sort_values(["dteday",'hr'])

y=df.cnt
features=['mnth','yr','hr','weekday','holiday','weathersit','hum','windspeed','temp','workingday']
X=df[features]

train_X,val_X,train_y,val_y=train_test_split(X,y,test_size=0.25,shuffle=False)

for leaves in [1,2,5,10]:
  model=RandomForestRegressor(min_samples_leaf=leaves,random_state=1)
  model.fit(train_X,train_y)
  modelPredict=model.predict(val_X)
  mae=mean_absolute_error(val_y,modelPredict)
  print(f"Leaves: {leaves},MAE: {mae:.2f}")

model1=RandomForestRegressor(min_samples_leaf=1,random_state=1)
model1.fit(train_X,train_y)
model1Predict=model1.predict(val_X)
mean_absolute_error(val_y,model1Predict)

comparison=pd.DataFrame({
    "Actual":val_y,
    "Predicted":model1Predict
})

comparison.head(10)

comparison["Absolute_Error"]=(
    comparison["Actual"]-comparison["Predicted"]
).abs()

comparison["hr"]=val_X["hr"]

error_by_hour=(
    comparison.groupby("hr")["Absolute_Error"].mean().sort_values(ascending=False)
)

print(error_by_hour.head(5))

hour_summary=comparison.groupby("hr")[["Actual","Predicted","Absolute_Error"]].mean()

print(hour_summary.sort_values("Absolute_Error",ascending=False).head(5).round(2))

import matplotlib.pyplot as plt

hour_summary[["Actual","Predicted"]].plot(figsize=(10,4),marker="o")

plt.title("Avergae Bike Rentals by Hour - Validation")
plt.xlabel("Hour of Day")
plt.ylabel("Average Rentals")
plt.xticks(range(24))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Missing Values:")
print(df.isna().sum())

print("\nDuplicate rows:",df.duplicated().sum())

print("Duplicate date-hour pairs:",
      df.duplicated(subset=["dteday","hr"]).sum()
      )
