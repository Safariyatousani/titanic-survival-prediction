def preprocess_data(df):
  df["Age"] = df["Age"].fillna(df["Age"].median())
  df["Fare"] = df["Fare"].fillna(df["Fare"].median())
  return df 