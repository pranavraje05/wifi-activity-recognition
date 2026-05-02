  import pandas as pd
  from sklearn.ensemble import RandomForestClassifier
  
  def train_model():
      data = pd.read_csv("data/sample_data.csv")
  
      X = data[["feature1", "feature2", "feature3"]]
      y = data["activity"]
  
      model = RandomForestClassifier()
    model.fit(X, y)

    return model
