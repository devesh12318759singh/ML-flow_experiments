import mlflow
print("Printing uri sceme below")
print(mlflow.get_tracking_uri())
print("\n")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("Printing new uri sceme below")
print(mlflow.get_tracking_uri())
print("\n")