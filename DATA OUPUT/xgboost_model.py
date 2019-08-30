import xgboost as xg

trainset = xg.DMatrix("train_data.csv?format=csv&label_column=3")
testset = xg.DMatrix("test_data.csv?format=csv&label_column=3")