import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv).
from pandas.core.frame import DataFrame  # Data format.
import numpy as np
from helpers import *

train_data_path = 'C:\\Users\\sevgi\\OneDrive\\Masaüstü\\Kaggle Housing Prices Competition\\train.csv'
train_data = pd.read_csv(train_data_path)
train_data['Bias'] = 1

copy_of_train_data = train_data.copy()
copy_of_train_data = remove_columns(copy_of_train_data, ["Id"])
unnecessary_columns = ['Street', 'Utilities', 'Condition2', 'Heating', 'PoolQC',
                       'MoSold', 'YrSold', 'GarageYrBlt', 'Alley', 'LotConfig',
                       'LandSlope', 'GarageFinish']
copy_of_train_data = remove_columns(copy_of_train_data, unnecessary_columns)

categorical_columns = ['MSSubClass', 'MSZoning', 'Street', 'LotShape',
                       'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                       'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                       'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle',
                       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                       'ExterQual', 'ExterCond', 'Foundation', 'BsmtCond', 'BsmtExposure',
                       'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'Heating',
                       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                       'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
                       'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
categorical_columns = list(set(categorical_columns) - set(unnecessary_columns))
numerical_columns = list(
    set(copy_of_train_data.columns) - set(categorical_columns))
copy_of_train_data[numerical_columns] = copy_of_train_data[numerical_columns].fillna(
    0)
copy_of_train_data = pd.get_dummies(
    copy_of_train_data, dummy_na=False, columns=categorical_columns)

model, column_order = linear_regression(copy_of_train_data, "SalePrice")

test_data_path = 'C:\\Users\\sevgi\\OneDrive\\Masaüstü\\Kaggle Housing Prices Competition\\test.csv'
test_data = pd.read_csv(test_data_path)
test_data['Bias'] = 1
copy_of_test_data = test_data.copy()
copy_of_test_data = remove_columns(copy_of_test_data, unnecessary_columns)
numerical_columns.remove("SalePrice")
copy_of_test_data[numerical_columns] = copy_of_test_data[numerical_columns].fillna(
    0)

copy_of_test_data = pd.get_dummies(
    copy_of_test_data, dummy_na=False, columns=categorical_columns)

missing_columns_in_test_data = list(
    set(copy_of_train_data.columns) - set(copy_of_test_data.columns))
copy_of_test_data[missing_columns_in_test_data] = 0

predicted_sale_prices = model.predict(copy_of_test_data[column_order])

result = pd.DataFrame()
result["Id"] = copy_of_test_data["Id"]
result["SalePrice"] = predicted_sale_prices

result_file = open("result.csv", "w")
result_file.write(result.to_csv(
    columns=["Id", "SalePrice"], line_terminator="\n", index=False))
result_file.close()
