from sklearn.linear_model import LinearRegression


def remove_columns(data_frame, column_list):
    """
      This function takes a data_frame, removes columns 
      given in column_list and returns the result as a 
      new data frame
    """
    remaining_columns = data_frame.drop(column_list, axis=1)
    return remaining_columns


def linear_regression(data_frame, output_column_name):
    """
      Given data_frame and output_column_name, fits a linear 
      model to columns in the data_frame to predict the 
      column with name output_column_name.

      Returns the learned model and the order of column names 
      used in the regression.
    """
    X = data_frame.drop(output_column_name, axis=1)
    y = data_frame[output_column_name]
    reg = LinearRegression().fit(X, y)
    return reg, X.columns
