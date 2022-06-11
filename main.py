from notebooks.linear.ClimateLinearRegression import run_linear_regression_for_climate
from notebooks.tree.ClimateTreeRegression import climate_tree_regression

# run_linear_regression_for_bikes('./data/Accidents_done.csv')
run_linear_regression_for_climate('./data/climate_and_air.csv')

climate_tree_regression('data/climate_and_air.csv')