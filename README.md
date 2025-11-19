# BlueBikesML

# Rider Demand Prediction (Machine Learning Project)

## Executive Summary
  Bluebikes experiences highly variable rider demand based on seasonality, day of week, and time of day, which directly affects staffing, bicycle distribution, inventory, and fleet maintenance planning. This project developed a machine learning forecasting model to accurately predict total rider count and support operational decision-making. After cleaning and engineering temporal features from the raw dataset, multiple models were developed and evaluated, including Linear Regression, Regression Tree, Random Forest (Bagging), and a Stacking Ensemble.

  This means the model reduces forecasting mistakes by more than 92% compared to not using a model at all, giving us a much more accurate and reliable understanding of how many riders to expect at any given time. We also reduced the average size of our prediction mistakes (RMSE) from about 182 riders off to about 68 riders off — a 62% improvement. Instead of being nearly 200 bikes wrong on a typical forecast, we are now usually within about 70, enabling far more precise operational planning. These improvements support better balancing of bike availability across stations, optimization of rebalancing logistics, reduced operational costs, and improved customer experience by minimizing stockouts and excess inventory. This model can be used to support daily demand forecasting, seasonal planning, and staffing allocation.
