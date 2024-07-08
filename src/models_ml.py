from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def get_ml_models():
    models = {
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor()
    }
    return models
