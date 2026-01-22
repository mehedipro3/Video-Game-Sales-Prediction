import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("vgsales.csv")


# Data Preprocessing
df["Year"] = df["Year"].fillna(0).astype(int)
df["Publisher"] = df["Publisher"].fillna(df["Publisher"].mode()[0])
df.drop(columns=['Rank', 'Name'], inplace=True)

# Feature Selection
X = df.drop(columns=['Global_Sales'])
y = df['Global_Sales']

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline , numerical_features),
    ('cat', categorical_pipeline , categorical_features)
])

model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ))
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

print("RMSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


cv_scores = cross_val_score(
    model_pipeline,
    X,
    y,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

print("Cross-Validation R² Scores:", cv_scores)
print("Average R² Score:", np.mean(cv_scores))
print("Standard Deviation:", np.std(cv_scores))

param_grid = {
    'model__n_estimators' : [100,200],
    'model__max_depth'  : [None,10,20],
    'model__min_samples_split' : [2,5]
}

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=2,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters Found:")
print(grid_search.best_params_)

print("\nBest Cross-Validation R² Score:")
print(grid_search.best_score_)


best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)

rmse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("RMSE:", rmse)
print("MAE:", mae)
print("R² Score:", r2)


#save the model with pickle
with open('video_game_sales_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("Model saved to video_game_sales_model.pkl")