import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import openai
from scipy.optimize import minimize

# Load environment variables
load_dotenv()

# Set up database connection using SQLAlchemy
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

# Function to fetch data from the database
def fetch_data_from_db(symbol, engine):
    try:
        query = text("SELECT * FROM stock_prices WHERE symbol = :symbol")
        with engine.connect() as conn:
            data = pd.read_sql(query, conn, params={"symbol": symbol})
        return data
    except SQLAlchemyError as e:
        print(f"Error fetching data from database: {e}")
        return None

# Function to preprocess data
def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['date']).set_index('date')
    df['Return'] = df['close'].pct_change().dropna()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['Volatility'] = df['close'].rolling(window=20).std()
    df.dropna(inplace=True)
    return df

# Feature engineering
def feature_engineering(df):
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['close'], window=14)
    df['MACD'], df['MACD_signal'] = calculate_macd(df['close'])
    df.dropna(inplace=True)
    return df

def calculate_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# Discover hidden correlations
def discover_hidden_correlations(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    return correlation_matrix

# Calculate Herfindahl-Hirschman Index (HHI)
def calculate_hhi(weights):
    return np.sum(np.square(weights))

# Optimize portfolio to mitigate concentration risk
def optimize_portfolio(returns, cov_matrix, risk_free_rate=0.01):
    num_assets = len(returns)
    args = (returns, cov_matrix, risk_free_rate)

    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights

    def portfolio_return(weights):
        return np.sum(returns * weights)

    def negative_sharpe_ratio(weights):
        p_var = portfolio_variance(weights)
        p_ret = portfolio_return(weights)
        return -(p_ret - risk_free_rate) / np.sqrt(p_var)

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]

    optimized_result = minimize(negative_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimized_result.x

# Train model with hyperparameter tuning
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")

    return best_rf, X_test, y_test

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")

    feature_importances = model.feature_importances_
    features = X_test.columns
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=features)
    plt.title('Feature Importances')
    plt.show()

# Get market sentiment using OpenAI
def get_market_sentiment(news_headlines):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or the appropriate chat model
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": f"Analyze the following news headlines and provide a market sentiment summary:\n{news_headlines}"}
        ]
    )
    sentiment = response['choices'][0]['message']['content'].strip()
    return sentiment

# Make portfolio decision
def make_portfolio_decision(model, X_latest, market_sentiment):
    predicted_returns = model.predict(X_latest)
    if 'positive' in market_sentiment:
        adjust_portfolio(predicted_returns, increase_weights=True)
    else:
        adjust_portfolio(predicted_returns, increase_weights=False)

def adjust_portfolio(predictions, increase_weights):
    # Example logic to adjust portfolio
    pass

# Main function to run the script
def main():
    symbols = ['AAPL', 'MSFT']  # Example symbols
    all_data = []
    for symbol in symbols:
        data = fetch_data_from_db(symbol, engine)
        if data is not None:
            data = preprocess_data(data)
            data = feature_engineering(data)
            all_data.append(data)
            X = data[['SMA_20', 'EMA_20', 'Volatility', 'SMA_50', 'RSI', 'MACD', 'MACD_signal']]
            y = data['close']
            best_rf, X_test, y_test = train_model(X, y)
            evaluate_model(best_rf, X_test, y_test)
            news_headlines = "Some recent news headlines about the market."
            market_sentiment = get_market_sentiment(news_headlines)
            make_portfolio_decision(best_rf, X_test.iloc[-1:], market_sentiment)

    # Combine all data for correlation analysis
    combined_data = pd.concat(all_data, axis=0)
    correlation_matrix = discover_hidden_correlations(combined_data)
    print("Correlation Matrix:\n", correlation_matrix)

    # Example weights (to be replaced with actual portfolio weights)
    weights = np.array([0.5, 0.5])
    hhi = calculate_hhi(weights)
    print(f"Herfindahl-Hirschman Index (HHI): {hhi}")

    # Example returns and covariance matrix (to be replaced with actual data)
    example_returns = np.array([0.01, 0.02])
    example_cov_matrix = np.array([[0.1, 0.05], [0.05, 0.2]])
    optimized_weights = optimize_portfolio(example_returns, example_cov_matrix)
    print(f"Optimized Weights: {optimized_weights}")

if __name__ == "__main__":
    main()