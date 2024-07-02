import os
import pandas as pd
import requests
from sqlalchemy import create_engine
from dotenv import load_dotenv
from io import StringIO

# Load environment variables
load_dotenv()

# Set up database connection
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

# Function to fetch data from Nasdaq Data Link
def fetch_nasdaq_data(api_key, symbol):
    url = f'https://data.nasdaq.com/api/v3/datasets/WIKI/{symbol}.csv?api_key={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = pd.read_csv(StringIO(response.text))
        return data
    else:
        print(f"Failed to fetch data for {symbol} from Nasdaq Data Link")
        return None

# Function to fetch data from Alpha Vantage
def fetch_alpha_vantage_data(api_key, symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full&datatype=csv'
    response = requests.get(url)
    if response.status_code == 200:
        data = pd.read_csv(StringIO(response.text))
        return data
    else:
        print(f"Failed to fetch data for {symbol} from Alpha Vantage")
        return None

# Function to save data to PostgreSQL
def save_to_db(data, table_name, engine):
    data.to_sql(table_name, engine, if_exists='append', index=False)
    print(f"Data saved to {table_name} table")

def main():
    nasdaq_api_key = os.getenv('NASDAQ_API_KEY')
    alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

    symbols = ['AAPL', 'MSFT']  # Add more symbols as needed

    for symbol in symbols:
        # Fetch data from Nasdaq Data Link
        nasdaq_data = fetch_nasdaq_data(nasdaq_api_key, symbol)
        if nasdaq_data is not None:
            nasdaq_data = nasdaq_data.rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high',
                'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
            nasdaq_data['symbol'] = symbol
            save_to_db(nasdaq_data, 'stock_prices', engine)

        # Fetch data from Alpha Vantage
        alpha_vantage_data = fetch_alpha_vantage_data(alpha_vantage_api_key, symbol)
        if alpha_vantage_data is not None:
            alpha_vantage_data = alpha_vantage_data.rename(columns={
                'timestamp': 'date', 'open': 'open', 'high': 'high',
                'low': 'low', 'close': 'close', 'volume': 'volume'
            })
            alpha_vantage_data['symbol'] = symbol
            save_to_db(alpha_vantage_data, 'stock_prices', engine)

if __name__ == "__main__":
    main()