import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ast  # for string → dict conversion

def extract_cpu(value):
    try:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return value.get('cpus', 0)
    except:
        return 0

def extract_memory(value):
    try:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return value.get('memory', 0)
    except:
        return 0

def preprocess_google_data(input_path, output_path):
    df = pd.read_csv(input_path)

    print("Original Data Shape:", df.shape)

    # Extract values from complex columns
    df['cpu'] = df['average_usage'].apply(extract_cpu)
    df['memory'] = df['average_usage'].apply(extract_memory)
    df['max_usage'] = df['maximum_usage'].apply(extract_cpu)

    # Keep only required columns
    df = df[['time', 'cpu', 'memory', 'max_usage', 'failed']]

    # Rename
    df.columns = ['time', 'cpu', 'memory', 'max_usage', 'fault']

    # Handle missing values (updated method)
    df.ffill(inplace=True)

    # Normalize
    scaler = MinMaxScaler()
    df[['cpu','memory','max_usage']] = scaler.fit_transform(
        df[['cpu','memory','max_usage']]
    )

    # Save
    df.to_csv(output_path, index=False)

    print(" Cleaned data saved at:", output_path)
    print(" Cleaned Data Shape:", df.shape)

    return df