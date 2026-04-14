def create_features(data):
    # Rolling mean (simple feature)
    data['cpu_mean'] = data['cpu'].rolling(5).mean()
    data['memory_mean'] = data['memory'].rolling(5).mean()

    data.fillna(0, inplace=True)

    print("Feature engineering done")
    return data