from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(df):
    df = df.dropna(subset=['leverage', 'size', 'closedpnl'])

    if len(df) < 10:
        return None, 0  # Not enough data

    X = df[['leverage', 'size']]
    y = df['is_profit']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy