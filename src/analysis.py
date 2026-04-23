def sentiment_performance(df):
    return df.groupby('classification')['closedpnl'].agg(['mean', 'count'])


def win_rate(df):
    return df.groupby('classification')['is_profit'].mean()


def leverage_analysis(df):
    return df.groupby('classification')['leverage'].mean()


def size_analysis(df):
    return df.groupby('classification')['abs_size'].mean()


def leverage_risk(df):
    return df.groupby('leverage_group')['closedpnl'].mean()


def hourly_performance(df):
    return df.groupby(['hour', 'classification'])['closedpnl'].mean().unstack()