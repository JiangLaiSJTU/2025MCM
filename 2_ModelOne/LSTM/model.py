import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from LSTM.preprocess import load_and_preprocess_data

def create_sequences(data, time_steps=3):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

def train_and_predict(file_path):
    # 加载和预处理数据
    data = load_and_preprocess_data(file_path)

    # 获取所有国家的列表
    countries = data['NOC'].unique()

    # 创建时间序列数据
    X, y, scalers = [], [], []
    for country in countries:
        country_data = data[data['NOC'] == country]
        medals = country_data[['Gold', 'Silver', 'Bronze']].values
        years = country_data['Year'].values

        # 为每个国家单独进行归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        medals_scaled = scaler.fit_transform(medals)
        scalers.append(scaler)

        # 增加最近三届数据的权重
        recent_years = sorted(years)[-3:]  # 获取最近三届的年份
        weights = np.where(np.isin(years, recent_years), 3, 1)
        medals_scaled_weighted = medals_scaled * weights[:, np.newaxis]

        # 创建序列
        X_country, y_country = create_sequences(medals_scaled_weighted)
        if X_country.size > 0:  # 确保序列不为空
            X.append(X_country)
            y.append(y_country)

    # 将所有国家的数据合并成一个大的 X 和 y
    X = np.vstack(X)
    y = np.vstack(y)  # y 是二维数组，包含金牌数和奖牌数

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 构建 LSTM 模型
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(3))  # 预测金牌数、银牌数和铜牌数

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # 训练模型
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    # 预测
    y_pred = model.predict(X_test)

    # 反归一化预测值
    y_pred_unscaled = []
    y_test_unscaled = []
    for i in range(len(countries)):
        if i < len(y_pred):  # 确保索引不超出范围
            scaler = scalers[i]
            y_pred_unscaled.append(scaler.inverse_transform([y_pred[i]]))
            y_test_unscaled.append(scaler.inverse_transform([y_test[i]]))

    y_pred_unscaled = np.vstack(y_pred_unscaled)
    y_test_unscaled = np.vstack(y_test_unscaled)

    # 保存预测结果到 data_2028.csv
    prediction_results = pd.DataFrame(columns=["NOC", "Predicted_Gold", "Predicted_Silver", "Predicted_Bronze"])
    for i, country in enumerate(countries):
        if i < len(y_pred_unscaled):  # 确保索引不超出范围
            predicted_gold = y_pred_unscaled[i, 0]
            predicted_silver = y_pred_unscaled[i, 1]
            predicted_bronze = y_pred_unscaled[i, 2]
            prediction_results = pd.concat([prediction_results, pd.DataFrame([{
                "NOC": country,
                "Predicted_Gold": predicted_gold,
                "Predicted_Silver": predicted_silver,
                "Predicted_Bronze": predicted_bronze
            }])], ignore_index=True)

    # 剔除不再存在的队伍
    prediction_results = prediction_results[~prediction_results['NOC'].isin(['Russia', 'East Germany', 'West Germany', 'Soviet Union', 'Unified Team'])]

    # 将预测结果保存到 data_2028.csv 文件
    prediction_results.to_csv('data_2028.csv', index=False)

    # 绘制柱状图
    plt.figure(figsize=(12, 8))
    plt.bar(prediction_results['NOC'], prediction_results['Predicted_Gold'], label='Gold')
    plt.bar(prediction_results['NOC'], prediction_results['Predicted_Silver'], bottom=prediction_results['Predicted_Gold'], label='Silver')
    plt.bar(prediction_results['NOC'], prediction_results['Predicted_Bronze'], bottom=prediction_results['Predicted_Gold'] + prediction_results['Predicted_Silver'], label='Bronze')
    plt.xlabel('Country')
    plt.ylabel('Predicted Medals')
    plt.title('Predicted Medals for 2028 Olympics')
    plt.xticks(rotation=90)  # 横坐标标签旋转
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 打印一些预测结果以验证
    print(prediction_results.head())