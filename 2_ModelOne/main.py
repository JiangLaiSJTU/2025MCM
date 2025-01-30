from LSTM.model import train_and_predict

def main():
    # 指定数据文件路径
    file_path = '2025_Problem_C_Data/country.csv' 
    
    # 运行训练和预测
    train_and_predict(file_path)

if __name__ == "__main__":
    main()

