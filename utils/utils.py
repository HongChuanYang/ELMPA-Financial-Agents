import os
from datetime import date, timedelta, datetime
import json
import re
import math
import numpy as np
import random
import shutil


def decorate_all_methods(decorator):
    def class_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator

def get_next_weekday(date):
    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        next_weekday = date + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date
    
def save_staff(company_path, agents):
    os.makedirs(company_path, exist_ok=True)
    company_file_path = os.path.join(company_path, 'company_staff.json')
    with open(company_file_path, 'w') as json_file:
        json.dump(agents, json_file, indent=4)

    
def load_processed_data(data_path):
    with open(data_path, 'r') as f:
        all_data = json.load(f)
    return all_data

def split_train_test(data_dict, train_ratio=0.8):
    """
    根据给定比例划分训练集和测试集。

    参数：
    - data_dict: 待划分的数据字典。
    - train_ratio: 训练集占总数据的比例（默认值为 0.8）。

    返回：
    - train_data: 训练集字典。
    - test_data: 测试集字典。
    """
    # 获取所有的键
    keys = list(data_dict.keys())
    
    # 随机打乱键的顺序
    random.shuffle(keys)
    
    # 计算训练集的大小
    train_size = int(len(keys) * train_ratio)
    
    # 按比例划分为训练集和测试集
    train_keys = keys[:train_size]
    test_keys = keys[train_size:]
    
    # 构建训练集和测试集字典
    train_data = {key: data_dict[key] for key in train_keys}
    test_data = {key: data_dict[key] for key in test_keys}
    
    return train_data, test_data

def split_fix_train_test(data_dict, test_size):
    """
    根据给定比例划分训练集和测试集。

    参数：
    - data_dict: 待划分的数据字典。
    - train_ratio: 训练集占总数据的比例（默认值为 0.8）。

    返回：
    - train_data: 训练集字典。
    - test_data: 测试集字典。
    """
    # 获取所有的键
    keys = list(data_dict.keys())
    
    # 随机打乱键的顺序
    random.shuffle(keys)
    
    # 按比例划分为训练集和测试集
    test_keys= keys[:test_size]
    train_keys = keys[test_size:]
    
    # 构建训练集和测试集字典
    train_data = {key: data_dict[key] for key in train_keys}
    test_data = {key: data_dict[key] for key in test_keys}
    
    return train_data, test_data

def split_data(processed_data_path, train_ratio=0.9):
    all_data = load_processed_data(processed_data_path)
    train_data, test_data = split_train_test(all_data, train_ratio)
    return train_data, test_data



def prompt_generate(prompt_input, prompt_path):
    with open(prompt_path, "r") as f:
        prompt = f.read()
    for count, i in enumerate(prompt_input):   
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    return prompt

def parse_d_evaluate_output(response):
    # Normalize the response to ignore case and spaces
    response_normalized = re.sub(r'\s+', '', response).lower()

    # Define patterns for each report score
    report_a_pattern = r'"report_a_score":(\d+)'
    report_b_pattern = r'"report_b_score":(\d+)'
    report_c_pattern = r'"report_c_score":(\d+)'
    
    # Search for each pattern individually
    report_a_match = re.search(report_a_pattern, response_normalized)
    report_b_match = re.search(report_b_pattern, response_normalized)
    report_c_match = re.search(report_c_pattern, response_normalized)
    
    # Collect the found scores
    scores = {}
    if report_a_match:
        scores["Report_A_score"] = int(report_a_match.group(1))
    if report_b_match:
        scores["Report_B_score"] = int(report_b_match.group(1))
    if report_c_match:
        scores["Report_C_score"] = int(report_c_match.group(1))

    # If all scores are found, return them
    if len(scores) == 3:
        return scores
    
    # If any score is missing, look for the first three digits as a fallback
    numbers = re.findall(r'\d+', response)
    if len(numbers) >= 3:
        return {
            "Report_A_score": int(numbers[0]),
            "Report_B_score": int(numbers[1]),
            "Report_C_score": int(numbers[2])
        }
    
    # Default to a score of 5 for each report if all else fails
    return {
        "Report_A_score": 5,
        "Report_B_score": 5,
        "Report_C_score": 5
    }
    
def single_parse_d_evaluate_output(response):
    # Normalize the response to ignore case and spaces
    response_normalized = re.sub(r'\s+', '', response).lower()

    # Define the pattern for the Report_A score
    report_a_pattern = r'"report_a_score":(\d+)'
    
    # Search for the Report_A pattern
    report_a_match = re.search(report_a_pattern, response_normalized)
    
    # Collect the found score
    if report_a_match:
        return {"Report_A_score": int(report_a_match.group(1))}
    
    # If the score is not found, look for the first digit as a fallback
    numbers = re.findall(r'\d+', response)
    if len(numbers) >= 1:
        return {"Report_A_score": int(numbers[0])}
    
    # Default to a score of 5 if all else fails
    return {"Report_A_score": 5}


def parse_c_evaluate_output(response):
        # Attempt to find the score using the exact "score": <value> pattern
    score_pattern = r'"score"\s*:\s*(\d+)'  # Match the "score": <value> pattern, case insensitive
    match = re.search(score_pattern, response, re.IGNORECASE)
    
    if match:
        return int(match.group(1))
    
    # If the exact pattern fails, search for the first number between 0 and 10
    number_pattern = r'\b([0-9]|10)\b'  # Match any number between 0 and 10
    match = re.search(number_pattern, response)
    
    if match:
        return int(match.group(1))
    
    # If all else fails, return the default score of 5
    return 5

def parse_prediction(prediction_text):
    # 使用正则表达式匹配涨跌预测和幅度范围
    direction_match = re.search(r'\b(up|down)\b', prediction_text, re.IGNORECASE)
    if direction_match:
        direction = direction_match.group(1).lower()
    else:
        direction = None
    try:
        numbers = re.findall(r'[\d.]+', prediction_text)
        if len(numbers) >= 2:
            change_min = float(numbers[0])
            change_max = float(numbers[1])
            return direction, change_min, change_max
        elif len(numbers) == 1:
            change = float(numbers[0])
            return direction, change, change  # 返回相同的值作为最小和最大值
        else:
            return direction, 0, 0
    except:
        return direction, 0, 0
    


        
        

def evaluate_prediction_accuracy(direction, predicted_change_min, predicted_change_max, actual_change):
    """
    Evaluate the prediction accuracy based on direction and range.

    :param direction: Predicted direction ("up", "down", or neutral).
    :param predicted_change_min: Minimum predicted percentage change.
    :param predicted_change_max: Maximum predicted percentage change.
    :param actual_change: Actual percentage change.
    :return: Score (0-10) based on prediction accuracy.
    """
    # 判断预测方向是否正确
    direction_correct = (
        (direction == "up" and actual_change > 0) or
        (direction == "down" and actual_change < 0) or
        (direction not in ["up", "down"] and abs(actual_change) < 2)
    )

    # 如果方向错误，返回 0 到 5 分
    if not direction_correct:
        # 偏离程度评分（越接近正确方向分数越高）
        deviation_penalty = min(5, math.ceil(abs(actual_change) / 5))
        return max(5 - deviation_penalty, 0)

    # 初始化基础评分为 5 分
    base_score = 5

    # 如果方向正确，根据预测范围计算评分
    if predicted_change_min <= abs(actual_change) <= predicted_change_max:
        # 实际变化值在预测范围内，满分 10 分
        return 10
    else:
        # 实际变化值在预测范围外，根据偏离程度递减评分
        if abs(actual_change) < predicted_change_min:
            # 实际变化小于预测范围
            deviation = predicted_change_min - abs(actual_change)
        else:
            # 实际变化大于预测范围
            deviation = abs(actual_change) - predicted_change_max

        # 每偏离 2%，减 1 分，最低为 5 分
        penalty = math.ceil(deviation / 2)
        return max(10 - penalty, base_score)

        
        
        

def market_response(report, information):
    # 检索报告中的预测股价涨跌，与实际涨跌对比得出市场反馈分。
    test_report = '{\n  "Positive Developments": "Agilent Technologies has shown strong recent performance with a 20.05% price increase over the last 30 days and has surpassed the average analyst target price of $132.36. Wolfe Research\'s \'Outperform\' recommendation and positive reports from Grand View Research and Mordor Intelligence highlight growth potential in key markets. Additionally, strategic acquisitions and alliances have bolstered growth and market positioning.",\n  "Potential Concerns": "Revenue growth is negative at -6.38% year-over-year, indicating potential issues with sales or market demand. Despite a robust EPS and margins, the high valuation could lead to volatility if the stock price adjusts. The lack of available data on debt-to-equity ratio may raise concerns about financial stability.",\n  "Forecast and Analysis": {\n    "Prediction": "up by 1-2%",\n    "Analysis": "Given the recent strong performance and positive analyst outlook, the stock is likely to continue a modest upward trend. However, potential concerns such as negative revenue growth and valuation risks might temper the gains, leading to a more modest increase in the short term."\n  }\n}'
    # report = test_report
    price_change = information['price_change']
    prediction_match = re.search(r'"Prediction":\s*"([^"]+)"', report, re.IGNORECASE)
    if prediction_match:
        prediction = prediction_match.group(1)
        direction, predicted_change_min, predicted_change_max = parse_prediction(prediction)
        m_score = evaluate_prediction_accuracy(direction, predicted_change_min, predicted_change_max, price_change)
    else:
        m_score = 5
    return m_score



def calculate_cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm_a = np.linalg.norm(embedding1)
    norm_b = np.linalg.norm(embedding2)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def clear_folder(folder_path):
    """
    Clear all files and subfolders in the specified folder.
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove subfolder
    else:
        os.makedirs(folder_path)  # Create folder if it doesn't exist





