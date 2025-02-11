from utils.text_generation_deepseek import generate
from utils.utils import parse_c_evaluate_output





class ClientAgent:
    def __init__(self,):
        pass
    
    def evaluate(self, symbol, information, report):
        prompt = 'You are an agent who has been approached by a client for an evaluation of the securities analysis report. Assess the report for the company with the stock symbol {}. The company, "{}", is introduced as: "{}". The analysis report provided by the securities firm states: "{}". \nFrom the client\'s perspective, critically evaluate the report by examining its clarity, accuracy, and thoroughness. Assess whether the report is well-organized, easy to understand, and free of errors, whether it supports its conclusions with credible data, and if it comprehensively covers all relevant aspects of the company\'s performance and potential risks. Based on these criteria, assign a rating that reflects the overall quality and reliability of the report for a client\'s decision-making needs. You are to rate the report on a scale from 0 to 10 and provide a specific score in json format. Example Output: {{"score": 7}}'.format(symbol, information['name'], information['introduction'], report)
        llm_output, token  = generate(prompt)                
        c_score = parse_c_evaluate_output(llm_output)
        return c_score, token


    


    

