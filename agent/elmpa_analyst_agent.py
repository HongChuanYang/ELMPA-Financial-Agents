# from utils.text_generation import generate
from utils.text_generation_deepseek import generate
from utils.utils import prompt_generate, parse_c_evaluate_output
import os
from memory.elmpa_memory import AnalystMemory
import random
from config.company import create_analyst
import uuid


class EvolutionaryAgent:
    def __init__(self, employe, model_name):
        if "_" in employe['name']:
            self.name = employe['name']
        else:
            self.name = employe['name'] + f"_{uuid.uuid4().hex[:8]}"
        self.sector = employe['sector']
        self.personality = employe['personality']
        self.industry_inclination = employe['industry_inclination']
        agent_mem_path = os.path.join('memory/analyst_memory', model_name, self.sector, self.name)
        self.mem = AnalystMemory(agent_mem_path)
        self.model_name = model_name
        self.rating = employe['ratings']  # Initial rating for evolutionary processes
        self.generation = employe['generations']  # Initial rating for evolutionary processes

        
    def save(self):
        pass

    def generate_report(self, symbol, information):
        prompt_path = "prompt/analyst.txt"
        Exemplary = self.mem.extract(information['introduction'])
        prompt_input = [self.personality, information['introduction'], " ".join(information['news_summary']), information['stock_price_summary'], information['financials'], Exemplary, information['analysis_period']]
        total_prompt = prompt_generate(prompt_input, prompt_path)
        response, token = generate(total_prompt)
        return response, token

    def update_memory(self, report):
        # Update memory based on feedback
        self.mem.update_memory(report)

    def self_evaluate(self, report, symbol, information):
        prompt = (
            'You are an experienced stock market analyst with the following profile: '
            'Name: "{}". Sector Expertise: "{}". Personality: "{}".'
            'You have just completed a securities analysis report for the company "{}" with the stock symbol "{}".'
            'The company introduction is: "{}".\n Your report stated: \n"{}".\n'
            'Your task is to critically evaluate your report based on its clarity, accuracy, and thoroughness. '
            'If the report is not perfect or misses important aspects, rate it strictly below 5. '
            'If the report is exceptional, provide a higher score up to a maximum of 10. '
            'Provide a single score in JSON format like: {{"score": X}} where X is your rating. '
        ).format(
            self.name,
            self.sector,
            self.personality,
            information['name'],
            symbol,
            information['introduction'],
            report
        )     
        llm_output, token  = generate(prompt)   
        s_score = parse_c_evaluate_output(llm_output)
        return s_score, token

    def crossover(self, other_agent):
        # Perform crossover between two agents to create a new agent
        # 1. Combine personality
        traits1 = set(self.personality.split(', '))
        traits2 = set(other_agent.personality.split(', '))

        # 随机选择特质组合
        combined_traits = traits1.union(traits2)  # 将两个父母的特质合并
        selected_traits = random.sample(combined_traits, min(len(combined_traits), max(len(traits1), len(traits2))))
        # 生成新的个性字符串
        child_personality = ', '.join(selected_traits)
        
                
        # 2. Merge memory
        mem_list1, mem_emb1 = self.mem.get_all_memory()  # 获取父代1的记忆
        mem_list2, mem_emb2 = other_agent.mem.get_all_memory()  # 获取父代2的记忆
        parent_mem_emb = mem_emb1 | mem_emb2

        # 合并记忆并按 weighted_score 排序
        # combined_memory = mem_list1 + mem_list2
        combined_memory = {memory['report']: memory for memory in mem_list1 + mem_list2}.values() # 去重
        sorted_memory = sorted(combined_memory, key=lambda x: x['weighted_score'], reverse=True)[:100]  # 只保留前100条
        sorted_memory_by_mem_id = sorted(sorted_memory, key=lambda x: x['mem_id'])



        # 3. Combine industry inclination        
        child_industry_inclination = {}
        industries = set(self.industry_inclination.keys()).union(other_agent.industry_inclination.keys())
        for industry in industries:
            score1 = self.industry_inclination.get(industry, 0)
            score2 = other_agent.industry_inclination.get(industry, 0)
            # 平均融合
            child_industry_inclination[industry] = 0.5 * score1 + 0.5 * score2
            
        child_staff = create_analyst([self.name, other_agent.name])
        child_staff['sector'] = 'general'
        child_staff['ratings'] = 0.5 * self.rating + 0.5 * other_agent.rating 
        child_staff['generations'] = self.generation + 1        
        child_staff['industry_inclination'] = child_industry_inclination
        child_staff['personality'] = child_personality
        child_agent = EvolutionaryAgent(child_staff, self.model_name)
        
        # 保存融合后的记忆到子代的记忆库
        for idx, memory in enumerate(sorted_memory_by_mem_id, start=1):
            # 更新 mem_id 和保存嵌入
            memory['mem_id'] = idx
            child_agent.mem.init_update_memory(memory, parent_mem_emb)
        return child_agent


    def mutate(self, mutation_rate=0.1, mutation_stddev=0.05):
        """
        Apply mutation to the agent's attributes (personality, memory, and industry inclination).
        :param mutation_rate: Probability of mutation for each trait.
        :param mutation_stddev: Standard deviation for Gaussian noise applied to numerical traits.
        """
        # 1. Mutate personality traits
        personality_pool = [
            "meticulous", "detail-oriented", "analytical", "logical", "intuitive", "perceptive",
            "curious", "open-minded", "objective", "unbiased", "thoughtful", "reflective",
            "proactive", "resourceful", "patient", "methodical", "innovative", "creative",
            "assertive", "confident", "collaborative", "cooperative", "adaptable", "versatile",
            "organized", "efficient", "energetic", "enthusiastic", "calm", "composed"
        ]
        current_traits = self.personality.split(", ")

        # 遍历当前个性特征
        for i, trait in enumerate(current_traits):
            if random.random() < mutation_rate:  # 以一定概率触发变异
                # 从 personality_pool 中选择一个与当前特质不同的新特质
                available_traits = list(set(personality_pool) - set(current_traits))
                if available_traits:
                    new_trait = random.choice(available_traits)
                    current_traits[i] = new_trait

        # 更新个性为新的特质组合
        self.personality = ", ".join(current_traits)

        # 2. Mutate memory weights
        for mem_id, memory in self.mem.mem.items():
            if random.random() < mutation_rate:  # 以一定概率修改记忆的权重
                memory['weighted_score'] += random.gauss(0, mutation_stddev)
                # 约束 weighted_score 在合理范围（如 [0, 10]）
                memory['weighted_score'] = max(0, min(10, memory['weighted_score']))
        # self.mem.save()  # 保存变异后的记忆

        # 3. Mutate industry inclination
        for industry in self.industry_inclination:
            if random.random() < mutation_rate:  # 以一定概率触发变异
                self.industry_inclination[industry] += random.gauss(0, mutation_stddev)
                # 保证分布的归一化
        total = sum(self.industry_inclination.values())
        self.industry_inclination = {k: v / total for k, v in self.industry_inclination.items()}

            
    def compute_industry_inclination(self):
        # 初始化行业评分字典
        mem_list, mem_emb = self.mem.get_all_memory()
        industries = list(set([item["industry"] for item in mem_list]))
        industry_scores = {industry: 0.0 for industry in industries}
        
        # 累加每个行业的评分
        for entry in mem_list:
            industry = entry["industry"]
            score = entry["weighted_score"]
            industry_scores[industry] += score
            
        # 归一化概率分布
        total_score = sum(industry_scores.values())
        industry_distribution = {industry: score / total_score for industry, score in industry_scores.items()}

        # 计算总评分用于归一化
        scores = [entry["weighted_score"] for entry in mem_list]
        mean_score = sum(scores) / len(scores) if scores else 0
        std_dev = (sum((x - mean_score) ** 2 for x in scores) / len(scores)) ** 0.5 if scores else 0
        stability_score = 1 - (std_dev / mean_score) if mean_score > 0 else 0


        
        return industry_distribution, total_score
    
    def compute_industry_inclination(self):
        # 获取所有记忆条目
        # mem_list = None
        mem_list, mem_emb = self.mem.get_all_memory()
        # 如果记忆库为空，返回默认值
        if not mem_list:
            return 0.0, {}

        # 初始化行业评分字典和计数字典
        industries = list(set([item["industry"] for item in mem_list]))
        industry_scores = {industry: 0.0 for industry in industries}
        industry_counts = {industry: 0 for industry in industries}

        # 累加每个行业的评分和计数
        for entry in mem_list:
            industry = entry["industry"]
            score = entry["weighted_score"]
            industry_scores[industry] += score
            industry_counts[industry] += 1

        # 计算总分和归一化分布
        total_score = sum(industry_scores.values())
        industry_distribution = {industry: score / total_score for industry, score in industry_scores.items()}

        # 计算每个行业的平均分
        industry_avg_scores = {industry: industry_scores[industry] / industry_counts[industry]
                            for industry in industries}

        return industry_avg_scores, industry_distribution

    def evaluate_fitness(self, generation):
        self.generation = generation
        # Evaluate fitness based on memory quality and diversity
        industry_avg_scores, industry_distribution = self.compute_industry_inclination()
        
        # 获取稳定性评分
        mem_list, mem_emb = self.mem.get_all_memory()
        scores = [entry["weighted_score"] for entry in mem_list]
        mean_score = sum(scores) / len(scores) if scores else 0
        std_dev = (sum((x - mean_score) ** 2 for x in scores) / len(scores)) ** 0.5 if scores else 0
        stability_score = 1 - (std_dev / mean_score) if mean_score > 0 else 0
        
        # 获取领域特长评分
        domain_expertise = sum(industry_distribution[industry] * industry_avg_scores[industry]
                            for industry in industry_avg_scores.keys())

        # 综合评分
        alpha, beta, gamma = 0.4, 0.3, 0.3
        overall_rating = (alpha * mean_score) + (beta * stability_score) + (gamma * domain_expertise)

        # 更新代理的适应度评分
        self.industry_inclination = industry_distribution
        self.rating = overall_rating
