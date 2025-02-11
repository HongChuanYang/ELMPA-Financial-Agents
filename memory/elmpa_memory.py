import json
import os 
from utils.emb_generation import generate_embedding
# from utils.text_generation import generate_embedding

from utils.utils import calculate_cosine_similarity
import numpy as np
import heapq

class AnalystMemory: 
    def __init__(self, agent_mem_path):
        self.agent_mem_path = agent_mem_path
        self.embeddings_file_path = os.path.join(self.agent_mem_path, "embeddings.json")
        self.mem_file_path = os.path.join(self.agent_mem_path, "memory.json")
        self.load()
        
    def load(self):
        if os.path.exists(self.mem_file_path):
            with open(self.mem_file_path, 'r') as f:
                self.mem = json.load(f)
        else:
            self.mem ={}
        self.embeddings = json.load(open(self.embeddings_file_path)) if os.path.exists(self.embeddings_file_path) else {}
        
    def save(self):
        os.makedirs(self.agent_mem_path, exist_ok=True)
        with open(os.path.join(self.mem_file_path), "w") as outfile:
            json.dump(self.mem, outfile)
        with open(os.path.join(self.embeddings_file_path), "w") as outfile:
            json.dump(self.embeddings, outfile)
    
    def extract(self, introduction):
        # 检索记忆，依据评分返回相关文章
        # 市场反馈，d_score,c_score,s_score
        if self.mem:
            # best_score = 0
            top_k = 3
            scores_with_memory = []
            for mem_id, memory in self.mem.items():
                query_embedding = self.get_embedding(introduction)
                memory_embedding = self.get_embedding(memory['report'])
                similarity = calculate_cosine_similarity(query_embedding, memory_embedding)
                relevance_score = similarity * 10 
                recency_score = 10*(0.995**(len(self.mem) - memory['mem_id']))
                poignancy_score = memory['weighted_score']
                overall_score = 0.2*recency_score + 0.6*poignancy_score + 0.2*relevance_score
                scores_with_memory.append((overall_score, memory['report']))
            top_k_memories = heapq.nlargest(top_k, scores_with_memory, key=lambda x: x[0])
            top_k_reports = [memory for _, memory in top_k_memories]
            relevant_mem = '\n'.join(top_k_reports)     
        else:
            relevant_mem = ''
        return relevant_mem
    

    
    def update_memory(self, report):
        # Update the analyst memory based on feedback d_score, c_score, m_score
        mem_count = len(self.mem) + 1             
        mem_id = f"mem_{str(mem_count)}"
        self.mem[mem_id] = {'mem_id':mem_count, 'symbol':report['symbol'], 'industry':report['information']['industry'], 'report':report['report'], 'weighted_score':report['weighted_score']}
        query_embedding = self.get_embedding(report['report'])
        
    def init_update_memory(self, memory, parent_mem_emb):
        # Update the analyst memory based on feedback d_score, c_score, m_score     
            # 检查 report 是否已存在于记忆中
        for existing_memory in self.mem.values():
            if existing_memory['report'] == memory['report']:
                return  # 如果报告已存在，直接跳过 
        mem_id = f"mem_{str(memory['mem_id'] )}"
        self.mem[mem_id] = memory
        try:
            self.embeddings[memory['report']] = parent_mem_emb[memory['report']]
        except:
            embedding = self.get_embedding(memory['report'])

        
    def get_embedding(self, text):
        try:
            # 尝试从字典中获取嵌入
            embedding = self.embeddings[text]
        except:
            # 如果抛出 KeyError 异常，说明字典中没有该键，生成一个新的嵌入
            embedding = generate_embedding(text)
            self.embeddings[text] = embedding  # 将新生成的嵌入保存到字典中
        return embedding   
    
    def get_all_memory(self):
        # 返回所有记忆内容，便于适应度评估
        return list(self.mem.values()), self.embeddings
        
        
        
        
        
        
    
    
    

class DirectorMemory: 
    def __init__(self, agent_mem_path):
        self.agent_mem_path = agent_mem_path
        self.employee_mem_path = os.path.join(self.agent_mem_path, "employee.json")
        # self.mem = None
        self.load()
    
    def load(self):
        if os.path.exists(self.employee_mem_path):
            with open(self.employee_mem_path, 'r') as f:
                self.mem = json.load(f)
        else:
            self.mem ={}
            
    def save(self):
        directory = os.path.dirname(self.employee_mem_path)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(self.employee_mem_path), "w") as outfile:
            json.dump(self.mem, outfile)
                
    def get_agent_scores(self, name):
        # 检索记忆，返回agent评分
        top_k = 3
        try:
            last_agent_mem = self.get_last_memory(name, top_k)
            mem_agent_scores = 0
            for mem in last_agent_mem:
                mem_agent_scores += mem['weighted_score']
            mem_agent_scores = mem_agent_scores/top_k
        except:
            mem_agent_scores = 5
        return mem_agent_scores
    
    def update_memory(self, reports):
        # Update the director's memory based on feedback d_score, c_score, m_score
        for report in reports: 
            if report['a_name'] not in self.mem:
                self.mem[report['a_name']] = {}
            mem_count = len(self.mem[report['a_name']]) + 1
            mem_id = f"mem_{str(mem_count)}"
            if report['best_report']:
                weighted_score = 0.2*report['d_score'] + 0.3*report['c_score'] + 0.5*report['m_score']
            else:
                weighted_score = 0.3*report['d_score'] + 0.7*report['m_score']
            self.mem[report['a_name']][mem_id] = {'mem_id':mem_count, 'symbol':report['symbol'], 'report':report['report'], 'weighted_score':weighted_score}
        # self.mem
        self.save()
        
    def get_last_memory(self, name, count):
        #根据输出的mem_type，找出node.type一致的，node_id最后的一个node
        agent_mem = self.mem[name]
        sorted_memories = sorted(agent_mem.values(), key=lambda x: x['mem_id'], reverse=True)
        return sorted_memories[:count]
