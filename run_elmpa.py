import json 
from utils.utils import market_response, clear_folder, save_staff, split_data
from agent.elmpa_analyst_agent import EvolutionaryAgent
from agent.client_agent import ClientAgent
from config.company import create_analyst
import os 
import random
from datetime import datetime
from scipy.spatial.distance import cosine
from tqdm import tqdm
import time



def init_add_staff(company_path, agent_nums, agent_mem_path):
    """
    Initialize and add staff agents. Clears agent memory path before initialization.

    :param company_path: Path where agents' data will be saved.
    :param agent_nums: Number of agents to create.
    :param agent_mem_path: Path to the agent memory folder to be cleared.
    :return: List of initialized agents.
    """
    # Clear memory folder
    clear_folder(agent_mem_path)

    all_name_list = []
    agents = []
    for _ in range(agent_nums):
        employe = create_analyst(all_name_list)  # Avoid duplicate names
        employe['sector'] = 'general'  # Sector will be dynamically decided during training
        employe['ratings'] = 0
        employe['generations'] = 0
        employe['industry_inclination'] = {}
        all_name_list.append(employe['name'])
        agents.append(employe)

    save_staff(company_path, agents)
    return agents

def load_staff(company_path):
    company_file_path = os.path.join(company_path, 'company_staff.json')
    with open(company_file_path, 'r') as f:
        agents = json.load(f)
    return agents

def load_previous_state(result_path):
    if os.path.exists(result_path):
        with open(result_path, 'r') as json_file:
            saved_data = json.load(json_file)
        return saved_data
    else:
        return []

def crossover_agents(parent1, parent2):
    return parent1.crossover(parent2)

def eliminate_agents(agents):
    agents.sort(key=lambda agent: agent.evaluate_fitness())
    eliminated = agents[:len(agents) // 2]  # Eliminate the lowest half
    for agent in eliminated:
        print(f"Eliminated agent: {agent.name} with fitness: {agent.evaluate_fitness()}")
    return agents[len(eliminated):]


def update_company_staff(agents):
    """
    Generate a new company_staff list directly from the current agents.

    :param agents: List of EvolutionaryAgent objects after training.
    :return: New company_staff list.
    """
    company_staff = []
    for agent in agents:
        staff_entry = {
            "name": agent.name,
            "personality": agent.personality,
            "sector": agent.sector,
            "ratings": agent.rating,
            "generations": agent.generation,
            "industry_inclination": agent.industry_inclination,
        }
        company_staff.append(staff_entry)

    return company_staff



def evolutionary_selection(agents, num_selected):
    """
    Select top-performing agents based on their total scores.

    :param agents: List of agents with computed scores.
    :param num_selected: Number of agents to retain.
    :return: List of selected agents.
    """
    # Sort agents by total_score in descending order
    agents.sort(key=lambda agent: agent.rating, reverse=True)

    # Select top-performing agents
    return agents[:num_selected]

def compute_similarity(agent1, agent2):
    """
    Compute the similarity between two agents based on their industry inclination.
    :param agent1: First agent.
    :param agent2: Second agent.
    :return: Similarity score (e.g., cosine similarity).
    """
    # Extract industry inclinations as vectors
    industries = list(set(agent1.industry_inclination.keys()).union(agent2.industry_inclination.keys()))
    vec1 = [agent1.industry_inclination.get(ind, 0) for ind in industries]
    vec2 = [agent2.industry_inclination.get(ind, 0) for ind in industries]
    
    # Compute cosine similarity
    similarity = 1 - cosine(vec1, vec2)
    return similarity




def generate_offspring_by_traversal(selected_agents):
    """
    Generate offspring by allowing each agent to select the most similar mate,
    and producing two offspring per pair.
    :param selected_agents: List of agents selected for reproduction.
    :return: List of new agents (offspring).
    """
    new_agents = []
    paired = set()  # 记录已配对的智能体

    for agent in selected_agents:
        # 找到相似度最高但未被配对的智能体
        best_mate = None
        max_similarity = -float('inf')
        
        for candidate in selected_agents:
            if candidate == agent or candidate in paired:
                continue
            similarity = compute_similarity(agent, candidate)
            if similarity > max_similarity:
                max_similarity = similarity
                best_mate = candidate
        
        # 如果找到合适的配偶
        if best_mate:
            paired.add(agent)
            paired.add(best_mate)
            
            # 生成两个子代
            child1 = agent.crossover(best_mate)  # 子代 1
            child2 = best_mate.crossover(agent)  # 子代 2
            child1.mutate()
            child2.mutate()
            new_agents.extend([child1, child2])
    
    while len(new_agents) < len(selected_agents) * 2:
        agent = random.choice(selected_agents)
        mate = random.choice(selected_agents)
        if agent != mate:
            child1 = agent.crossover(mate)
            child2 = mate.crossover(agent)
            child1.mutate()
            child2.mutate()
            new_agents.extend([child1, child2])
    
    return new_agents







def main_loop(train_data, company_staff, result_path, temp_result_path, model_name, save_result, company_path, temp_fit_result_path, fit_result_path, save_fit_result, agent_mem_path):
    start_time = time.time()
    client_agent = ClientAgent()
    
    agents = [EvolutionaryAgent(employe, model_name) for employe in company_staff]
    

    generations = 20  
    mini_batch_size = 32  
    epsilon = 0.01  
    fitness_history = save_fit_result
    save_result = [] if not save_result else save_result
    init_generation = len(save_result) // mini_batch_size
    generations = generations - init_generation
    early_stop = 5
    total_token = 0

    for generation in tqdm(range(generations), desc="Training Progress", unit="generation"):
        generation_number = generation + init_generation + 1
        print(f"Generation {generation_number}")
        # Sample a mini-batch
        mini_batch = {symbol: train_data[symbol] for symbol in random.sample(list(train_data.keys()), mini_batch_size)}
        generation_token = 0
        
        for symbol, information in tqdm(mini_batch.items(), desc=f"Generation {generation_number} Samples", leave=False):
            information['industry']
            # Generate reports and update agents
            reports = []
            for agent in agents:
                report, r_token = agent.generate_report(symbol, information)
                s_score, s_token = agent.self_evaluate(report, symbol, information)  # Analyst's Self-Score
                m_score = market_response(report, information)  # Market's Score
                c_score, c_token = client_agent.evaluate(symbol, information, report)  # Client's Score

                generation_token += r_token + s_token + c_token
                total_token += r_token + s_token + c_token
                
                weighted_score = (0.2 * s_score + 
                                  0.5 * m_score + 
                                  0.3 * c_score)
                report = {'symbol': symbol, 'agent': agent.name, 'report': report, 's_score': s_score, 'm_score': m_score, 'c_score': c_score, 'weighted_score': weighted_score, 'information': information}
                agent.update_memory(report)
                reports.append(report)
            save_result.append(reports)

        # Evaluate fitness for each agent
        for agent in agents:
            agent.evaluate_fitness(generation_number)
            
        # 每代结束时更新总 token 消耗信息
        tqdm.write(f"Generation {generation_number} completed. Tokens used: {generation_token}. Total tokens: {total_token}")

        # Record fitness metrics
        avg_fitness = sum(agent.rating for agent in agents) / len(agents)
        max_fitness = max(agent.rating for agent in agents)
        fitness_list = [agent.rating for agent in agents]
        fitness_history.append({"generations":generation_number, "Average Fitness": avg_fitness, "Max Fitness": max_fitness, "Fitness List": fitness_list})
        
        

        print(f"Average Fitness: {avg_fitness}, Max Fitness: {max_fitness}")


        # Evolutionary selection and mutation
        selected_agents = evolutionary_selection(agents, len(agents) // 2)
        new_agents = generate_offspring_by_traversal(selected_agents)
        agents = new_agents

        # Update company staff with trained agents' information
        updated_company_staff = update_company_staff(agents)

        # Save updated company staff
        save_staff(company_path, updated_company_staff)
        temp_save_result = save_result.copy()
        # temp_save_result.insert(0, fitness_history) 
        with open(temp_result_path, 'w') as json_file:
            json.dump(temp_save_result, json_file, indent=4)
        with open(temp_fit_result_path, 'w') as json_file:
            json.dump(fitness_history, json_file, indent=4)
            
        # Clear memory folder and save new memory
        clear_folder(agent_mem_path)    
        for agent in agents:
            agent.mem.save()
        # 早停条件
        if len(fitness_history) > early_stop:
            recent_avg_improvement = abs(fitness_history[-1]["Average Fitness"] - fitness_history[-early_stop]["Average Fitness"])
            if recent_avg_improvement < epsilon:
                print(f"Convergence reached after {generation_number} generations.")
                break

    print(f"Training completed. Total tokens used: {total_token}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    with open(result_path, 'w') as json_file:
        json.dump(save_result, json_file, indent=4)
    with open(fit_result_path, 'w') as json_file:
        json.dump(fitness_history, json_file, indent=4)


def evaluate_agents_on_test_set(agents, test_data, result_path):
    """
    Evaluate trained agents on the test set.

    :param agents: List of trained agents.
    :param test_data: Test data for evaluation.
    :param result_path: Path to save evaluation results.
    """
    # 增加根据倾向选择agent的功能
    evaluation_results = []
    scores_summary = {'s_score': [], 'm_score': [], 'c_score': [], 'weighted_score': []}
    print("Evaluating agents on the test set...")
    total_token = 0
    
    for symbol, information in test_data.items():
        industry = information.get('industry', 'general')  # 获取数据点的行业信息
        suitable_agents = sorted(agents, key=lambda agent: agent.industry_inclination.get(industry, 0), reverse=True)
        top_agent = suitable_agents[0]  # 选择行业倾向最高的代理
        
        report, r_token = top_agent.generate_report(symbol, information)
        s_score, s_token = top_agent.self_evaluate(report, symbol, information)  # Analyst's Self-Score
        m_score = market_response(report, information)  # Market's Score
        c_score, c_token = ClientAgent().evaluate(symbol, information, report)  # Client's Score
        # 累计 token 消耗
        total_token += r_token + s_token + c_token
        weighted_score = 0.2 * s_score + 0.5 * m_score + 0.3 * c_score
        # 保存结果
        evaluation_results.append({
            'symbol': symbol,
            'industry': industry,
            'agent': top_agent.name,
            'report': report,
            's_score': s_score,
            'm_score': m_score,
            'c_score': c_score,
            'weighted_score': weighted_score
        })

                # 汇总评分数据
        scores_summary['s_score'].append(s_score)
        scores_summary['m_score'].append(m_score)
        scores_summary['c_score'].append(c_score)
        scores_summary['weighted_score'].append(weighted_score)
        
    # Save evaluation results to a JSON file
    evaluation_path = result_path.replace("report_results", "test_evaluation_results")
    with open(evaluation_path, 'w') as json_file:
        json.dump(evaluation_results, json_file, indent=4)

    print(f"Evaluation results saved to {evaluation_path}")






if __name__ == "__main__":
    model_name = 'elmpa'
    raw_data_path = f'data/processed_data.json'
    
    # load data
    train_data, test_data = split_data(raw_data_path, train_ratio=0.9)
    
        
    company_path = f'data/company/{model_name}'
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(f'result/{model_name}', exist_ok=True)
    result_path = f'result/{model_name}/report_results_{current_time}.json'
    fit_result_path = f'result/{model_name}/fitness_results_{current_time}.json'
    temp_result_path = f'result/{model_name}/temp_report_results.json'
    temp_fit_result_path = f'result/{model_name}/temp_fitness_results.json'
    

    
    # load training checkpoint
    save_result = load_previous_state(temp_result_path)
    save_fit_result = load_previous_state(temp_fit_result_path)
    
    
    # set staff
    agent_nums = 10 
    agent_mem_path = os.path.join('memory/analyst_memory', model_name)
    company_staff = load_staff(company_path) if save_result else init_add_staff(company_path, agent_nums, agent_mem_path)



    mode = "train" # test

    if mode == "train":
        main_loop(train_data, company_staff, result_path, temp_result_path, model_name, save_result, company_path, temp_fit_result_path, fit_result_path, save_fit_result, agent_mem_path)
    elif mode == "test":
        agents = [EvolutionaryAgent(employe, model_name=model_name) for employe in company_staff]
        evaluate_agents_on_test_set(agents, test_data, result_path)
    else:
        print("Invalid mode. Please enter 'train' or 'test'.")





