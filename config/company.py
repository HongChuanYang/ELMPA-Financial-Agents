import random

company_sector = {
    "Communication Services": {
        "description": "Companies that provide communication services through fixed-line, cellular, wireless, high bandwidth and fiber optic cable network.",
        "employees": []
    },
    "Consumer Discretionary": {
        "description": "Comprises businesses that tend to be the most sensitive to economic cycles.",
        "employees": []
    },
    "Consumer Staples": {
        "description": "Industries within this sector are less sensitive to economic cycles.",
        "employees": []
    },
    "Energy": {
        "description": "Includes companies involved in the exploration, production, and energy supply.",
        "employees": []
    },
    "Financials": {
        "description": "Includes banks, investment funds, and insurance companies that provide financial services.",
        "employees": []
    },
    "Health Care": {
        "description": "Includes health care providers, medical goods, and services.",
        "employees": []
    },
    "Industrials": {
        "description": "Includes manufacturers and distributors of capital goods in addition to providers of services.",
        "employees": []
    },
    "Materials": {
        "description": "Includes companies that extract or process raw materials.",
        "employees": []
    },
    "Real Estate": {
        "description": "Includes companies involved in real estate development and operations.",
        "employees": []
    },
    "Information Technology": {
        "description": "Includes companies that produce software, hardware or semiconductor equipment, and related services.",
        "employees": []
    },
    "Utilities": {
        "description": "Includes utility companies such as electric, gas and water firms.",
        "employees": []
    },
    "General": {
        "description": "Handles miscellaneous and undetermined sector reports that do not fit into other specific categories.",
        "employees": []
    }
}


def create_analyst(all_name_list):
    # 扩展到50个的随机名字列表
    names = [
        "Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah", "Ian", "Julia", "Kevin",
        "Liam", "Mia", "Noah", "Olivia", "Peyton", "Quinn", "Ruby", "Sophie", "Tyler", "Uma", "Victor",
        "Wendy", "Xavier", "Yara", "Zach", "Amelia", "Brian", "Cora", "David", "Eleanor", "Frank", "Grace",
        "Henry", "Isabella", "Jack", "Kylie", "Lucas", "Molly", "Nathan", "Ophelia", "Patrick", "Quincy",
        "Rosalie", "Samuel", "Tina", "Ulysses", "Vera", "William", "Xena"
    ]

    # 基础性格特征列表
    basic_personalities = [
        "meticulous", "detail-oriented", "analytical", "logical", "intuitive", "perceptive",
        "curious", "open-minded", "objective", "unbiased", "thoughtful", "reflective",
        "proactive", "resourceful", "patient", "methodical", "innovative", "creative",
        "assertive", "confident", "collaborative", "cooperative", "adaptable", "versatile",
        "organized", "efficient", "energetic", "enthusiastic", "calm", "composed"
    ]

    names = [name for name in names if name not in all_name_list]

    # 随机选择名字
    name = random.choice(names)
    # 随机组合两个或三个不同的基础性格特征
    personality_traits = random.sample(basic_personalities, k=random.randint(3, 5))
    personality = ", ".join(personality_traits)

    return {
        "name": name,
        "personality": personality
    }

industry_to_sector = {
    'Life Sciences Tools & Services': 'Health Care',
    'Metals & Mining': 'Materials',
    'Diversified Consumer Services': 'Consumer Discretionary',
    'Airlines': 'Industrials',
    'Real Estate': 'Real Estate',
    'Insurance': 'Financials',
    'Retail': 'Consumer Discretionary',
    'Communications': 'Communication Services',
    'Building': 'Industrials',
    'Technology': 'Information Technology',
    'Financial Services': 'Financials',
    'Biotechnology': 'Health Care',
    'Banking': 'Financials',
    'Beverages': 'Consumer Staples',
    'Commercial Services & Supplies': 'Industrials',
    'Health Care': 'Health Care',
    'Construction': 'Industrials',
    'Pharmaceuticals': 'Health Care',
    'Hotels, Restaurants & Leisure': 'Consumer Discretionary',
    'Semiconductors': 'Information Technology',
    'N/A': 'General',  # 无法确定的类型
    'Food Products': 'Consumer Staples',
    'Auto Components': 'Consumer Discretionary',
    'Professional Services': 'Industrials',
    'Energy': 'Energy',
    'Utilities': 'Utilities',
    'Electrical Equipment': 'Industrials',
    'Trading Companies & Distributors': 'Industrials',
    'Machinery': 'Industrials',
    'Aerospace & Defense': 'Industrials',
    'Logistics & Transportation': 'Industrials',
    'Chemicals': 'Materials',
    'Media': 'Communication Services',
    'Distributors': 'Industrials',
    'Telecommunication': 'Communication Services',
    'Road & Rail': 'Industrials',
    'Packaging': 'Materials',
    'Automobiles': 'Consumer Discretionary',
    'Industrial Conglomerates': 'Industrials',
    'Leisure Products': 'Consumer Discretionary',
    'Consumer products': 'Consumer Discretionary',
    'Tobacco': 'Consumer Staples',
    'Transportation Infrastructure': 'Industrials',
    'Paper & Forest': 'Materials',
    'Marine': 'Industrials',
    'Textiles, Apparel & Luxury Goods': 'Consumer Discretionary'
}