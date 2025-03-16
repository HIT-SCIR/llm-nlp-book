from collections import defaultdict
import pandas as pd
import json

def compute_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    # 初始化模型得分
    ratings = defaultdict(lambda: INIT_RATING)

    # 遍历每次两两比较
    for _, model_a, model_b, winner in battles[['model_a', 'model_b', 'winner']].itertuples():
        ra = ratings[model_a]
        rb = ratings[model_b]

        # 计算期望胜率
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))

        # 根据真实胜率更新等级分
        if winner == "model_a":
            sa = 1
        elif winner == "model_b":
            sa = 0
        elif winner in ["tie", "tie (bothbad)"]:
            sa = 0.5
        else:
            raise ValueError(f"unexpected winner value: {winner}")
        ratings[model_a] += K * (sa - ea)
        ratings[model_b] += K * (1 - sa - eb)

    return ratings

# 示例数
battles = pd.DataFrame({
    'model_a': ['A', 'A', 'B', 'C', 'C', 'D'],
    'model_b': ['B', 'C', 'C', 'D', 'A', 'A'],
    'winner': ['model_a', 'model_b', 'model_b', 'model_a', 'tie', 'model_b']
})

# 计算Elo评分
elo_scores = compute_elo(battles)
print(json.dumps(elo_scores, indent=2))
