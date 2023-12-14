from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import matplotlib.pyplot as plt
import networkx as nx
# 创建贝叶斯网络对象
model = BayesianNetwork()

# 添加节点
model.add_nodes_from(['0GB', '1HJ', '2LF', '3QK', '4SJ', '5SP', '6KD', 'Grade', 'FlawNum'])

# 添加边（节点之间的关系）
model.add_edges_from([
    ('0GB', 'Grade'),
    ('1HJ', 'Grade'),
    ('2LF', 'Grade'),
    ('3QK', 'Grade'),
    ('4SJ', 'Grade'),
    ('5SP', 'Grade'),
    ('6KD', 'Grade'),
    ('FlawNum', 'Grade')
])

# 设置节点的先验概率
prior_probabilities = {
    '0GB': [0.611891, 0.388109],
    '1HJ': [0.90831,  0.09169],
    '2LF': [0.940912, 0.059088],
    '3QK': [0.977635, 0.022365],
    '4SJ': [0.629444, 0.370556],
    '5SP': [0.938721, 0.061279],
    '6KD': [0.993088, 0.006912],
    'Grade': [0.7, 0.3],  # 分级节点的先验概率为 70% 为 Low，30% 为 High
    'FlawNum': [0.6, 0.4]  # FlawNum节点的先验概率为 60% 为 Low，40% 为 High
}

# 添加节点的条件概率表
cpd_0GB = TabularCPD(variable='0GB', variable_card=2, values=[[0.611891], [0.388109]], state_names={'0GB': [0, 1]})
cpd_1HJ = TabularCPD(variable='1HJ', variable_card=2, values=[[0.90831],  [0.09169]], state_names={'1HJ': [0, 1]})
cpd_2LF = TabularCPD(variable='2LF', variable_card=2, values=[[0.940912], [0.059088]], state_names={'2LF': [0, 1]})
cpd_3QK = TabularCPD(variable='3QK', variable_card=2, values=[[0.977635], [0.022365]], state_names={'3QK': [0, 1]})
cpd_4SJ = TabularCPD(variable='4SJ', variable_card=2, values=[[0.629444], [0.370556]], state_names={'4SJ': [0, 1]})
cpd_5SP = TabularCPD(variable='5SP', variable_card=2, values=[[0.938721], [0.061279]], state_names={'5SP': [0, 1]})
cpd_6KD = TabularCPD(variable='6KD', variable_card=2, values=[[0.993088], [0.006912]], state_names={'6KD': [0, 1]})
cpd_Grade = TabularCPD(variable='Grade', variable_card=2, values=[[0.7], [0.3]], state_names={'Grade': [0, 1]})
cpd_FlawNum = TabularCPD(variable='FlawNum', variable_card=2, values=[[0.6], [0.4]], state_names={'FlawNum': [0, 1]})
# cpd_Grade2 = TabularCPD(variable='Grade', variable_card=2,
#                        values=[[0, 0, 1, 1], [1, 1, 0, 0]],
#                        evidence=['3QK', '5SP', 'FlawNum'],
#                        evidence_card=[2, 2, 2],
#                        state_names={'Grade': [0, 1],
#                                     '3QK': [0, 1],
#                                     '5SP': [0, 1],
#                                     'FlawNum': [0, 1]})


model.add_cpds(cpd_0GB, cpd_1HJ, cpd_2LF, cpd_3QK, cpd_4SJ, cpd_5SP, cpd_6KD, cpd_Grade, cpd_FlawNum)

# 验证模型的结构和条件概率表是否有效
print("Bayesian Network Structure:")
print(model.edges())
print("\nCPDs:")
for cpd in model.get_cpds():
    print(cpd)


# 创建一个空的有向图
G = nx.DiGraph()

# 添加节点
G.add_nodes_from(['0GB', '1HJ', '2LF', '3QK', '4SJ', '5SP', '6KD', 'Grade', 'FlawNum'])

# 添加边（节点之间的关系）
edges = [('0GB', 'Grade'), ('1HJ', 'Grade'), ('2LF', 'Grade'), ('3QK', 'Grade'),
         ('4SJ', 'Grade'), ('5SP', 'Grade'), ('6KD', 'Grade'), ('FlawNum', 'Grade')]

G.add_edges_from(edges)

# 设置节点位置
pos = {
    '0GB': (0, 3),
    '1HJ': (1, 3),
    '2LF': (2, 3),
    '3QK': (3, 3),
    '4SJ': (4, 3),
    '5SP': (5, 3),
    '6KD': (6, 3),
    'Grade': (3, 1),
    'FlawNum': (7, 3)
}

# 绘制贝叶斯网络图
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=4000, node_color='skyblue', font_weight='bold', font_size=10,
        arrowsize=20, arrowstyle='->')
plt.title("Bayesian Network In FlawDetection ")
plt.show()
