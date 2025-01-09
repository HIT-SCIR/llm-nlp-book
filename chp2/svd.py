# Defined in Section 2.1.2
import matplotlib.pyplot as plt
import numpy as np

M = np.array([[0, 2, 1, 1, 1, 1, 1, 2, 1, 3],
              [2, 0, 1, 1, 1, 0, 0, 1, 1, 2],
              [1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
              [1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
              [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
              [1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
              [2, 1, 0, 0, 0, 1, 1, 0, 1, 2],
              [1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
              [3, 2, 1, 1, 1, 1, 1, 2, 1, 0]])

def pmi(M, positive=True):
    col_totals = M.sum(axis=0)
    row_totals = M.sum(axis=1)
    total = col_totals.sum()
    expected = np.outer(row_totals, col_totals) / total
    M = M / expected
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        M = np.log(M)
    M[np.isinf(M)] = 0.0  # log(0) = 0
    if positive:
        M[M < 0] = 0.0
    return M

M_pmi = pmi(M)

np.set_printoptions(precision=2)
print(M_pmi)

U, s, Vh = np.linalg.svd(M_pmi)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

words = ["我", "喜欢", "自然", "语言", "处理", "爱", "深度", "学习", "机器", "。"]

for i in range(len(words)):
    plt.text(U[i, 0], U[i, 1], words[i])
    plt.scatter(U[i, 0], U[i, 1], c='red', s=50)

plt.title('词向量分布图')
plt.xlabel('第一维度')
plt.ylabel('第二维度')
plt.grid(True, linestyle='--', alpha=0.7)
plt.margins(0.1)
output_file = 'svd.pdf'
plt.savefig(output_file, bbox_inches='tight', dpi=300)
print(f"图形已保存至 {output_file}")
plt.show()
