import pandas as pd
import numpy as np
from scipy import stats

# ==========================================================
# Load data
# ==========================================================
train = pd.read_csv('train.csv')

# ==========================================================
# Example configuration (edit if needed)
# ==========================================================
# 예시: 두 독립 표본의 평균 차이에 대한 t-검정
# 'data1', 'data2'는 문제에서 요구하는 두 그룹 컬럼명으로 바꿔서 사용하세요.
col1 = 'data1'
col2 = 'data2'

# 필요한 컬럼이 있는지 확인
for col in [col1, col2]:
    if col not in train.columns:
        raise ValueError(f"필요한 컬럼이 없습니다: {col}")

x = train[col1].dropna().astype(float)
y = train[col2].dropna().astype(float)

# ==========================================================
# 1) 기본 통계량 계산
# ==========================================================
nx = len(x)
ny = len(y)

mean_x = np.mean(x)
mean_y = np.mean(y)

var_x = np.var(x, ddof=1)   # 표본 분산
var_y = np.var(y, ddof=1)

std_x = np.sqrt(var_x)
std_y = np.sqrt(var_y)

print("[1단계] 기본 통계량")
print(f"n1 = {nx}, n2 = {ny}")
print(f"mean1 = {round(mean_x, 4)}")
print(f"mean2 = {round(mean_y, 4)}")
print(f"var1 = {round(var_x, 4)}")
print(f"var2 = {round(var_y, 4)}")
print()

# ==========================================================
# 2) 합동 분산 추정량 (pooled variance)
# ==========================================================
# s_p^2 = ((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2)
pooled_var = ((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2)
pooled_std = np.sqrt(pooled_var)

print("[2단계] 합동 분산 추정량")
print(f"pooled_var = {round(pooled_var, 4)}")
print(f"pooled_std = {round(pooled_std, 4)}")
print()

# ==========================================================
# 3) 표준오차 계산
# ==========================================================
# SE = sqrt(s_p^2 / n1 + s_p^2 / n2)
se = np.sqrt(pooled_var / nx + pooled_var / ny)

print("[3단계] 표준오차")
print(f"SE = {round(se, 4)}")
print()

# ==========================================================
# 4) t 통계량 계산 (평균 차이 / 표준오차)
# ==========================================================
# t = (mean1 - mean2) / SE
obs_diff = mean_x - mean_y
t_stat_formula = obs_diff / se

print("[4단계] t 통계량")
print(f"difference = {round(obs_diff, 4)}")
print(f"t_stat_formula = {round(t_stat_formula, 4)}")
print()

# ==========================================================
# 5) p-value 계산 (자유도: n1+n2-2)
# ==========================================================
df = nx + ny - 2
p_value_formula = 2 * (1 - stats.t.cdf(abs(t_stat_formula), df=df))

print("[5단계] p-value")
print(f"df = {df}")
print(f"p_value_formula = {round(p_value_formula, 4)}")
print()

# ==========================================================
# 6) 교차 검증: scipy.stats 함수로 비교
# ==========================================================
# 두 독립 표본 t-검정 (등분산 가정)
# stats.ttest_ind(x, y, equal_var=True) -> statistic, pvalue

scipy_t, scipy_p = stats.ttest_ind(x, y, equal_var=True)

print("[6단계] 교차 검증")
print(f"manual t = {round(t_stat_formula, 4)}")
print(f"scipy t = {round(scipy_t, 4)}")
print(f"manual p = {round(p_value_formula, 4)}")
print(f"scipy p = {round(scipy_p, 4)}")
print()

# 일치 여부 확인
print("결과 비교")
print(f"t 일치 여부: {abs(round(t_stat_formula, 6) - round(scipy_t, 6)) < 1e-6}")
print(f"p 일치 여부: {abs(round(p_value_formula, 6) - round(scipy_p, 6)) < 1e-6}")
