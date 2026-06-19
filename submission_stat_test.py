import pandas as pd
from scipy import stats

# ==========================================================
# [설정] 시험장 환경에 맞게 이 부분만 딱! 수정하세요.
# ==========================================================
# 1. 실행할 검정 선택: 'independent_ttest' / 'paired_ttest' / 'chi_square' / 'anova' / 'wilcoxon'
TEST_TYPE = 'independent_ttest' 

# 2. 분석할 데이터와 컬럼 지정
train = pd.read_csv('train.csv')
TARGET = 'target'
COL1 = 'data1'
COL2 = 'data2'

# 결측치 제거 및 부동소수점 변환 (에러 방지용 필수 전처리)
x = train[COL1].dropna().astype(float) if COL1 in train.columns else None
y = train[COL2].dropna().astype(float) if COL2 in train.columns else None

print(f"🔍 [실행 중인 검정]: {TEST_TYPE}\n")

# ==========================================================
# 🚀 수식 제로! 오직 라이브러리 함수로만 통계량, p-value 추출
# ==========================================================
if TEST_TYPE == 'independent_ttest':
    # [1] 독립표본 t-검정 (수식 없이 한 줄로 끝!)
    # 등분산 가정이면 equal_var=True, 아니면 False (지문에 분산이 같다고 주어지면 True)
    statistic, pvalue = stats.ttest_ind(x, y, equal_var=True)
    print("📊 독립표본 t-검정 완료")

elif TEST_TYPE == 'paired_ttest':
    # [2] 대응표본 t-검정 (치료 전/후 비교 등)
    statistic, pvalue = stats.ttest_rel(x, y)
    print("📊 대응표본 t-검정 완료")

elif TEST_TYPE == 'chi_square':
    # [3] 카이제곱 검정 (범주형 빈도 데이터 비교)
    # 크로스탭(교차표)만 만들어서 넣어주면 알아서 다 계산해 줍니다.
    table = pd.crosstab(train[TARGET], train[COL1])
    statistic, pvalue, dof, expected = stats.chi2_contingency(table)
    print("📊 카이제곱 독립성 검정 완료")

elif TEST_TYPE == 'anova':
    # [4] 분산분석 (3개 이상 집단의 평균 비교)
    # 지문에 "세 집단의 평균 차이"가 나오면 무조건 이 함수를 씁니다.
    statistic, pvalue = stats.f_oneway(x, y)
    print("📊 일원분산분석(ANOVA) 완료")

elif TEST_TYPE == 'wilcoxon':
    # [5] 윌콕슨 부호순위 검정 (정규분포를 따르지 않는 대응표본 비모수 검정)
    statistic, pvalue = stats.wilcoxon(x, y)
    print("📊 윌콕슨 비모수 검정 완료")

else:
    raise ValueError("올바른 TEST_TYPE을 입력하세요.")

# ==========================================================
# 🏆 최종 제출용 결과 출력 (반올림 처리 완료)
# ==========================================================
print("-" * 50)
print(f"🎯 검정 통계량 (statistic): {round(statistic, 4)}")
print(f"🎯 p-value: {round(pvalue, 6)}")
print("-" * 50)

# 귀무가설 판단 (0.05 기준)
alpha = 0.05
if pvalue < alpha:
    print(f"📢 결론: p-value가 {alpha}보다 작으므로 귀무가설을 [기각]합니다.")
    print("   👉 두 집단 간에 유의미한 차이가 존재합니다.")
else:
    print(f"📢 결론: p-value가 {alpha}보다 크므로 귀무가설을 [채택/기각하지 않음]합니다.")
    print("   👉 두 집단 간에 유의미한 차이가 없습니다.")