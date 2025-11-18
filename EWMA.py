import pandas as pd
import numpy as np

# --- 1. 데이터 로드 및 준비 ---

# CSV 파일 경로를 여기에 입력
file_path = 'ELS.csv' 

# file 찾기 및 열기
try:
    df = pd.read_csv(file_path)
    
    df['date'] = pd.to_datetime(df['date'])
    # date열의 행이 기준이 되어서 계산을 수행할 예정임.
    df = df.set_index('date')

    # 맨 처음 새로 만들 열 드롭
    columns_to_drop = ['vol_samsung', 'vol_kospi', 'corr']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
except FileNotFoundError:
    print(f"Error: 파일 경로 '{file_path}'를 확인해주세요.")


# --- 2. 로그 수익률 계산 ---

# 열 숫자변환
# 삼성은 ,가 들어있어서 제거해주어야함.
df['samsung'] = df['samsung'].astype(str).str.replace(',', '', regex=False).str.strip()
df['samsung'] = pd.to_numeric(df['samsung'], errors='coerce')
df['kospi200'] = pd.to_numeric(df['kospi200'], errors='coerce')


# shift를 통해 로그 수익률 계산하기
df['Samsung_Return'] = np.log(df['samsung'] / df['samsung'].shift(1))
df['KOSPI200_Return'] = np.log(df['kospi200'] / df['kospi200'].shift(1))

# --- 3. EWMA 변동성 및 상관계수 추정 ---

LAMBDA = 0.94 # EWMA 감쇠 인자
WINDOW_SIZE = 125 # 6개월치 영업일 수 설정

# 결과를 저장할 새로운 열 생성 (초기값은 NaN) - 이후 연율화 할 예정임.
df['vol_samsung_ewma_6m_ann'] = np.nan
df['vol_kospi_ewma_6m_ann'] = np.nan
df['corr_ewma_6m'] = np.nan

# EWMA 추정의 반복 계산 (최소 윈도우 크기부터 시작)
# i는 EWMA 변동성을 기록할 시점 (t)의 인덱스입니다.
# WINDOW_SIZE-1 인덱스까지는 계산할 과거 데이터가 부족하므로, WINDOW_SIZE 인덱스부터 시작합니다.
for i in range(WINDOW_SIZE, len(df)):
    
    # 1. 6개월치 데이터 추출 (현재 시점 i 바로 직전까지의 WINDOW_SIZE개 행을 자름)
    # window_df는 i-WINDOW_SIZE부터 i-1까지의 수익률을 포함합니다.
    window_df = df.iloc[i - WINDOW_SIZE : i].copy()
    
    # 윈도우 내의 로그수익률 뽑아오기
    r_s_window = window_df['Samsung_Return']
    r_k_window = window_df['KOSPI200_Return']
    
    # 2. 초기값 설정 (매 윈도우마다 재설정)
    # 초기 분산/공분산은 윈도우 내 첫 번째 수익률의 제곱/곱을 사용합니다 (r_1^2 방식).
    initial_var_s = r_s_window.iloc[0]**2
    initial_var_k = r_k_window.iloc[0]**2
    initial_cov = r_s_window.iloc[0] * r_k_window.iloc[0]

    ewma_var_s = [initial_var_s] 
    ewma_var_k = [initial_var_k]
    ewma_cov = [initial_cov]
    
    # 3. 윈도우 내에서 EWMA 반복 계산 (두 번째 시점부터)
    # 윈도우의 크기는 WINDOW_SIZE 이므로, 1부터 WINDOW_SIZE-1까지 반복합니다.
    for j in range(1, WINDOW_SIZE):
        
        # t-1 시점의 수익률 (window_df 기준)
        r_s_t1 = r_s_window.iloc[j] 
        r_k_t1 = r_k_window.iloc[j]
        
        # EWMA 분산/공분산 계산 (sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2)
        # ewma_var, cov를 리스트로 하나씩 늘려가는 형식으로 저장함.

        # 삼성 분산
        var_s = LAMBDA * ewma_var_s[-1] + (1 - LAMBDA) * r_s_t1**2
        ewma_var_s.append(var_s)
        
        # KOSPI200 분산
        var_k = LAMBDA * ewma_var_k[-1] + (1 - LAMBDA) * r_k_t1**2
        ewma_var_k.append(var_k)
        
        # 공분산
        cov = LAMBDA * ewma_cov[-1] + (1 - LAMBDA) * r_s_t1 * r_k_t1
        ewma_cov.append(cov)

    # 4. 계산된 최종 값 (윈도우의 마지막 값)을 전체 df의 i 시점에 저장
    final_var_s = ewma_var_s[-1]
    final_var_k = ewma_var_k[-1]
    final_cov = ewma_cov[-1]
    
    # 연율화 (252는 영업일 기준 연간 일수)
    final_vol_s_ann = np.sqrt(final_var_s * 252)
    final_vol_k_ann = np.sqrt(final_var_k * 252)
    
    # 상관계수 계산
    corr = final_cov / (np.sqrt(final_var_s) * np.sqrt(final_var_k))
    
    # 전체 df에 결과 저장 (날짜 인덱스 사용)
    current_date = df.index[i]
    df.loc[current_date, 'vol_samsung_ewma_6m_ann'] = final_vol_s_ann
    df.loc[current_date, 'vol_kospi_ewma_6m_ann'] = final_vol_k_ann
    df.loc[current_date, 'corr_ewma_6m'] = corr

# --- 4. 변동성 및 상관계수 결과 열 추가 ---

# 이름 csv파일에 맞게 변경
df = df.rename(columns={'vol_samsung_ewma_6m_ann': 'vol_samsung'})
df = df.rename(columns={'vol_kospi_ewma_6m_ann': 'vol_kospi'})
df = df.rename(columns={'corr_ewma_6m': 'corr'})

# 2. 임시로 생성된 일간 수익률 열을 제거합니다.
df = df.drop(columns=[
    'Samsung_Return', 'KOSPI200_Return'
], errors='ignore') # 'errors=ignore'는 열이 없더라도 오류를 발생시키지 않도록 합니다.

# 3. 보기 쉽게 소수점 포맷을 조정합니다.
df = df.round(4)
# --- 5. 결과 출력 ---

print("\n## 📊 EWMA 변동성 및 상관계수 추정 결과 (Lambda = 0.94)")
# 'date'는 현재 인덱스로 설정되어 있으므로, reset_index()를 사용하여 열로 다시 변환합니다.
df_display = df.reset_index()

# 기준 날짜 설정
start_date = '2021-10-22'

# 날짜 조건을 만족하는 행만 필터링합니다.
# df_display['date']가 Datetime 형식이라고 가정합니다.
filtered_df = df_display[df_display['date'] >= start_date]

# 원하는 열만 선택하여 출력합니다.
print(filtered_df[['date', 'samsung', 'vol_samsung', 'vol_kospi', 'corr']])

# --- 6. CSV 파일에 계산 결과 입력 및 저장 ---
try:
    # 1. 기존 CSV 파일을 다시 읽어와서 기존 데이터프레임(original_df)으로 만듭니다.
    # 인덱스는 'date' 열로 설정해야 병합이 정확하게 이루어집니다.
    original_df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    
    # 2. 업데이트할 열 이름 리스트 정의
    update_columns = ['vol_samsung', 'vol_kospi', 'corr']
    
    # 3. CSV 파일의 열에 업데이트할 모든 열이 존재하는지 확인
    all_columns_exist = all(col in original_df.columns for col in update_columns)
    
    if not all_columns_exist:
        # 4. 존재하지 않는 열이 있다면 새로 생성
        print(f"⚠️ 경고: CSV 파일에 {update_columns} 중 일부 또는 전부가 없습니다. 새로운 열을 추가합니다.")
        
    # 5. 기존 CSV 데이터프레임(original_df)에 현재 계산된 df의 열을 병합(업데이트)합니다.
    # df와 original_df는 모두 'date'를 인덱스로 가지고 있으므로 인덱스를 기준으로 정렬되어 데이터가 입력됩니다.
    for col in update_columns:
        # df[col]의 값을 original_df의 해당 열에 입력합니다.
        # 인덱스(날짜)가 일치하는 행에 값이 입력됩니다.
        original_df[col] = df[col]

    # 6. 최종 결과를 'ELS.csv' 파일에 저장합니다. (기존 파일 덮어쓰기)
    original_df.to_csv(file_path)
    print(f"💾 계산된 {update_columns} 열의 값이 '{file_path}' 파일에 성공적으로 저장되었습니다.")

except FileNotFoundError:
    print(f"Error: 저장 중 파일 경로 '{file_path}'를 다시 확인해주세요.")
except Exception as e:
    print(f"Error: CSV 파일 저장 중 오류가 발생했습니다: {e}")