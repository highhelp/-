import openai
import pandas as pd

openai.api_key = "api_key"

model = "gpt-3.5-turbo"

df = pd.read_csv(r'C:\Users\gohun\Desktop\AI융합학과.csv', encoding='euc-kr')

print("\n질문을 입력하세요. 예시: '3학점 수업들을 알려주세요'")
user_question = input("질문: ")

prompt = f"""
질문: {user_question}
CSV 데이터: 아래는 관련된 데이터를 포함한 CSV의 미리보기입니다.
{df.head().to_string(index=False)}

위 데이터를 참고하여 질문에 대해 간결하고 정확하게 답해주세요.
"""

response = openai.ChatCompletion.create(
    model=model,
    messages=[
        {"role": "system", "content": "대화하듯이 작성해줘"},
        {"role": "user", "content": prompt},
    ],
    temperature=0.5,
    max_tokens=500,
    top_p=1
)

print(response.choices[0].message.content)

//1차 test//
----------------------------------------------------------------------

import openai

openai.api_key = "your_api_key"  # 실제 API 키를 여기에 입력하세요.

model = "gpt-3.5-turbo"

# 예시 질문 및 데이터 (정리된 데이터로 교체하세요)
question = "강화캠퍼스 4교시는 몇 시부터 시작하나요?"
data = """
8. 수업시간 안내
[강화캠퍼스 - 주간]
1교시: 10:00 ~ 10:50
2교시: 11:00 ~ 11:50
3교시: 12:00 ~ 12:50
4교시: 13:30 ~ 14:20
...
"""

prompt = f"""
데이터:
{data}

질문:
{question}

위 데이터를 참고하여 질문에 대해 간결하고 정확하게 답변해주세요.
"""

response = openai.ChatCompletion.create(
    model=model,
    messages=[
        {"role": "system", "content": "데이터를 참고해 간결하고 정확히 답변해주세요."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.5,
    max_tokens=100,
    top_p=1,
)

print(response.choices[0].message.content)


API 키 설정: openai.api_key에 자신의 OpenAI API 키를 입력해야 합니다. api_key를 실제 키로 교체하세요.
extracted_text 사용: OCR에서 추출한 텍스트가 정제되지 않은 상태라면, 모델의 답변이 부정확하거나 길어질 수 있습니다. 이를 먼저 정리하거나 필요한 정보만 추출하세요.
프롬프트 설계: prompt에서 구체적인 질문이나 명령을 명시해야 합니다. 현재 텍스트에서 질문에 대해 명확히 대답하도록 요구합니다.
API 요청 설정:
temperature: 모델의 창의성 수준을 제어합니다. (낮은 값은 더 결정적이고, 높은 값은 더 창의적입니다.)
max_tokens: 응답의 최대 길이를 지정합니다.
실행 전 수정이 필요한 부분은 다음과 같습니다:

api_key를 실제 OpenAI API 키로 변경.
OCR에서 추출한 텍스트를 extracted_text로 그대로 사용하지 않고, 필요한 정보만 간결히 정리.
//2차테스트//
