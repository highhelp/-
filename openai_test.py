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