from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

print("Загрузка локальной LLM для Агента...")
# Используем ту же локальную модель для скорости
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    max_new_tokens=150,
    temperature=0.3  # Низкая температура снижает галлюцинации
)

# Оборачиваем модель в формат LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# ТЗ для Агента (Системный промпт)
template = """<|system|>
Ты - умный ИИ-агент службы поддержки.
Твоя задача: прочитать запрос клиента и дать краткий, вежливый ответ на русском языке.
Если клиент спрашивает про доставку, скажи, что она занимает 3-5 дней.
Если про возврат, скажи, что возврат возможен в течение 14 дней.
</s>
<|user|>
Запрос клиента: {customer_request}
</s>
<|assistant|>
"""

prompt = PromptTemplate(template=template, input_variables=["customer_request"])

# Создаем цепочку (Chain) агента: Промпт -> LLM
agent_chain = prompt | llm

# Тестируем агента на разных запросах
requests = [
    "Где моя посылка? Я заказал ее неделю назад!",
    "Как я могу вернуть бракованный товар?"
]

print("\n=== Старт работы ИИ-Агента ===")
for req in requests:
    print(f"\n👤 Клиент: {req}")
    response = agent_chain.invoke({"customer_request": req})

    # Очищаем ответ от промпта (оставляем только текст ассистента)
    clean_response = response.split("<|assistant|>")[-1].strip()
    print(f"🤖 Агент: {clean_response}")