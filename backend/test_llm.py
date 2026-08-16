from app.services.llm.llm_services import LLMService


llm = LLMService()

answer = llm.generate(
    query="What is machine learning?",
    context="""
Machine learning is a branch of artificial intelligence
that allows computers to learn patterns from data and make
predictions or decisions without being explicitly programmed
for every individual task.
""",
)

print("\nANSWER:")
print(answer)