import asyncio

# Placeholder for the actual PromptBuilder class and its methods
class PromptBuilder:
    def __init__(self):
        pass

    async def generate_market_summary_prompt(self, data):
        # Placeholder implementation
        return "Generated market summary prompt based on data."

    async def generate_reflection_prompt(self, data):
        # Placeholder implementation
        return "Generated reflection prompt based on data."

async def test_mock_prompt_builder():
    # Placeholder for testing the prompt builder
    builder = PromptBuilder()
    summary = await builder.generate_market_summary_prompt({"price": "100", "volume": "1000"})
    print("Market Summary (mock):", summary)
    reflection = await builder.generate_reflection_prompt({"trade_id": "123", "profit": "10%"})
    print("Reflection Prompt (mock):", reflection)

if __name__ == '__main__':
    # import asyncio # Already imported at top level
    asyncio.run(test_mock_prompt_builder())
