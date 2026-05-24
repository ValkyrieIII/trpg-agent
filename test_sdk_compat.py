"""Phase 0.2: Verify openai-agents SDK + DeepSeek function calling compatibility."""
import os
from openai import AsyncOpenAI
from agents import (
    Agent, Runner, function_tool,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)


@function_tool
def get_weather(city: str) -> str:
    """Get weather info for a city."""
    weather_data = {
        "beijing": "Sunny, 25C",
        "tokyo": "Cloudy, 20C",
        "new york": "Rainy, 15C",
    }
    return weather_data.get(city.lower(), f"{city}: Sunny, 22C")


@function_tool
def roll_dice(expression: str) -> str:
    """Roll dice. expression='d20' or '3d6+2'."""
    return f"Roll {expression} -> result: 15"


def main():
    # 1. Configure DeepSeek client + force Chat Completions API
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    set_default_openai_client(client)
    set_default_openai_api("chat_completions")  # DeepSeek doesn't support Responses API
    set_tracing_disabled(True)  # Disable tracing for cleaner output
    print("[OK] DeepSeek client configured (chat_completions mode)")

    # 2. Test: bare Agent (no tools)
    print("\n--- Test 1: Bare Agent (no tools) ---")
    simple_agent = Agent(
        name="TestAgent",
        instructions="You are a test assistant. Reply briefly in English.",
        model="deepseek-v4-flash",
    )
    result = Runner.run_sync(simple_agent, "Say hello in one sentence")
    print(f"Output: {result.final_output[:200].encode('ascii', 'replace').decode()}")
    print("[OK] Bare Agent passed")

    # 3. Test: Agent + single tool
    print("\n--- Test 2: Agent + single tool ---")
    weather_agent = Agent(
        name="WeatherAgent",
        instructions="You can use get_weather tool. Reply in English.",
        tools=[get_weather],
        model="deepseek-v4-flash",
    )
    result = Runner.run_sync(weather_agent, "What's the weather in Tokyo?")
    print(f"Output: {result.final_output[:200].encode('ascii', 'replace').decode()}")
    print("[OK] Single tool passed")

    # 4. Test: Agent + multiple tools
    print("\n--- Test 3: Multi-tool agent ---")
    multi_agent = Agent(
        name="MultiToolAgent",
        instructions="You can use get_weather and roll_dice. Reply in English.",
        tools=[get_weather, roll_dice],
        model="deepseek-v4-flash",
    )
    result = Runner.run_sync(
        multi_agent,
        "Check Tokyo weather and roll a d20",
        max_turns=5,
    )
    print(f"Output: {result.final_output[:300].encode('ascii', 'replace').decode()}")
    print("[OK] Multi-tool passed")

    print("\n===== All compatibility tests passed =====")


if __name__ == "__main__":
    main()
