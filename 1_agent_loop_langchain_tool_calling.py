from dotenv import load_dotenv
from langchain.messages import HumanMessage,SystemMessage, ToolMessage
from langchain.tools import tool
from langsmith import traceable
from langchain.chat_models import init_chat_model

load_dotenv()


@tool
def get_product_price(product_name:str)->float:
    """Get the price of a product by its name."""
    dummy_prices = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99,
    }
    return dummy_prices.get(product_name.lower(), 0.0)

@tool 
def apply_discount(price:float, discount_tier:str)->float:
    """Apply a discount to a price based on the discount tier."""
    discount_rates = {
        "bronze": 0.05,
        "silver": 0.10,
        "gold": 0.15,
    }
    discount_rate = discount_rates.get(discount_tier.lower(), 0.0)
    discounted_price = price * (1 - discount_rate)
    return discounted_price

@traceable(name="ecommerce_agent_main")
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tool_dict = {tool.name:tool for tool in tools}
    llm = init_chat_model(f"ollama:qwen3:1.7b",temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("="*60)

    messages = [
        SystemMessage(
            content="You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."),

        HumanMessage(content=question)
    ]

    for iteration in range(1,11):
        print(f"\n --- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\n --- Final Answer --- {ai_message.content}")
            return ai_message.content
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_id = tool_call.get("id")

        tool = tool_dict.get(tool_name)

        print(f"\nTool Call: {tool_name} :=> {tool_args}")

        if not tool:
            print(f"Tool '{tool_name}' not found")
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tool.invoke(tool_args)

        print(f"\nTool Call result: {tool_name} :=> {tool_args} :=> {observation}")
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation),tool_call_id=tool_id))

    print("ERROR: Max iterations reached without a final answer")
    return None






def main():
    print("Hello from ecommerce-agent!")
    result = run_agent("What is the price of a smartphone after applying a bronze discount?")
    print(f"\nFinal Result: {result}")


if __name__ == "__main__":
    main()