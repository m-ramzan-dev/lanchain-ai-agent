from dotenv import load_dotenv
import ollama
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@traceable(run_type="tool")
def get_product_price(product:str)->float:
    # Simulate fetching product price from a database or API
    product_prices = {
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 199.99
    }
    return product_prices.get(product.lower(), 0.0)

@traceable(run_type="tool")
def apply_discount(price:float, discount_tier:str)->float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

@traceable(name="Ollama Chat",run_type="llm")
def ollama_chat_traced(messages):
    response = ollama.chat(model=MODEL, messages=messages, tools=tools_for_llm)
    return response

@traceable(name="Ollama Agent Loop")
def run_agent(question:str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }
    print(f"User question: {question}")
    print("#"*40)

    messages = [
        {"role":"system","content":(
                "You are a helpful shopping assistant. "
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
                "ask them which tier to use — do NOT assume one."
        )},
        {"role":"user","content":question}
    ]

    for i in range(1,11):
        print(f"\n=== Iteration {i} ===")

        response = ollama_chat_traced(messages=messages)
        print(f"LLM response:\n{response}")
        ai_message = response.message
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        print(f"Tool call detected: {tool_name} with args {tool_args}")
        tool = tools_dict.get(tool_name)

        if tool is None:
            print(f"Error: Tool '{tool_name}' not found.")
            return
        
        observation = tool(**tool_args)
        print(f"Tool observation: {observation}")
        messages.append(ai_message)
        messages.append({"role":"tool","content":str(observation)})
        
    print("ERROR: Max iterations reached without a final answer")
    return None



def main():
    print("Hello from ecommerce-agent!")
    result = run_agent("What is the price of a my school bag after applying a bronze discount?")


if __name__ == "__main__":
    main()