import ollama
import re
import inspect
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@traceable(run_type="tool")
def get_product_price(product:str)->float:
    # Simulate fetching product price from a database or API
    product_prices = {
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 199.99,
        "superwidget": 149.99
    }
    return product_prices.get(product.lower(), 0.0)

@traceable(run_type="tool")
def apply_discount(price:float, discount_tier:str)->float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools_dict = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools_dict)
tool_names = ", ".join(tools_dict.keys())


react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

@traceable(name="Ollama Chat",run_type="llm")
def ollama_chat_traced(model,messages,options):
    response = ollama.chat(model=model, messages=messages, options=options)
    return response

@traceable(name="Ollama Agent Loop")
def run_agent(question:str):
    print(f"User question: {question}")
    print("#"*40)

    prompt = react_prompt.format(question=question)
    scratchpad = ""
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = f"{prompt}\n{scratchpad}"
        messages = [{"role": "system", "content": full_prompt}]
        response = ollama_chat_traced(
            model=MODEL,
            messages=messages,
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        print(f"LLM response:\n{response}")
        output = response.message.content
        print(f"LLM Output:\n{output}")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        
        
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break


        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools_dict:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools_dict.keys())}"
        else:
            observation = str(tools_dict[tool_name](*args))


        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"


        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")




def main():
    question = "What is the price of the SuperWidget and how much would it be with a silver tier discount?"
    result = run_agent(question)



if __name__ == "__main__":
    main()