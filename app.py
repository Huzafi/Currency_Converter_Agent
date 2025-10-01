import streamlit as st
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool
from agents.run import RunConfig
import asyncio
import requests

gemini_api_key = st.secrets["GEMINI_API_KEY"]

# Check API key
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

# Client setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


# --------- TOOL ---------
@function_tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert amount from one currency to another using live exchange rates.
    """
    try:
        response = requests.get(f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}")
        response.raise_for_status()
        data = response.json()
        rate = data['rates'].get(to_currency.upper())
        if rate:
            converted = amount * rate
            return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"
        else:
            return f"❌ Currency {to_currency.upper()} not supported."
    except Exception as e:
        return f"⚠️ Failed to fetch exchange rate. Error: {str(e)}"

agent = Agent(
    name="Currency Converter Agent",
    instructions="You are a helpful currency converter chatbot. You convert currencies using live exchange rates.",
    model=model,
    tools=[convert_currency]
)

# --------- STREAMLIT UI ---------
st.set_page_config(page_title="💱 Currency Converter Agent (Grok)", page_icon="💱")
st.title("💱 Currency Converter Agent")
st.markdown("Welcome! I can convert currencies using live exchange rates. Type your request below 👇")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Type your conversion request here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Converting..."):
            result = Runner.run_sync(agent, user_input)
            response = result.final_output or "⚠️ Sorry, I couldn’t process your request."
            st.markdown(f"💬 **Response:**\n\n{response}")
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()
