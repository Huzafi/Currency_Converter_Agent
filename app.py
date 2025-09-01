import streamlit as st
import nest_asyncio
import requests
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from openai import AsyncOpenAI

gemini_api_key = st.secrets["GEMINI_API_KEY"]

if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY is not set. Please add it in .env or Streamlit secrets.")
    st.stop()

# Apply nest_asyncio fix
nest_asyncio.apply()

# OpenAI Client
client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=gemini_api_key
)

# Disable tracing
set_tracing_disabled(disabled=True)

@function_tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
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
    except Exception:
        return "⚠️ Failed to fetch exchange rate. Please try again later."

# Agent setup
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client,
)

agent = Agent(
    name="Currency Converter Agent",
    instructions="You are a helpful currency converter chatbot. You convert currencies using live exchange rates.",
    model=model,
    tools=[convert_currency]
)

# Streamlit UI
st.set_page_config(page_title="💱 Currency Converter Agent", page_icon="💱")
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
            response = result.final_output
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()
