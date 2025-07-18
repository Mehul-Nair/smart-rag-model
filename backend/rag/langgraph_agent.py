import os
from langgraph.graph import StateGraph, END

# from langgraph.nodes import Node
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import Dict, Any, TypedDict, Optional, List
from pydantic import SecretStr

# --- Node Definitions ---


class AgentState(TypedDict, total=False):
    user_message: str
    retriever: Any
    llm: Any
    retrieved_docs: Optional[List[Document]]
    llm_response: Optional[str]
    intent: Optional[str]


def classify_node(state: AgentState) -> AgentState:
    """Classify user intent using greetings/meta detection, vector store, and LLM for intelligent classification."""
    user_message = state.get("user_message", "")
    retriever = state.get("retriever")
    llm = state.get("llm")

    # 1. Check for greetings or meta queries first
    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good evening",
        "good afternoon",
        "how are you",
        "what's up",
        "who are you",
        "help",
        "about",
        "catalog",
        "what can you do",
        "capabilities",
    ]
    meta_queries = [
        "what categories",
        "what products",
        "what do you have",
        "show me categories",
        "list categories",
        "available products",
        "available categories",
        "what can i buy",
        "what's available",
        "list the categories",
        "categories you have",
        "what categories do you have",
        "show me what you have",
    ]

    user_lower = user_message.lower()
    if any(greet in user_lower for greet in greetings) or any(
        meta in user_lower for meta in meta_queries
    ):
        state["intent"] = "meta"
        print(
            f"Classified intent for '{user_message}' as: meta (greeting/meta detected)"
        )
        return state

    # 2. Try to retrieve relevant documents to see if this is a product query
    try:
        docs = retriever.retrieve(user_message, k=3) if retriever else []
        if docs and len(docs) > 0:
            context = "\n".join([doc.page_content for doc in docs])
            # 3. Use LLM to classify intent based on retrieved context
            classification_prompt = f"""
            You are an intent classifier for a home decor product catalog.

            User message: "{user_message}"

            Retrieved context: {context[:500]}...

            Classify the user's intent into one of these categories:
            1. "product" - if the user is asking about specific products, categories, or product-related queries
            2. "meta" - if the user is greeting, saying hello, asking about the assistant's capabilities, help, or general questions (e.g., 'hi', 'hello', 'who are you', 'what can you do', 'help', 'about', 'catalog', etc.)
            3. "invalid" - if the query is completely unrelated to home decor products or is just chit-chat

            Examples:
            - "hi" => meta
            - "show me bedside tables" => product
            - "what can you do?" => meta
            - "weather today" => invalid
            - "I want a rug under 10k" => product
            - "hello" => meta

            Respond with ONLY the category name: product, meta, or invalid
            """
            intent_response = llm.predict(classification_prompt) if llm else ""
            intent = intent_response.strip().lower()
            if intent in ["product", "meta", "invalid"]:
                state["intent"] = intent
            else:
                state["intent"] = "product"
        else:
            # No relevant documents found, check if it's a meta query
            state["intent"] = "invalid"
    except Exception as e:
        print(f"Error in classify_node: {e}")
        state["intent"] = "invalid"
    print(f"Classified intent for '{user_message}' as: {state['intent']}")
    return state


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from the vector store."""
    retriever = state.get("retriever")
    user_message = state.get("user_message", "")

    # Get more documents for the final response (we already got 3 in classify_node)
    docs = retriever.retrieve(user_message, k=20) if retriever else []
    state["retrieved_docs"] = docs
    print(f"Retrieved {len(docs)} documents for response generation")
    return state


import json


def reason_node(state: AgentState) -> AgentState:
    """Generate a final answer using the LLM, grounded in retrieved docs."""
    llm = state.get("llm")
    docs = state.get("retrieved_docs") or []
    user_message = state.get("user_message", "")
    context = "\n".join([doc.page_content for doc in docs])
    prompt = (
        "You are a strict, helpful assistant for a home decor product catalog.\n"
        "Use ONLY the provided context to answer. If the context does NOT contain relevant information, do NOT attempt to answer.\n"
        "If the context lacks sufficient detail, ask a clear follow-up question instead of assuming.\n"
        "\n"
        "CRITICAL ANALYSIS RULES:\n"
        "1. FIRST, check if the requested category exists in the context\n"
        "2. If category doesn't exist, return a special response indicating category not found\n"
        "3. If category exists but no products match the budget, then return budget-related response\n"
        "4. If category exists and products match budget, return normal product suggestions\n"
        "\n"
        "RESPONSE FORMATS:\n"
        "\n"
        "If category NOT found:\n"
        "{{\n"
        '  "type": "category_not_found",\n'
        '  "requested_category": "the category user asked for",\n'
        '  "available_categories": ["list", "of", "available", "categories"],\n'
        '  "message": "We don\'t have the requested category in our catalog. Here are our available categories."\n'
        "}}\n"
        "\n"
        "If category found but no products match budget:\n"
        "{{\n"
        '  "type": "budget_constraint",\n'
        '  "category": "the category user asked for",\n'
        '  "requested_budget": "user\'s budget",\n'
        '  "message": "We found the requested category but none under your budget."\n'
        "}}\n"
        "\n"
        "If products found:\n"
        "{{\n"
        '  "type": "product_suggestion",\n'
        '  "summary": "Summary of suggestions.",\n'
        '  "products": [\n'
        '    {{"name": "Product Name", "price": "Price", "url": "Product URL"}},\n'
        "    ...\n"
        "  ]\n"
        "}}\n"
        "\n"
        "IMPORTANT: Use double quotes (\") for JSON keys and string values, NOT single quotes (').\n"
        "DO NOT reply in free text.\n"
        "\n"
        "Context:\n{context}\n\n"
        "User Message:\n{user_message}\n\n"
        "Assistant:"
    )
    full_prompt = prompt.format(context=context, user_message=user_message)
    response = llm.predict(full_prompt) if llm else ""

    # Try to parse the response as JSON
    try:
        # Clean up the response to extract JSON
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Try to parse as JSON first
        try:
            parsed_response = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to convert Python dict to JSON
            # Replace single quotes with double quotes for JSON compatibility
            import re

            # Convert Python-style dict to JSON
            json_text = response_text.replace("'", '"')
            # Handle None -> null
            json_text = json_text.replace('"None"', "null")
            # Handle True/False
            json_text = json_text.replace('"True"', "true")
            json_text = json_text.replace('"False"', "false")
            parsed_response = json.loads(json_text)

        if isinstance(parsed_response, dict):
            response_type = parsed_response.get("type")
            if response_type in [
                "product_suggestion",
                "category_not_found",
                "budget_constraint",
            ]:
                # Return the parsed object, not a string
                state["llm_response"] = parsed_response
            else:
                state["llm_response"] = response
        else:
            state["llm_response"] = response
    except (json.JSONDecodeError, KeyError) as e:
        print(f"JSON parsing error: {e}")
        print(f"Response text: {response_text}")
        # If JSON parsing fails, return as regular text
        state["llm_response"] = response

    return state


def clarify_node(state: AgentState) -> AgentState:
    """Ask for missing info (e.g., room type, budget)."""
    llm = state.get("llm")
    user_message = state.get("user_message", "")
    prompt = (
        "You are a helpful assistant specialized in home decor products.\n"
        "If the user's request is missing important details (like room type, budget, preferred style, or quantity), ask a clear, concise follow-up question to gather that missing information.\n"
        "Do NOT attempt to answer unless required details are available.\n"
        "\n"
        "Example Follow-up:\n"
        "What type of room is this product for? Or do you have a preferred budget range?\n"
        "\n"
        "User Request:\n{user_message}\n\n"
        "Your follow-up question:"
    )
    response = llm.predict(prompt.format(user_message=user_message)) if llm else ""
    state["llm_response"] = response
    return state


def reject_node(state: AgentState) -> AgentState:
    """Respond with fallback for irrelevant queries."""
    user_message = state.get("user_message", "")
    retriever = state.get("retriever")
    llm = state.get("llm")

    # Use LLM to intelligently discover available categories from the vector store
    try:
        # Get a sample of documents to understand what's available
        sample_docs = retriever.retrieve("product", k=10) if retriever else []

        if sample_docs and llm:
            # Create context from sample documents
            context = "\n".join([doc.page_content for doc in sample_docs])

            # Use LLM to extract categories intelligently
            category_prompt = f"""
            Based on the following product catalog data, extract 3-5 main product categories or types that customers might ask about.
            
            Catalog Data:
            {context[:1000]}...
            
            Extract only the main product categories (e.g., "bedside tables", "sofas", "curtains", "rugs", "fabrics").
            Return ONLY a comma-separated list of categories, nothing else.
            """

            category_response = llm.predict(category_prompt)
            categories = [
                cat.strip() for cat in category_response.split(",") if cat.strip()
            ]

            if categories:
                category_list = ", ".join(categories[:5])  # Show up to 5 categories
                state["llm_response"] = (
                    f"I'm a home decor assistant. I can help you find products like {category_list}, and more. "
                    f"Try asking about specific products or categories you're interested in!"
                )
            else:
                state["llm_response"] = (
                    "I'm a home decor assistant. I can help you find various home decor products. "
                    "Try asking about specific products you're looking for!"
                )
        else:
            # Fallback if no documents or LLM
            state["llm_response"] = (
                "I'm a home decor assistant. I can help you find various home decor products. "
                "Try asking about specific products you're looking for!"
            )

    except Exception as e:
        print(f"Error in reject_node: {e}")
        # Ultimate fallback - no hardcoded assumptions
        state["llm_response"] = (
            "I'm a home decor assistant. I can help you find various home decor products. "
            "Try asking about specific products you're looking for!"
        )

    return state


def meta_node(state: AgentState) -> AgentState:
    """Provide information about available categories and capabilities."""
    user_message = state.get("user_message", "")
    retriever = state.get("retriever")
    llm = state.get("llm")

    # List of queries that should trigger category listing
    category_queries = [
        "what categories",
        "what products",
        "what do you have",
        "show me categories",
        "list categories",
        "available products",
        "available categories",
        "what can i buy",
        "what's available",
        "list the categories",
        "categories you have",
        "what categories do you have",
    ]
    user_lower = user_message.lower()

    if any(query in user_lower for query in category_queries):
        try:
            # Retrieve a sample of documents
            sample_docs = retriever.retrieve("product", k=50) if retriever else []
            categories = set()
            for doc in sample_docs:
                # Try to extract from metadata
                if hasattr(doc, "metadata") and "category" in doc.metadata:
                    categories.add(doc.metadata["category"].strip().lower())
            if categories:
                # Return categories as an array for the frontend to display as chips
                state["llm_response"] = sorted(categories)
                return state
            # Fallback to LLM/text if no metadata categories found
            if sample_docs and llm:
                context = "\n".join([doc.page_content for doc in sample_docs])
                category_prompt = (
                    "From the following product catalog data, extract the main product categories that are actually available. "
                    "If you only see one category (like 'bedside tables'), return just that. "
                    "Return ONLY a comma-separated list of categories, nothing else.\n\n"
                    f"{context[:1500]}"
                )
                category_response = llm.predict(category_prompt)
                categories = [
                    cat.strip() for cat in category_response.split(",") if cat.strip()
                ]
                if categories:
                    # Return categories as an array for the frontend to display as chips
                    state["llm_response"] = categories
                    return state
        except Exception as e:
            print(f"Error in meta_node category detection: {e}")
        # Fallback
        state["llm_response"] = (
            "I can help you find various home decor products! "
            "You can ask me about specific products, prices, or request items within your budget."
        )
    else:
        # General greeting/help
        state["llm_response"] = (
            "Hello! I'm your home decor assistant. I can help you find furniture, fabrics, rugs, and other home decor products. "
            "You can ask me about specific products, prices, or request items within your budget. "
            "What would you like to find today?"
        )
    return state


# --- LangGraph Workflow ---


def build_langgraph_agent(retriever, openai_api_key: str):
    llm = ChatOpenAI(api_key=SecretStr(openai_api_key), temperature=0)

    state_schema = AgentState
    graph = StateGraph(state_schema)

    # Register nodes
    # graph.add_node(Node("classify", classify_node))
    # graph.add_node(Node("retrieve", retrieve_node))
    # graph.add_node(Node("reason", reason_node))
    # graph.add_node(Node("clarify", clarify_node))
    # graph.add_node(Node("reject", reject_node))

    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("reject", reject_node)
    graph.add_node("meta", meta_node)

    # Edges: classify -> retrieve/reject/clarify/meta
    # graph.add_edge("classify", "retrieve", condition=lambda s: s["intent"] == "product")
    # graph.add_edge("classify", "meta", condition=lambda s: s["intent"] == "meta")
    # graph.add_edge("classify", "reject", condition=lambda s: s["intent"] == "invalid")

    graph.add_conditional_edges(
        "classify",
        lambda s: s["intent"],
        {
            "product": "retrieve",
            "meta": "meta",
            "invalid": "reject",
        },
    )

    # retrieve -> reason
    graph.add_edge("retrieve", "reason")
    # reason -> END
    graph.add_edge("reason", END)
    # clarify -> END
    graph.add_edge("clarify", END)
    # reject -> END
    graph.add_edge("reject", END)

    # Set start node
    graph.set_entry_point("classify")
    app = graph.compile()

    def run_agent(user_message: str):
        state: AgentState = {
            "user_message": user_message,
            "retriever": retriever,
            "llm": llm,
        }
        # Classify intent
        state = classify_node(state)
        # Run the graph
        result = app.invoke(state)
        return result.get("llm_response", "Sorry, I couldn't process your request.")

    return run_agent
