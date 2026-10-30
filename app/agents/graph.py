
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END

from app.rag.answer import generate_answer
from app.agents.web_search import search_web


# ==========================================
# 1. DEFINE THE SHARED STATE
# ==========================================

class FilmAssistantState(TypedDict, total=False):

    question: str

    route: Literal["rag", "web", "both"]

    rag_answer: str

    web_results: list[dict]

    answer: str

    sources: list[dict]


# ==========================================
# 2. ROUTER NODE
# ==========================================

def route_question(state: FilmAssistantState):

    question = state["question"].lower()

    current_info_terms = [
        "latest",
        "today",
        "recent",
        "current",
        "new release",
        "this year",
        "2026",
        "news",
        "upcoming",
    ]

    compare_terms = [
        "compare",
        "both",
        "according to my pdf and",
        "from the pdf and the web",
    ]

    if any(term in question for term in compare_terms):

        route = "both"

    elif any(term in question for term in current_info_terms):

        route = "web"

    else:

        route = "rag"

    print(f"\nSelected Route: {route.upper()}")

    return {"route": route}


# ==========================================
# 3. RAG NODE
# ==========================================

def run_rag(state: FilmAssistantState):

    print("\nSearching your filmmaking PDF...")

    result = generate_answer(state["question"])

    return {
        "rag_answer": result["answer"],
        "sources": result["sources"],
    }


# ==========================================
# 4. WEB SEARCH NODE
# ==========================================

def run_web(state: FilmAssistantState):

    print("\nSearching the internet using Tavily...")

    result = search_web(state["question"], max_results=5)

    return {
        "web_results": result["results"]
    }


# ==========================================
# 5. CONDITIONAL ROUTING AFTER RAG
# ==========================================

def after_rag(state: FilmAssistantState):

    if state["route"] == "both":

        return "web"

    return "combine"


# ==========================================
# 6. COMBINE NODE
# ==========================================

def combine_results(state: FilmAssistantState):

    print("\nCombining information from available sources...")

    parts = []

    # Collect PDF information

    if state.get("rag_answer"):

        parts.append(
            "PDF KNOWLEDGE:\n"
            + state["rag_answer"]
        )

    # Collect web information

    if state.get("web_results"):

        web_text = "\n\n".join(

            f"""
Title: {item['title']}

URL: {item['url']}

Content: {item['content']}
"""

            for item in state["web_results"]

        )

        parts.append(
            "WEB SEARCH RESULTS:\n"
            + web_text
        )

    final_answer = "\n\n".join(parts)

    return {
        "answer": final_answer
    }


# ==========================================
# 7. BUILD LANGGRAPH WORKFLOW
# ==========================================

builder = StateGraph(FilmAssistantState)


# Register nodes

builder.add_node("router", route_question)

builder.add_node("rag", run_rag)

builder.add_node("web", run_web)

builder.add_node("combine", combine_results)


# Start -> Router

builder.add_edge(START, "router")


# Router -> Select RAG or Web

builder.add_conditional_edges(

    "router",

    lambda state: state["route"],

    {
        "rag": "rag",
        "web": "web",
        "both": "rag",
    }

)


# RAG -> Web or Combine

builder.add_conditional_edges(

    "rag",

    after_rag,

    {
        "web": "web",
        "combine": "combine",
    }

)


# Web -> Combine

builder.add_edge("web", "combine")


# Combine -> End

builder.add_edge("combine", END)


# Compile the graph

graph = builder.compile()


# ==========================================
# 8. MAIN CHATBOT FUNCTION
# ==========================================

def ask_film_assistant(question: str) -> dict:

    if not question.strip():

        raise ValueError("Question cannot be empty.")

    result = graph.invoke({

        "question": question

    })

    return {

        "question": question,

        "route": result.get("route"),

        "answer": result.get("answer", ""),

        "sources": result.get("sources", []),

        "web_results": result.get("web_results", []),

    }


# ==========================================
# 9. RUN DIRECTLY FROM TERMINAL
# ==========================================

if __name__ == "__main__":

    print("\n======================================")

    print(" FILM DIRECTOR AI ASSISTANT ")

    print("======================================")

    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "bye","clear"]:

            print("Goodbye!")

            break

        if not question:

            print("Please enter a question.")

            continue

        try:

            result = ask_film_assistant(question)

            print("\nSelected Route:", result["route"])

            print("\nANSWER:\n")

            print(result["answer"])

            if result["sources"]:

                print("\nPDF SOURCES:")

                for source in result["sources"]:

                    print(
                        f"- {source['source']}, "
                        f"Page {source['page']}"
                    )

            if result["web_results"]:

                print("\nWEB SOURCES:")

                for item in result["web_results"]:

                    print(
                        f"- {item['title']}: "
                        f"{item['url']}"
                    )

        except Exception as error:

            print("\nError:", error)

        print("\n" + "=" * 50 + "\n")