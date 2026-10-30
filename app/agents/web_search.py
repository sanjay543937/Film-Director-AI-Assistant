
import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def search_web(question: str, max_results: int = 5) -> dict:
    """Search the web for filmmaking information."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY is missing from your .env file.")

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=question,
        search_depth="basic",
        max_results=max_results,
        include_answer=False,
    )

    results = []

    for item in response.get("results", []):
        results.append({
            "title": item.get("title", "Untitled"),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
        })

    return {
        "question": question,
        "results": results,
    }



if __name__ == "__main__":

    print("\n========================================")
    print(" FILM DIRECTOR AI - WEB SEARCH")
    print("========================================")
    print("Ask multiple questions.")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! Happy filmmaking.")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        try:
            result = search_web(question, max_results=5)

            print("\nWEB SEARCH RESULTS")
            print("-" * 50)

            if not result["results"]:
                print("No results found.")
            
            for index, item in enumerate(result["results"], start=1):
                print(f"\nResult {index}")
                print("Title:", item["title"])
                print("URL:", item["url"])
                print("Content:", item["content"][:500])

        except Exception as error:
            print("\nSearch error:", error)

        print("\n" + "=" * 50 + "\n")