from app.agents.graph import ask_film_assistant

if __name__ == "__main__":
    question = input("Ask Film Director AI Assistant: ")
    result = ask_film_assistant(question)

    print("\nSelected route:", result["route"])
    print("\nAnswer:\n", result["answer"])

    if result["sources"]:
        print("\nPDF sources:")
        for source in result["sources"]:
            print(source)

    if result["web_results"]:
        print("\nWeb sources:")
        for item in result["web_results"]:
            print(item["title"], "-", item["url"])