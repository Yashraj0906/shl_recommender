"""Quick test to verify reranking improves results."""
import retriever

retriever.load_index()

test_cases = [
    "Python backend developer with Django and REST API skills",
    "contact center agents customer service",
    "Java developer Spring Boot coding",
    "senior manager leadership personality assessment",
]

for query in test_cases:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")
    results = retriever.search(query, top_k=5)
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['name']} (score: {r['relevance_score']})")
