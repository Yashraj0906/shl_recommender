"""
Automated test runner: Simulates the SHL grader against all 10 public traces.
Measures Recall@10 for each conversation.
"""
import json
import re
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import retriever
import agent

# Expected URLs from each conversation trace (ground truth)
GROUND_TRUTH = {
    "C1": [
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/",
        "https://www.shl.com/products/product-catalog/view/opq-leadership-report/",
    ],
    "C2": [
        "https://www.shl.com/products/product-catalog/view/smart-interview-live-coding/",
        "https://www.shl.com/products/product-catalog/view/linux-programming-general/",
        "https://www.shl.com/products/product-catalog/view/networking-and-implementation-new/",
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C3": [
        "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/",
        "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/",
        "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/",
        "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/",
    ],
    "C4": [
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/",
        "https://www.shl.com/products/product-catalog/view/financial-accounting-new/",
        "https://www.shl.com/products/product-catalog/view/basic-statistics-new/",
        "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C5": [
        "https://www.shl.com/products/product-catalog/view/global-skills-assessment/",
        "https://www.shl.com/products/product-catalog/view/global-skills-development-report/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "https://www.shl.com/products/product-catalog/view/opq-mq-sales-report/",
        "https://www.shl.com/products/product-catalog/view/salestransformationreport2-0-individualcontributor/",
    ],
    "C6": [
        "https://www.shl.com/products/product-catalog/view/safety-and-dependability-focus-8-0/",
        "https://www.shl.com/products/product-catalog/view/workplace-health-and-safety-new/",
    ],
    "C7": [
        "https://www.shl.com/products/product-catalog/view/hipaa-security/",
        "https://www.shl.com/products/product-catalog/view/medical-terminology-new/",
        "https://www.shl.com/products/product-catalog/view/microsoft-word-365-essentials-new/",
        "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C8": [
        "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/",
        "https://www.shl.com/products/product-catalog/view/microsoft-word-365-new/",
        "https://www.shl.com/products/product-catalog/view/ms-excel-new/",
        "https://www.shl.com/products/product-catalog/view/ms-word-new/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C9": [
        "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
        "https://www.shl.com/products/product-catalog/view/spring-new/",
        "https://www.shl.com/products/product-catalog/view/sql-new/",
        "https://www.shl.com/products/product-catalog/view/amazon-web-services-aws-development-new/",
        "https://www.shl.com/products/product-catalog/view/docker-new/",
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C10": [
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
        "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
    ],
}

# Simplified single-turn test messages (simulating what grader sends)
TEST_MESSAGES = {
    "C1": [
        {"role": "user", "content": "We need a solution for senior leadership. CXOs, director-level, 15+ years experience. Selection - comparing candidates against a leadership benchmark."},
    ],
    "C2": [
        {"role": "user", "content": "I'm hiring a senior Rust engineer for high-performance networking infrastructure. Include cognitive tests and personality assessment."},
    ],
    "C3": [
        {"role": "user", "content": "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. English US."},
    ],
    "C4": [
        {"role": "user", "content": "Hiring graduate financial analysts - final-year students, no work experience. We need numerical reasoning, finance knowledge, and situational judgement."},
    ],
    "C5": [
        {"role": "user", "content": "We need to re-skill our Sales organization as part of restructuring. Need skills assessment, personality, and sales-specific reports."},
    ],
    "C6": [
        {"role": "user", "content": "We're hiring plant operators for a chemical facility. Safety is absolute top priority - reliability, procedure compliance, never cutting corners. Industrial setting."},
    ],
    "C7": [
        {"role": "user", "content": "We're hiring bilingual healthcare admin staff in South Texas. They handle patient records, HIPAA compliance is critical. They're functionally bilingual English/Spanish. Need knowledge tests and personality."},
    ],
    "C8": [
        {"role": "user", "content": "I need to quickly screen admin assistants for Excel and Word skills. Include simulations and personality assessment."},
    ],
    "C9": [
        {"role": "user", "content": "Senior Full-Stack Engineer JD: 5+ years, Core Java, Spring, REST API, SQL, AWS, Docker. Backend-leaning senior IC. Need technical tests plus cognitive and personality."},
    ],
    "C10": [
        {"role": "user", "content": "We run a graduate management trainee scheme. We need cognitive, personality, and situational judgement assessments. All recent graduates."},
    ],
}


def compute_recall_at_k(predicted_urls, ground_truth_urls, k=10):
    """Recall@K = fraction of relevant items found in top K predictions."""
    predicted_set = set(predicted_urls[:k])
    relevant_set = set(ground_truth_urls)
    if not relevant_set:
        return 1.0
    hits = len(predicted_set & relevant_set)
    return hits / len(relevant_set)


def main():
    print("Loading retrieval engine...")
    retriever.load_index()
    print("Engine loaded!\n")
    
    total_recall = 0
    num_traces = 0
    
    for trace_id in sorted(TEST_MESSAGES.keys()):
        messages = TEST_MESSAGES[trace_id]
        ground_truth = GROUND_TRUTH[trace_id]
        
        print(f"{'='*60}")
        print(f"TRACE {trace_id}")
        print(f"Query: {messages[0]['content'][:80]}...")
        print(f"Expected: {len(ground_truth)} items")
        
        # Run agent
        result = agent.run_agent(messages)
        
        # Extract predicted URLs
        predicted_urls = [r.url for r in result.recommendations]
        
        # Compute recall
        recall = compute_recall_at_k(predicted_urls, ground_truth)
        total_recall += recall
        num_traces += 1
        
        print(f"Got: {len(result.recommendations)} recommendations")
        print(f"Recall@10: {recall:.2%}")
        
        # Show hits and misses
        for url in ground_truth:
            name = url.split("/view/")[1].rstrip("/") if "/view/" in url else url
            hit = "HIT" if url in predicted_urls else "MISS"
            print(f"  [{hit}] {name}")
        
        print()
    
    mean_recall = total_recall / num_traces if num_traces > 0 else 0
    print(f"{'='*60}")
    print(f"MEAN RECALL@10: {mean_recall:.2%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
