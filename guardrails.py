"""
Guardrails: Post-generation validation layer.

This is the LAST LINE OF DEFENSE against hallucination.
Every response passes through here before reaching the user.
"""

import json
import re
from typing import List, Dict, Set

from schemas import Assessment, ChatResponse


def load_valid_urls(catalog_path: str = "catalog.json") -> Set[str]:
    """Load all valid URLs from catalog.json."""
    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        return {item["url"] for item in catalog}
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def load_valid_names(catalog_path: str = "catalog.json") -> Set[str]:
    """Load all valid assessment names from catalog.json."""
    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        return {item["name"] for item in catalog}
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def validate_recommendations(
    recommendations: List[Dict],
    valid_urls: Set[str],
    catalog: List[Dict]
) -> List[Assessment]:
    """
    Validate every recommendation against the catalog.
    
    - Strips any recommendation with a URL not in catalog.json
    - Corrects test_type if it doesn't match the catalog
    - Returns only valid Assessment objects
    
    This is a HARD programmatic check — no LLM involved.
    """
    validated = []
    catalog_by_url = {item["url"]: item for item in catalog}
    catalog_by_name = {item["name"].lower(): item for item in catalog}
    
    for rec in recommendations:
        url = rec.get("url", "")
        name = rec.get("name", "")
        
        # Strategy 1: URL matches exactly
        if url in valid_urls:
            catalog_item = catalog_by_url[url]
            validated.append(Assessment(
                name=catalog_item["name"],  # Use catalog's exact name
                url=url,
                test_type=catalog_item.get("test_type", rec.get("test_type", "K"))
            ))
            continue
        
        # Strategy 2: Name matches (URL might be slightly wrong)
        name_lower = name.lower()
        if name_lower in catalog_by_name:
            catalog_item = catalog_by_name[name_lower]
            validated.append(Assessment(
                name=catalog_item["name"],
                url=catalog_item["url"],  # Use the CORRECT URL from catalog
                test_type=catalog_item.get("test_type", rec.get("test_type", "K"))
            ))
            continue
        
        # Strategy 3: Fuzzy name match (handle minor LLM variations)
        for catalog_name, catalog_item in catalog_by_name.items():
            if (name_lower in catalog_name) or (catalog_name in name_lower):
                validated.append(Assessment(
                    name=catalog_item["name"],
                    url=catalog_item["url"],
                    test_type=catalog_item.get("test_type", rec.get("test_type", "K"))
                ))
                break
        # If no match found at all → silently drop the hallucinated recommendation
    
    return validated


def infer_test_type(name: str) -> str:
    """Infer the correct test_type from the assessment name.
    
    Types: A=Ability/Aptitude, P=Personality, B=Biodata/SJT,
           S=Simulation, K=Knowledge, C=Competency, D=Development
    """
    name_lower = name.lower()
    
    # Personality & Behavior (P)
    if any(kw in name_lower for kw in [
        "opq", "personality", "motivation questionnaire",
        "dependability and safety", "dsi"
    ]):
        return "P"
    
    # Ability & Aptitude (A)
    if any(kw in name_lower for kw in [
        "verify", "reasoning", "numerical ability",
        "verbal ability", "inductive", "deductive",
        "cognitive", "g+"
    ]):
        return "A"
    
    # Simulations (S)
    if any(kw in name_lower for kw in [
        "simulation", "interactive", "live coding"
    ]):
        return "S"
    
    # Biodata & Situational Judgment (B)
    if any(kw in name_lower for kw in [
        "scenarios", "situational", "sjt",
        "judgement", "judgment"
    ]):
        return "B"
    
    # Competency (C)
    if any(kw in name_lower for kw in [
        "competency", "competencies", "global skills assessment"
    ]):
        return "C"
    
    # Development (D)
    if any(kw in name_lower for kw in [
        "development report", "development"
    ]):
        return "D"
    
    # Default to Knowledge (K)
    return "K"


def enforce_response_schema(
    reply: str,
    recommendations: List[Assessment],
    end_of_conversation: bool
) -> ChatResponse:
    """
    Enforce the non-negotiable response schema.
    
    Guarantees:
    - reply is always a non-empty string
    - recommendations is always a list (never null)
    - end_of_conversation is always a boolean
    - recommendations has 0-10 items
    - test_type is correctly inferred from name
    """
    # Ensure reply is never empty
    if not reply or not reply.strip():
        reply = "I can help you find the right SHL assessments. What role are you hiring for?"
    
    # Cap recommendations at 10
    if len(recommendations) > 10:
        recommendations = recommendations[:10]
    
    # Fix test_type for all recommendations
    for rec in recommendations:
        rec.test_type = infer_test_type(rec.name)
    
    # Ensure end_of_conversation is boolean
    end_of_conversation = bool(end_of_conversation)
    
    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation
    )


def detect_prompt_injection(message: str) -> bool:
    """
    Detect common prompt injection patterns.
    Returns True if injection is detected.
    """
    injection_patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above",
        r"forget\s+(all\s+)?previous",
        r"you\s+are\s+now\s+a",
        r"act\s+as\s+a\s+different",
        r"system\s*:\s*",
        r"override\s+(the\s+)?system",
        r"new\s+instructions?\s*:",
        r"disregard\s+(all\s+)?prior",
        r"pretend\s+you\s+are",
        r"jailbreak",
        r"DAN\s+mode",
    ]
    
    message_lower = message.lower()
    for pattern in injection_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False
