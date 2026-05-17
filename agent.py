"""
LangGraph Agent for SHL Assessment Recommendation.

Implements a StateGraph that processes stateless conversation history
and routes to the appropriate action: clarify, recommend, refine, compare, or refuse.

The agent receives the FULL conversation on every call (stateless API).
"""

import os
import json
import re
from typing import TypedDict, List, Dict, Optional, Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

import retriever
import guardrails
from prompts import (
    SYSTEM_PROMPT,
    CATALOG_CONTEXT_TEMPLATE,
    INTENT_CLASSIFICATION_PROMPT,
    QUERY_EXTRACTION_PROMPT,
)
from schemas import Assessment, ChatResponse

load_dotenv()


# ──────────────────────────────────────────────
# LLM Setup
# ──────────────────────────────────────────────

def _get_llm(temperature: float = 0.1):
    """Get Groq LLM instance. Low temperature for consistency."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        max_tokens=1500,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _llm_invoke_with_retry(llm, messages, max_retries=3):
    """Invoke LLM with retry logic for rate limits."""
    import time
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                print(f"  [RATE LIMIT] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise
    # Final attempt without catch
    return llm.invoke(messages)


# ──────────────────────────────────────────────
# Agent State
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    """Typed state for the LangGraph agent."""
    messages: List[Dict]                    # Full conversation history
    turn_count: int                         # Current turn number
    intent: str                             # Classified intent
    search_query: str                       # Optimized search query
    catalog_context: List[Dict]             # Retrieved assessments
    reply: str                              # Agent's response text
    recommendations: List[Dict]             # Final recommendations
    end_of_conversation: bool               # Whether to end


# ──────────────────────────────────────────────
# Agent Nodes
# ──────────────────────────────────────────────

def classify_intent(state: AgentState) -> AgentState:
    """
    Entry node: Classify the user's intent to determine routing.
    
    Routes to: clarify | retrieve | compare | refuse | end
    """
    messages = state["messages"]
    turn_count = state["turn_count"]
    
    # Check for prompt injection in the latest user message
    latest_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            latest_user_msg = msg["content"]
            break
    
    if guardrails.detect_prompt_injection(latest_user_msg):
        return {**state, "intent": "refuse"}
    
    # Programmatic off-topic detection (catches salary, legal advice, etc.)
    # CAREFUL: Don't catch legitimate assessment queries about compliance/safety
    off_topic_patterns = [
        r"\bwhat\s+salary\b", r"\bhow\s+much\s+(should|do|does|to)\s+.*\bpay\b",
        r"\bsalary\s+(range|band|scale|offer)\b", r"\bcompensation\s+package\b",
        r"\binterview\s+(tips|techniques|questions)\b",
        r"\bresume\b", r"\bcv\b", r"\bcover\s+letter\b",
        r"\bwrite\s+(a|my|the)\s+(email|code|essay|letter)\b",
        r"\btell\s+me\s+(a\s+)?joke\b", r"\bweather\b",
        r"\bwhat\s+should\s+I\s+offer\b",
    ]
    msg_lower = latest_user_msg.lower()
    for pattern in off_topic_patterns:
        if re.search(pattern, msg_lower):
            return {**state, "intent": "refuse"}
    
    # Check if previous recommendations exist in conversation
    has_prior_recs = any(
        msg["role"] == "assistant" and "http" in msg.get("content", "")
        for msg in messages
    )
    
    # Build conversation summary for intent classification
    conversation_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in messages
    )
    
    # Use LLM to classify intent
    llm = _get_llm(temperature=0.0)
    
    prompt = INTENT_CLASSIFICATION_PROMPT.format(
        turn_count=turn_count
    )
    
    response = _llm_invoke_with_retry(llm, [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Conversation:\n{conversation_text}")
    ])
    
    intent_text = response.content.strip().lower()
    
    # Parse intent (handle LLM verbosity)
    valid_intents = ["clarify", "recommend", "refine", "compare", "refuse", "end"]
    intent = "recommend"  # Default fallback
    
    for valid in valid_intents:
        if valid in intent_text:
            intent = valid
            break
    
    # Force recommend if we're running out of turns
    if turn_count >= 6 and intent in ("clarify",):
        intent = "recommend"
    
    # Treat "end" with prior recs as the final confirmation
    if intent == "end" and has_prior_recs:
        intent = "end"
    elif intent == "end" and not has_prior_recs:
        intent = "recommend"  # Can't end without recommending
    
    # Refine is a special case of recommend
    if intent == "refine":
        intent = "recommend"  # Same pipeline, but context-aware
    
    return {**state, "intent": intent}


def clarify(state: AgentState) -> AgentState:
    """
    Ask ONE focused clarifying question.
    Returns empty recommendations.
    """
    messages = state["messages"]
    
    conversation_msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in messages:
        if msg["role"] == "user":
            conversation_msgs.append(HumanMessage(content=msg["content"]))
        else:
            conversation_msgs.append(AIMessage(content=msg["content"]))
    
    conversation_msgs.append(SystemMessage(
        content=(
            "The user's query is too vague to recommend assessments. "
            "Ask exactly ONE focused clarifying question to gather the most "
            "critical missing information. Be concise and specific. "
            "Do NOT recommend anything yet."
        )
    ))
    
    llm = _get_llm()
    response = _llm_invoke_with_retry(llm, conversation_msgs)
    
    return {
        **state,
        "reply": response.content,
        "recommendations": [],
        "end_of_conversation": False,
    }


def retrieve_and_recommend(state: AgentState) -> AgentState:
    """
    Core recommendation pipeline:
    1. Extract optimized search query from conversation
    2. Run hybrid search (FAISS + BM25 + RRF)
    3. Inject catalog context into LLM prompt
    4. Generate recommendations grounded in catalog
    """
    messages = state["messages"]
    turn_count = state["turn_count"]
    
    # Step 1: Extract search query from conversation
    conversation_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in messages
    )
    
    llm = _get_llm(temperature=0.0)
    
    query_response = _llm_invoke_with_retry(llm, [
        SystemMessage(content=QUERY_EXTRACTION_PROMPT.format(
            conversation=conversation_text
        ))
    ])
    
    # Parse multiple queries (one per line)
    raw_queries = query_response.content.strip().split("\n")
    search_queries = [q.strip().strip("0123456789.-) ") for q in raw_queries if q.strip()][:3]
    
    # Fallback: if LLM returned only 1 query, use it alone
    if not search_queries:
        search_queries = [conversation_text[-200:]]
    
    # Step 2: Multi-query hybrid search — run each query and merge results
    seen_urls = {}  # url -> item (keeps best score)
    for sq in search_queries:
        results = retriever.search(sq, top_k=10)
        for item in results:
            url = item["url"]
            if url not in seen_urls or item.get("relevance_score", 0) > seen_urls[url].get("relevance_score", 0):
                seen_urls[url] = item
    
    # Sort merged results by relevance score and take top 10
    search_results = sorted(seen_urls.values(), key=lambda x: x.get("relevance_score", 0), reverse=True)[:10]
    
    # Step 3: Build catalog context
    catalog_context = []
    for item in search_results:
        catalog_context.append({
            "name": item["name"],
            "url": item["url"],
            "test_type": item.get("test_type", "K"),
            "description": item.get("description", ""),
            "job_levels": item.get("job_levels", []),
            "duration_minutes": item.get("duration_minutes"),
            "remote_testing": item.get("remote_testing", False),
        })
    
    catalog_json = json.dumps(catalog_context, indent=2)
    
    # Step 4: Generate recommendation with LLM
    conversation_msgs = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=CATALOG_CONTEXT_TEMPLATE.format(
            catalog_json=catalog_json
        )),
    ]
    
    for msg in messages:
        if msg["role"] == "user":
            conversation_msgs.append(HumanMessage(content=msg["content"]))
        else:
            conversation_msgs.append(AIMessage(content=msg["content"]))
    
    # Add turn-budget awareness
    if turn_count >= 5:
        conversation_msgs.append(SystemMessage(
            content=(
                f"IMPORTANT: This is turn {turn_count} of 8 max. "
                "You MUST provide your final assessment recommendations NOW. "
                "Do not ask any more questions."
            )
        ))
    
    # Build a numbered menu of available assessments to force exact name usage
    assessment_menu = "\n".join(
        f'{i+1}. NAME: "{item["name"]}" | URL: "{item["url"]}" | TYPE: {item.get("test_type", "K")}'
        for i, item in enumerate(catalog_context)
    )
    
    conversation_msgs.append(SystemMessage(
        content=(
            "Based on the conversation, recommend the most relevant SHL assessments.\n\n"
            "AVAILABLE ASSESSMENTS (pick ONLY from this list):\n"
            f"{assessment_menu}\n\n"
            "RULES:\n"
            "- You MUST use the EXACT 'NAME' and 'URL' values from the list above\n"
            "- Do NOT rename, shorten, or invent assessment names\n"
            "- Do NOT modify URLs in any way\n"
            "- Recommend ALL relevant assessments (aim for 5-10). More is better.\n"
            "- Include personality tests (OPQ32r) when assessing people skills or leadership\n"
            "- Include cognitive tests (Verify G+) for senior/technical roles\n"
            "- For each, briefly explain why it fits the role\n\n"
            "After your explanation, output a JSON array:\n"
            "```json\n"
            "[{\"name\": \"EXACT name from list\", \"url\": \"EXACT url from list\", \"test_type\": \"K\"}]\n"
            "```"
        )
    ))
    
    response = _llm_invoke_with_retry(llm, conversation_msgs)
    reply_text = response.content
    
    # Step 5: Parse recommendations from LLM response
    recommendations = _parse_recommendations(reply_text, catalog_context)
    
    # CRITICAL FALLBACK: If LLM hallucinated names and guardrails stripped them,
    # use the top FAISS search results directly. This guarantees we never return
    # empty recommendations when we have relevant search results.
    if not recommendations and catalog_context:
        for item in catalog_context[:10]:  # Use ALL search results
            try:
                recommendations.append(Assessment(
                    name=item["name"],
                    url=item["url"],
                    test_type=item.get("test_type", "K"),
                ))
            except Exception:
                continue
    
    # Clean the reply (remove JSON block from user-facing text)
    clean_reply = re.sub(
        r'```json\s*\[.*?\]\s*```',
        '',
        reply_text,
        flags=re.DOTALL
    ).strip()
    
    if not clean_reply:
        clean_reply = reply_text  # Fallback if regex removes everything
    
    return {
        **state,
        "search_query": " | ".join(search_queries),
        "catalog_context": catalog_context,
        "reply": clean_reply,
        "recommendations": [r.model_dump() for r in recommendations],
        "end_of_conversation": False,  # Let user confirm
    }


def compare(state: AgentState) -> AgentState:
    """
    Compare two or more assessments using ONLY catalog data.
    No prior knowledge allowed.
    """
    messages = state["messages"]
    
    # Get the latest user message to find what they want to compare
    latest_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            latest_user_msg = msg["content"]
            break
    
    # Search for the assessments being compared
    search_results = retriever.search(latest_user_msg, top_k=10)
    catalog_json = json.dumps(search_results, indent=2)
    
    conversation_msgs = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=CATALOG_CONTEXT_TEMPLATE.format(
            catalog_json=catalog_json
        )),
    ]
    
    for msg in messages:
        if msg["role"] == "user":
            conversation_msgs.append(HumanMessage(content=msg["content"]))
        else:
            conversation_msgs.append(AIMessage(content=msg["content"]))
    
    conversation_msgs.append(SystemMessage(
        content=(
            "The user wants to compare assessments. Use ONLY the catalog "
            "data provided above. Do NOT use your prior knowledge about "
            "SHL products. Compare them factually based on: test type, "
            "description, duration, job levels, and any other catalog fields."
        )
    ))
    
    llm = _get_llm()
    response = _llm_invoke_with_retry(llm, conversation_msgs)
    
    return {
        **state,
        "reply": response.content,
        "recommendations": [],  # Comparisons don't produce new recs
        "end_of_conversation": False,
    }


def refuse(state: AgentState) -> AgentState:
    """
    Politely refuse off-topic queries or prompt injection attempts.
    Always returns empty recommendations.
    """
    messages = state["messages"]
    
    latest_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            latest_user_msg = msg["content"]
            break
    
    # Check if it's a prompt injection
    is_injection = guardrails.detect_prompt_injection(latest_user_msg)
    
    if is_injection:
        reply = (
            "I can only recommend assessments from the SHL catalog. "
            "I'm not able to modify my instructions or recommend "
            "non-SHL products. How can I help you find the right "
            "SHL assessment for your hiring needs?"
        )
    else:
        reply = (
            "I can only help with SHL assessment recommendations. "
            "For other questions (salary guidance, legal compliance, "
            "general HR advice), please consult the appropriate "
            "resources. Would you like help finding an SHL assessment "
            "for a specific role?"
        )
    
    return {
        **state,
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False,
    }


def finalize(state: AgentState) -> AgentState:
    """
    User confirmed satisfaction — end the conversation.
    Re-emit the last recommendations if they exist.
    """
    messages = state["messages"]
    
    # Find the most recent recommendations from conversation context
    # Since we're stateless, we need to re-derive them
    # The user is confirming, so we generate a brief closing
    
    conversation_msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in messages:
        if msg["role"] == "user":
            conversation_msgs.append(HumanMessage(content=msg["content"]))
        else:
            conversation_msgs.append(AIMessage(content=msg["content"]))
    
    conversation_msgs.append(SystemMessage(
        content=(
            "The user has confirmed they are satisfied. Provide a brief "
            "closing summary. If recommendations were given earlier in the "
            "conversation, re-state them in a final JSON block:\n"
            "```json\n"
            '[{"name": "...", "url": "...", "test_type": "..."}]\n'
            "```"
        )
    ))
    
    # Search based on full conversation to re-derive recs
    conversation_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in messages
    )
    
    llm = _get_llm()
    
    # Extract what was discussed to search
    query_response = _llm_invoke_with_retry(llm, [
        SystemMessage(content=QUERY_EXTRACTION_PROMPT.format(
            conversation=conversation_text
        ))
    ])
    search_results = retriever.search(query_response.content.strip(), top_k=10)
    catalog_json = json.dumps(search_results, indent=2)
    
    conversation_msgs.insert(1, SystemMessage(
        content=CATALOG_CONTEXT_TEMPLATE.format(catalog_json=catalog_json)
    ))
    
    response = _llm_invoke_with_retry(llm, conversation_msgs)
    reply_text = response.content
    
    recommendations = _parse_recommendations(reply_text, search_results)
    
    clean_reply = re.sub(
        r'```json\s*\[.*?\]\s*```', '', reply_text, flags=re.DOTALL
    ).strip()
    
    if not clean_reply:
        clean_reply = "Your assessment battery is confirmed. Good luck with your hiring!"
    
    return {
        **state,
        "reply": clean_reply,
        "recommendations": [r.model_dump() for r in recommendations],
        "end_of_conversation": True,
    }


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _parse_recommendations(
    llm_response: str,
    catalog_context: List[Dict]
) -> List[Assessment]:
    """
    Parse recommendations from LLM response text.
    Falls back to extracting from catalog context if JSON parsing fails.
    Then validates EVERY recommendation against the catalog.
    """
    recommendations = []
    
    # Try to extract JSON block
    json_match = re.search(
        r'```(?:json)?\s*(\[.*?\])\s*```',
        llm_response,
        re.DOTALL
    )
    
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, list):
                recommendations = parsed
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to find JSON array without code blocks
    if not recommendations:
        json_array_match = re.search(r'\[[\s\S]*?"name"[\s\S]*?\]', llm_response)
        if json_array_match:
            try:
                parsed = json.loads(json_array_match.group())
                if isinstance(parsed, list):
                    recommendations = parsed
            except json.JSONDecodeError:
                pass
    
    # Fallback: extract mentioned assessment names from text
    if not recommendations and catalog_context:
        for item in catalog_context:
            if item["name"].lower() in llm_response.lower():
                recommendations.append({
                    "name": item["name"],
                    "url": item["url"],
                    "test_type": item.get("test_type", "K"),
                })
    
    # GUARDRAIL: Validate all recommendations against catalog
    catalog = retriever.get_catalog()
    valid_urls = {item["url"] for item in catalog}
    validated = guardrails.validate_recommendations(
        recommendations, valid_urls, catalog
    )
    
    return validated


def _route_intent(state: AgentState) -> str:
    """Route to the correct node based on classified intent."""
    intent = state.get("intent", "recommend")
    if intent == "clarify":
        return "clarify"
    elif intent == "compare":
        return "compare"
    elif intent == "refuse":
        return "refuse"
    elif intent == "end":
        return "finalize"
    else:
        return "retrieve_and_recommend"


# ──────────────────────────────────────────────
# Graph Construction
# ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent."""
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("clarify", clarify)
    graph.add_node("retrieve_and_recommend", retrieve_and_recommend)
    graph.add_node("compare", compare)
    graph.add_node("refuse", refuse)
    graph.add_node("finalize", finalize)
    
    # Set entry point
    graph.set_entry_point("classify_intent")
    
    # Add conditional routing from classify_intent
    graph.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "clarify": "clarify",
            "retrieve_and_recommend": "retrieve_and_recommend",
            "compare": "compare",
            "refuse": "refuse",
            "finalize": "finalize",
        }
    )
    
    # All action nodes go to END
    graph.add_edge("clarify", END)
    graph.add_edge("retrieve_and_recommend", END)
    graph.add_edge("compare", END)
    graph.add_edge("refuse", END)
    graph.add_edge("finalize", END)
    
    return graph.compile()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

# Compile once at module level
_agent = build_graph()


def run_agent(messages: List[Dict]) -> ChatResponse:
    """
    Run the agent on a conversation.
    
    Args:
        messages: Full conversation history [{role, content}, ...]
    
    Returns:
        ChatResponse with reply, recommendations, and end_of_conversation
    """
    turn_count = len(messages)
    
    initial_state: AgentState = {
        "messages": messages,
        "turn_count": turn_count,
        "intent": "",
        "search_query": "",
        "catalog_context": [],
        "reply": "",
        "recommendations": [],
        "end_of_conversation": False,
    }
    
    try:
        result = _agent.invoke(initial_state)
        
        # Build Assessment objects from recommendations
        rec_objects = []
        for rec in result.get("recommendations", []):
            if isinstance(rec, dict):
                try:
                    rec_objects.append(Assessment(**rec))
                except Exception:
                    continue
            elif isinstance(rec, Assessment):
                rec_objects.append(rec)
        
        # SAFETY NET: If recs are empty after the agent ran,
        # do a direct search using the user's messages.
        # Only fires when user gave meaningful content (not just "hi").
        intent = result.get("intent", "")
        if not rec_objects and intent not in ("refuse", "clarify"):
            user_msgs = [m["content"] for m in messages if m["role"] == "user"]
            search_text = " ".join(user_msgs[-2:]) if user_msgs else ""
            
            # Only fire safety net if user gave real content (10+ chars)
            if search_text and len(search_text.strip()) >= 10:
                print(f"[SAFETY NET] Recs empty (intent={intent}), doing direct search...")
                search_results = retriever.search(search_text, top_k=10)
                catalog = retriever.get_catalog()
                valid_urls = {item["url"] for item in catalog}
                
                for item in search_results:
                    if item["url"] in valid_urls:
                        try:
                            rec_objects.append(Assessment(
                                name=item["name"],
                                url=item["url"],
                                test_type=item.get("test_type", "K"),
                            ))
                        except Exception:
                            continue
                
                print(f"[SAFETY NET] Added {len(rec_objects)} recommendations")
        
        # POST-REPLY REFUSAL CHECK: If the reply indicates a refusal,
        # strip all recommendations (catches cases where intent was wrong
        # but the LLM correctly refused in its reply text)
        reply_text = result.get("reply", "")
        refusal_phrases = [
            "cannot provide", "can't provide", "outside my scope",
            "not able to", "i can only recommend", "i'm not able",
            "cannot help with", "can't help with", "unable to assist",
            "not within my scope", "beyond my scope",
        ]
        reply_lower = reply_text.lower()
        if any(phrase in reply_lower for phrase in refusal_phrases):
            rec_objects = []  # Strip all recs on refusal
        
        return guardrails.enforce_response_schema(
            reply=reply_text,
            recommendations=rec_objects,
            end_of_conversation=result.get("end_of_conversation", False),
        )
    
    except Exception as e:
        # Graceful degradation: return a safe response on any error
        print(f"Agent error: {e}")
        return guardrails.enforce_response_schema(
            reply=(
                "I'm having trouble processing your request right now. "
                "Could you rephrase your question about SHL assessments?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )
