#!/usr/bin/env python3
"""
Comprehensive test of the newly trained DeBERTa model
"""

import requests
import json
import time
from typing import List, Dict, Tuple


def test_trained_model():
    """Comprehensive test of all intents with the newly trained model"""

    print("🧪 COMPREHENSIVE MODEL TESTING")
    print("=" * 60)

    base_url = "http://localhost:8000/chat"

    # Comprehensive test cases covering all intents
    test_cases = [
        # ===== PRODUCT_SEARCH (Previously Failing) =====
        ("do you have rugs", "PRODUCT_SEARCH", "product_suggestion"),
        ("do you have furniture", "PRODUCT_SEARCH", "product_suggestion"),
        ("do you have curtains", "PRODUCT_SEARCH", "product_suggestion"),
        ("can you show rugs", "PRODUCT_SEARCH", "product_suggestion"),
        ("show me furniture", "PRODUCT_SEARCH", "product_suggestion"),
        ("I'm looking for rugs", "PRODUCT_SEARCH", "product_suggestion"),
        ("find me furniture", "PRODUCT_SEARCH", "product_suggestion"),
        ("get me some rugs", "PRODUCT_SEARCH", "product_suggestion"),
        # ===== NEW PRODUCT_SEARCH VARIATIONS =====
        ("do you sell rugs", "PRODUCT_SEARCH", "product_suggestion"),
        ("got any furniture", "PRODUCT_SEARCH", "product_suggestion"),
        ("have you got curtains", "PRODUCT_SEARCH", "product_suggestion"),
        ("looking for sofas", "PRODUCT_SEARCH", "product_suggestion"),
        ("need some chairs", "PRODUCT_SEARCH", "product_suggestion"),
        ("want to buy tables", "PRODUCT_SEARCH", "product_suggestion"),
        ("searching for lamps", "PRODUCT_SEARCH", "product_suggestion"),
        ("interested in mirrors", "PRODUCT_SEARCH", "product_suggestion"),
        ("checking for beds", "PRODUCT_SEARCH", "product_suggestion"),
        ("browse rugs", "PRODUCT_SEARCH", "product_suggestion"),
        ("explore furniture", "PRODUCT_SEARCH", "product_suggestion"),
        # ===== BUDGET_QUERY (Previously Failing) =====
        ("show rugs under 5000", "BUDGET_QUERY", "budget_constraint"),
        ("show furniture under 10000", "BUDGET_QUERY", "budget_constraint"),
        ("show curtains under 3000", "BUDGET_QUERY", "budget_constraint"),
        ("furniture under 10000", "BUDGET_QUERY", "budget_constraint"),
        ("what can I get for under 5000", "BUDGET_QUERY", "budget_constraint"),
        ("rugs within 5000 budget", "BUDGET_QUERY", "budget_constraint"),
        # ===== NEW BUDGET_QUERY VARIATIONS =====
        ("sofas under 15000", "BUDGET_QUERY", "budget_constraint"),
        ("chairs below 5000", "BUDGET_QUERY", "budget_constraint"),
        ("tables less than 8000", "BUDGET_QUERY", "budget_constraint"),
        ("lamps under 2000", "BUDGET_QUERY", "budget_constraint"),
        ("mirrors within 3000", "BUDGET_QUERY", "budget_constraint"),
        ("beds under 25000", "BUDGET_QUERY", "budget_constraint"),
        ("curtains below 4000", "BUDGET_QUERY", "budget_constraint"),
        ("furniture around 12000", "BUDGET_QUERY", "budget_constraint"),
        ("rugs up to 6000", "BUDGET_QUERY", "budget_constraint"),
        ("accessories under 1000", "BUDGET_QUERY", "budget_constraint"),
        # ===== HELP (Previously Failing) =====
        ("help", "HELP", "help"),
        ("I need help", "HELP", "help"),
        ("can you help me", "HELP", "help"),
        ("what can you do", "HELP", "help"),
        ("how do you work", "HELP", "help"),
        ("I'm confused", "HELP", "help"),
        ("how do I search", "HELP", "help"),
        # ===== NEW HELP VARIATIONS =====
        ("help me", "HELP", "help"),
        ("need assistance", "HELP", "help"),
        ("how to use this", "HELP", "help"),
        ("what are my options", "HELP", "help"),
        ("guide me", "HELP", "help"),
        ("I don't understand", "HELP", "help"),
        ("explain how this works", "HELP", "help"),
        ("show me how to search", "HELP", "help"),
        ("how can I find products", "HELP", "help"),
        ("what's available", "HELP", "help"),
        # ===== GREETING (Should Work) =====
        ("hello", "GREETING", "greeting"),
        ("hi", "GREETING", "greeting"),
        ("good morning", "GREETING", "greeting"),
        ("hey", "GREETING", "greeting"),
        # ===== NEW GREETING VARIATIONS =====
        ("good afternoon", "GREETING", "greeting"),
        ("good evening", "GREETING", "greeting"),
        ("hi there", "GREETING", "greeting"),
        ("hello there", "GREETING", "greeting"),
        ("greetings", "GREETING", "greeting"),
        ("good day", "GREETING", "greeting"),
        ("morning", "GREETING", "greeting"),
        ("afternoon", "GREETING", "greeting"),
        ("evening", "GREETING", "greeting"),
        ("yo", "GREETING", "greeting"),
        # ===== CATEGORY_LIST (Should Work) =====
        ("what categories do you have", "CATEGORY_LIST", "category_list"),
        ("show me categories", "CATEGORY_LIST", "category_list"),
        ("what do you sell", "CATEGORY_LIST", "category_list"),
        ("what products do you have", "CATEGORY_LIST", "category_list"),
        # ===== NEW CATEGORY_LIST VARIATIONS =====
        ("list categories", "CATEGORY_LIST", "category_list"),
        ("show all categories", "CATEGORY_LIST", "category_list"),
        ("what's available", "CATEGORY_LIST", "category_list"),
        ("browse categories", "CATEGORY_LIST", "category_list"),
        ("what can I buy", "CATEGORY_LIST", "category_list"),
        ("show me what you have", "CATEGORY_LIST", "category_list"),
        ("what are the categories", "CATEGORY_LIST", "category_list"),
        ("list all products", "CATEGORY_LIST", "category_list"),
        ("what do you offer", "CATEGORY_LIST", "category_list"),
        ("show product categories", "CATEGORY_LIST", "category_list"),
        # ===== CLARIFY (Should Work) =====
        ("I need something for my living room", "CLARIFY", "clarification"),
        ("looking for bedroom items", "CLARIFY", "clarification"),
        ("need bathroom accessories", "CLARIFY", "clarification"),
        # ===== NEW CLARIFY VARIATIONS =====
        ("want kitchen furniture", "CLARIFY", "clarification"),
        ("need dining room items", "CLARIFY", "clarification"),
        ("looking for office furniture", "CLARIFY", "clarification"),
        ("need study room accessories", "CLARIFY", "clarification"),
        ("want balcony furniture", "CLARIFY", "clarification"),
        ("need garden items", "CLARIFY", "clarification"),
        ("looking for kids room", "CLARIFY", "clarification"),
        ("need guest room furniture", "CLARIFY", "clarification"),
        ("want patio furniture", "CLARIFY", "clarification"),
        ("need storage solutions", "CLARIFY", "clarification"),
        # ===== PRODUCT_DETAIL (Should Work) =====
        ("tell me about this sofa", "PRODUCT_DETAIL", "product_detail"),
        ("what are the features of this rug", "PRODUCT_DETAIL", "product_detail"),
        ("show me details of this chair", "PRODUCT_DETAIL", "product_detail"),
        # ===== NEW PRODUCT_DETAIL VARIATIONS =====
        ("more info about this table", "PRODUCT_DETAIL", "product_detail"),
        ("what's special about this lamp", "PRODUCT_DETAIL", "product_detail"),
        ("details of this mirror", "PRODUCT_DETAIL", "product_detail"),
        ("tell me about this bed", "PRODUCT_DETAIL", "product_detail"),
        ("what are the specs of this furniture", "PRODUCT_DETAIL", "product_detail"),
        ("show features of this product", "PRODUCT_DETAIL", "product_detail"),
        ("what's included with this", "PRODUCT_DETAIL", "product_detail"),
        ("more details please", "PRODUCT_DETAIL", "product_detail"),
        ("what's the material of this", "PRODUCT_DETAIL", "product_detail"),
        ("show product information", "PRODUCT_DETAIL", "product_detail"),
        # ===== WARRANTY_QUERY (Should Work) =====
        ("what's the warranty on this", "WARRANTY_QUERY", "warranty_info"),
        ("does this come with warranty", "WARRANTY_QUERY", "warranty_info"),
        ("warranty period for furniture", "WARRANTY_QUERY", "warranty_info"),
        # ===== NEW WARRANTY_QUERY VARIATIONS =====
        ("how long is the warranty", "WARRANTY_QUERY", "warranty_info"),
        ("what warranty do you offer", "WARRANTY_QUERY", "warranty_info"),
        ("is there a guarantee", "WARRANTY_QUERY", "warranty_info"),
        ("warranty terms", "WARRANTY_QUERY", "warranty_info"),
        ("return policy", "WARRANTY_QUERY", "warranty_info"),
        ("what if it breaks", "WARRANTY_QUERY", "warranty_info"),
        ("repair policy", "WARRANTY_QUERY", "warranty_info"),
        ("warranty coverage", "WARRANTY_QUERY", "warranty_info"),
        ("guarantee period", "WARRANTY_QUERY", "warranty_info"),
        ("what's covered", "WARRANTY_QUERY", "warranty_info"),
        # ===== EDGE CASES & VARIATIONS =====
        ("", "INVALID", "clarification"),
        ("random gibberish text", "INVALID", "clarification"),
        ("xyz123", "INVALID", "clarification"),
        ("123456", "INVALID", "clarification"),
        ("???", "INVALID", "clarification"),
        ("...", "INVALID", "clarification"),
        ("a", "INVALID", "clarification"),
        ("the", "INVALID", "clarification"),
        ("and", "INVALID", "clarification"),
        ("or", "INVALID", "clarification"),
        # ===== MIXED/COMPLEX QUERIES =====
        ("do you have rugs and furniture", "PRODUCT_SEARCH", "product_suggestion"),
        (
            "show me rugs under 5000 and furniture under 10000",
            "BUDGET_QUERY",
            "budget_constraint",
        ),
        ("hello, I need help finding rugs", "GREETING", "greeting"),
        ("hi, what categories do you have", "GREETING", "greeting"),
        ("help me find furniture under 5000", "HELP", "help"),
        ("good morning, show me categories", "GREETING", "greeting"),
    ]

    results = []
    passed = 0
    failed = 0
    errors = 0

    print(f"Testing {len(test_cases)} scenarios...")
    print()

    for i, (query, expected_intent, expected_response_type) in enumerate(test_cases, 1):
        print(f"{i:2d}. Testing: '{query}'")
        print(f"    Expected Intent: {expected_intent}")
        print(f"    Expected Response: {expected_response_type}")

        try:
            payload = {"message": query, "session_id": f"test_{i}"}

            response = requests.post(base_url, json=payload, timeout=30)

            if response.status_code == 200:
                try:
                    result = response.json()
                    response_type = result.get("type", "unknown")
                except json.JSONDecodeError:
                    # Handle case where response is a plain string
                    response_text = response.text
                    print(f"    ⚠️  Got string response: '{response_text[:100]}...'")
                    response_type = "error_response"

                # Check if response type matches expected
                if response_type == expected_response_type:
                    print(f"    ✅ PASS - Got: {response_type}")
                    results.append(
                        {
                            "query": query,
                            "expected_intent": expected_intent,
                            "expected_response": expected_response_type,
                            "got_response": response_type,
                            "status": "PASS",
                        }
                    )
                    passed += 1
                else:
                    print(
                        f"    ❌ FAIL - Expected: {expected_response_type}, Got: {response_type}"
                    )
                    results.append(
                        {
                            "query": query,
                            "expected_intent": expected_intent,
                            "expected_response": expected_response_type,
                            "got_response": response_type,
                            "status": "FAIL",
                        }
                    )
                    failed += 1

            else:
                print(f"    ❌ HTTP Error: {response.status_code}")
                results.append(
                    {
                        "query": query,
                        "expected_intent": expected_intent,
                        "expected_response": expected_response_type,
                        "status": "ERROR",
                        "error": f"HTTP {response.status_code}",
                    }
                )
                errors += 1

        except Exception as e:
            print(f"    ❌ Exception: {e}")
            results.append(
                {
                    "query": query,
                    "expected_intent": expected_intent,
                    "expected_response": expected_response_type,
                    "status": "ERROR",
                    "error": str(e),
                }
            )
            errors += 1

        print()
        time.sleep(0.5)  # Small delay between requests

    # Summary
    print("=" * 60)
    print("📊 TESTING SUMMARY")
    print("=" * 60)

    total = len(test_cases)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {failed}/{total}")
    print(f"⚠️  Errors: {errors}/{total}")
    print(f"📈 Success Rate: {passed/total*100:.1f}%")

    # Show failures
    if failed > 0:
        print(f"\n🔍 FAILED TESTS:")
        for result in results:
            if result["status"] == "FAIL":
                print(
                    f"   • '{result['query']}' -> Expected: {result['expected_response']}, Got: {result['got_response']}"
                )

    # Show errors
    if errors > 0:
        print(f"\n⚠️  ERRORS:")
        for result in results:
            if result["status"] == "ERROR":
                print(
                    f"   • '{result['query']}' -> {result.get('error', 'Unknown error')}"
                )

    # Intent-specific analysis
    print(f"\n🎯 INTENT ANALYSIS:")
    intent_results = {}
    for result in results:
        intent = result["expected_intent"]
        if intent not in intent_results:
            intent_results[intent] = {"total": 0, "passed": 0, "failed": 0, "errors": 0}

        intent_results[intent]["total"] += 1
        if result["status"] == "PASS":
            intent_results[intent]["passed"] += 1
        elif result["status"] == "FAIL":
            intent_results[intent]["failed"] += 1
        else:
            intent_results[intent]["errors"] += 1

    for intent, stats in intent_results.items():
        success_rate = stats["passed"] / stats["total"] * 100
        status = "✅" if success_rate >= 90 else "⚠️" if success_rate >= 70 else "❌"
        print(
            f"   {status} {intent}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)"
        )

    return results


if __name__ == "__main__":
    test_trained_model()
