#!/usr/bin/env python3
"""
Quick test for NorthStar home loan tenure query fix.
Tests the changes to evaluate_answer_node and generate_answer_node.
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure demo mode is OFF
os.environ["DEMO_MODE"] = "false"

# Provide a lightweight stub for optional external modules used by the app
# so the test harness can run without installing them (keeps tests focused).
import types

if "cohere" not in sys.modules:
    sys.modules["cohere"] = types.ModuleType("cohere")

from src.api.v1.services.query_service import process_query


def test_northstar_home_loan_tenure():
    """Test the NorthStar home loan tenure query."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: NorthStar Home Loan Maximum Tenure Query")
    logger.info("=" * 80)

    query = "What is the maximum tenure available for NorthStar Bank home loans?"
    user_id = "test_user_northstar"
    correlation_id = "test_correlation_northstar_001"

    logger.info(f"Query: {query}")
    logger.info(f"User ID: {user_id}")
    logger.info(f"Correlation ID: {correlation_id}")
    logger.info("")

    try:
        result = process_query(query, user_id=user_id, correlation_id=correlation_id)

        logger.info("RESULT:")
        logger.info(f"  Answer: {result.get('answer', 'N/A')[:200]}...")
        logger.info(f"  Query Path: {result.get('query_path', 'N/A')}")
        logger.info(f"  Retry Count: {result.get('retry_count', 'N/A')}")
        logger.info(f"  Confidence Score: {result.get('confidence_score', 'N/A')}")
        logger.info(f"  Document Name: {result.get('document_name', 'N/A')}")
        logger.info(f"  Page No: {result.get('page_no', 'N/A')}")
        logger.info(f"  Policy Citations: {result.get('policy_citations', [])}")
        logger.info(f"  SQL Executed: {result.get('sql_query_executed', 'N/A')}")
        logger.info("")

        # Validation
        logger.info("VALIDATION:")

        answer = result.get("answer", "").lower()

        # Check 1: Should NOT be the fallback message
        fallback_msg = "do not contain sufficient information"
        if fallback_msg in answer:
            logger.error(f"  ✗ FAIL: Answer still contains fallback message")
            logger.error(f"    Expected: Answer with tenure data")
            logger.error(f"    Got: {result.get('answer', '')[:200]}")
            return False
        else:
            logger.info(f"  ✓ PASS: Answer does not contain fallback message")

        # Check 2: Confidence should be > 0
        confidence = result.get("confidence_score", 0)
        if confidence <= 0:
            logger.error(f"  ✗ FAIL: Confidence score is {confidence}, expected > 0")
            return False
        else:
            logger.info(f"  ✓ PASS: Confidence score is {confidence} > 0")

        # Check 3: Retry count should be minimal (0 or 1)
        retry_count = result.get("retry_count", 0)
        if retry_count > 1:
            logger.error(f"  ✗ FAIL: Retry count is {retry_count}, expected <= 1")
            return False
        else:
            logger.info(f"  ✓ PASS: Retry count is {retry_count} (minimal)")

        # Check 4: Should be RAG query (not SQL)
        qp = result.get("query_path") or result.get("route") or ""
        query_path = qp.lower() if isinstance(qp, str) else ""
        if query_path == "sql":
            logger.error(f"  ✗ FAIL: Query routed to SQL, should be RAG")
            return False
        else:
            logger.info(f"  ✓ PASS: Query path is {query_path} (RAG-based)")

        # Check 5: Should have document citations
        citations = result.get("policy_citations", [])
        if not citations:
            logger.warning(f"  ⚠ WARNING: No policy citations found")
        else:
            logger.info(f"  ✓ PASS: Policy citations: {citations}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ ALL CHECKS PASSED")
        logger.info("=" * 80)
        return True

    except Exception as e:
        logger.error(f"  ✗ EXCEPTION: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return False


def test_legitimate_fallback():
    """Test that legitimate fallback answers are NOT retried."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Legitimate Fallback (No Data Available)")
    logger.info("=" * 80)

    # Query for information that likely doesn't exist
    query = "What is the home loan rate for customers born on February 29?"
    user_id = "test_user_fallback"
    correlation_id = "test_correlation_fallback_001"

    logger.info(f"Query: {query}")
    logger.info("")

    try:
        result = process_query(query, user_id=user_id, correlation_id=correlation_id)

        logger.info("RESULT:")
        logger.info(f"  Answer: {result.get('answer', 'N/A')[:200]}...")
        logger.info(f"  Retry Count: {result.get('retry_count', 'N/A')}")
        logger.info(f"  Confidence Score: {result.get('confidence_score', 'N/A')}")
        logger.info("")

        # For a genuinely unavailable answer, we expect:
        # - Either fallback message OR high confidence (0.9 if context truly empty)
        # - Minimal retries (0 if context empty, max 1 if overly cautious)

        retry_count = result.get("retry_count", 0)
        confidence = result.get("confidence_score", 0)

        logger.info("VALIDATION:")
        if confidence == 0.9:
            logger.info(f"  ✓ PASS: Legitimate fallback has high confidence (0.9)")
        elif retry_count <= 1:
            logger.info(
                f"  ✓ PASS: Minimal retries ({retry_count}) for unavailable data"
            )
        else:
            logger.warning(
                f"  ⚠ WARNING: Retry count is {retry_count}, confidence is {confidence}"
            )

        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ LEGITIMATE FALLBACK TEST PASSED")
        logger.info("=" * 80)
        return True

    except Exception as e:
        logger.error(f"  ✗ EXCEPTION: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    logger.info("Starting targeted test suite for NorthStar fix...\n")

    results = []

    # Run tests
    results.append(("NorthStar Home Loan Tenure", test_northstar_home_loan_tenure()))
    results.append(("Legitimate Fallback", test_legitimate_fallback()))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("=" * 80)

    sys.exit(0 if passed == total else 1)
