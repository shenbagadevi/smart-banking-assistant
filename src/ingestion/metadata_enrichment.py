def identify_product_category(text):
    """
    Lightweight canonical mapping for product category.

    Returns one of the allowed canonical product categories:
    - Loans
    - Deposits
    - Cards
    - Customer Support
    Or None if no confident mapping.
    """
    if not text:
        return None
    t = text.lower()

    if any(k in t for k in ("home loan", "personal loan", "loan")):
        return "loan"
    if any(k in t for k in ("fixed deposit", "term deposit", "deposit")):
        return "deposit"
    if any(k in t for k in ("credit card", "card")):
        return "card"
    if any(
        k in t
        for k in (
            "contact",
            "complain",
            "complaint",
            "customer service",
            "escalation",
            "support",
        )
    ):
        return "customer_support"

    return None


def canonical_product_name(text):
    """Map text snippets to canonical product_name values.

    Allowed values:
    - Home Loan
    - Personal Loan
    - Fixed Deposit
    - Credit Card
    - Regulatory and Disclosure Compliance
    - Customer Support
    """
    if not text:
        return None
    t = text.lower()
    if "home loan" in t:
        return "Home Loan"
    if "personal loan" in t:
        return "Personal Loan"
    if any(k in t for k in ("fixed deposit", "term deposit", "fd")):
        return "Fixed Deposit"
    if "credit card" in t or "card" in t:
        return "Credit Card"
    if any(k in t for k in ("regulatory", "disclosure", "compliance")):
        return "Regulatory and Disclosure Compliance"
    if any(
        k in t
        for k in ("contact", "complaint", "customer service", "support", "escalation")
    ):
        return "Customer Support"
    return None


def canonical_loan_type(text):
    """Return canonical loan_type display names or None.

    Allowed values:
    - Home Loan
    - Personal Loan
    """
    if not text:
        return None
    t = text.lower()
    if "home loan" in t:
        return "Home Loan"
    if "personal loan" in t:
        return "Personal Loan"
    return None
