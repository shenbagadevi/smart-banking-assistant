from src.ingestion.parser import _infer_product_metadata


def test_infer_product_metadata_only_matches_strong_local_evidence():
    home_text = "Home Loan eligibility requires a minimum income of 30000 and a 20% down payment."
    fixed_deposit_text = (
        "Fixed Deposit offers competitive interest rates and a 5-year lock-in period."
    )
    generic_text = (
        "This product includes flexible repayment options and digital onboarding."
    )

    home_meta = _infer_product_metadata(home_text, element_id="home-loan-1")
    fixed_deposit_meta = _infer_product_metadata(fixed_deposit_text, element_id="fd-1")
    generic_meta = _infer_product_metadata(generic_text, element_id="generic-1")

    assert home_meta["product_category"] == "loan"
    assert home_meta["loan_type"] == "home_loan"
    assert home_meta["product_name"] == "Home Loan"

    assert fixed_deposit_meta["product_category"] is None
    assert fixed_deposit_meta["loan_type"] is None
    assert fixed_deposit_meta["product_name"] is None

    assert generic_meta["product_category"] is None
    assert generic_meta["loan_type"] is None
    assert generic_meta["product_name"] is None


def test_infer_product_metadata_ignores_document_wide_keyword_contamination():
    mixed_document = "Home Loans Fixed Deposits Credit Cards Personal Loans"

    inferred = _infer_product_metadata(mixed_document, element_id="mixed-section")

    assert inferred["product_category"] is None
    assert inferred["loan_type"] is None
    assert inferred["product_name"] is None
