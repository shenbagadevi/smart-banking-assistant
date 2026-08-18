def identify_product_category(text):

    text = text.lower()

    mapping = {
        "home loan": "HOME_LOAN",
        "car loan": "CAR_LOAN",
        "personal loan": "PERSONAL_LOAN",
        "credit card": "CREDIT_CARD",
        "savings account": "SAVINGS_ACCOUNT",
    }

    for key, value in mapping.items():

        if key in text:
            return value

    return "GENERAL"
