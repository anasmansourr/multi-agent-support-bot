"""Explicit mapping from public-dataset labels to OUR 4 intents.

Rules of this file:
  * Every public label we USE is listed here with its target intent.
  * Every public label NOT listed is deliberately dropped (never guessed).
  * Names are matched case-insensitively (Banking77 has e.g. 'Refund_not_showing_up').
  * `data_acquisition.py` validates these names against the real dataset and
    raises an error if any name does not exist, so typos can't pass silently.

Honest limitation: Banking77 is a BANKING dataset and CLINC150 is a generic
voice-assistant dataset. We only borrow labels whose meaning transfers to SaaS
support (charges, refunds, bills, greetings...). `technical` and `feature_request`
have almost no real public data, so they will be mostly SYNTHETIC (Phase 2b).
"""

# ---------------------------------------------------------------- Banking77
# Only money-related complaints that read naturally for a SaaS subscription.
BANKING77_TO_INTENT = {
    "card_payment_fee_charged": "billing",
    "card_payment_not_recognised": "billing",
    "declined_card_payment": "billing",
    "extra_charge_on_statement": "billing",
    "pending_card_payment": "billing",
    "refund_not_showing_up": "billing",
    "request_refund": "billing",
    "transaction_charged_twice": "billing",
    # Login/credential trouble transfers to a SaaS "can't access my account" issue.
    "passcode_forgotten": "technical",
}

# ------------------------------------------------------------------ CLINC150
CLINC150_TO_INTENT = {
    # bills / payments
    "bill_balance": "billing",
    "bill_due": "billing",
    "pay_bill": "billing",
    "interest_rate": "billing",
    # account / settings problems
    "account_blocked": "technical",
    "reset_settings": "technical",
    "sync_device": "technical",
    # chit-chat / meta questions to the assistant
    "greeting": "general",
    "goodbye": "general",
    "thank_you": "general",
    "what_can_i_ask_you": "general",
    "who_made_you": "general",
    "are_you_a_bot": "general",
    "what_is_your_name": "general",
    "how_old_are_you": "general",
    "who_do_you_work_for": "general",
}
