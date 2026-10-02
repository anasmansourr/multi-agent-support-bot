# Labelling policy for the 4 intents

Written after Phase 6 error analysis exposed contradictory synthetic labels. Applies to all
synthetic data and to the hard test set.

| Intent | A message belongs here when it is... | Typical examples |
|---|---|---|
| `billing` | about **the customer's own** money or account: charges, invoices, receipts, refunds, payment methods, renewals, billing-related cancellation, taxes on their invoice | "i was charged twice", "refund request for order 5521", "will i be charged if i cancel today" |
| `technical` | something **broken or not working** in the product, or a how-to-fix-it problem | "the export button does nothing", "api returns 401" |
| `feature_request` | asking for something the product **does not do yet** (new capability, integration, option) | "please add bulk editing", "any plans for a linux app" |
| `general` | **pre-sales / public information**, company questions, greetings, thanks, questions about the bot | "what does the enterprise plan cost", "do you offer a non profit discount", "hello" |

Tie-breakers
1. Public price lists, plan comparisons and discount programmes are `general` (they are not about the user's account); "how much do **I** owe / pay" is `billing`.
2. If a message reports something broken, it is `technical` even when money is mentioned ("payment went through but my order does not show up").
3. "Can you add X?" / "do you support X?" where X is a missing capability is `feature_request`; where X is a company or product fact it is `general`.
4. Genuinely ambiguous messages are kept in the hard test set, marked in its `note` column.
