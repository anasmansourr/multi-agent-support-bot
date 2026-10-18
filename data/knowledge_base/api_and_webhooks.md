# API and webhooks

The REST API base URL is https://api.flowdesk.example/v1. Create an API key under Settings > API. The key is shown only once. Send it as "Authorization: Bearer <key>".

Errors: 401 means the key is invalid or revoked. 403 means your role lacks permission. 429 means you exceeded 100 requests per minute per key; wait for the number of seconds in the Retry-After header. List endpoints return a next_page_token for pagination.

Webhooks: the endpoint must be a public HTTPS URL that answers with a 2xx status within 10 seconds. Failed deliveries are retried 3 times, after 1, 5 and 30 minutes. A webhook is disabled after 20 consecutive failures.

Each delivery has an X-FlowDesk-Signature header, an HMAC-SHA256 of the body using your signing secret. Check it to confirm the request is genuine.

See delivery attempts under Settings > Webhooks > Logs (kept for 30 days). Use "Send test event" to test your endpoint.
