#!/usr/bin/env python3
"""Ticket triage via Jeviathan, grounded by the Brave Search API.

Flow
----
1. Triage   POST {JEVIATHAN_BASE_URL}/v1/systemone
            -> department (choice), urgency (noul), frustration (score)
2. Ground   when routing confidence < threshold (or --always-ground):
            GET https://api.search.brave.com/res/v1/web/search?q=...&count=N
            with X-Subscription-Token: $BRAVE_API_KEY
3. Report   decision + cited web evidence, ready to hand to an agent or a human.

Deterministic on purpose: no second model call in this example - it runs
anywhere and costs only search calls (Brave includes $5 free credits/month).

Setup
-----
    export JEVIATHAN_BASE_URL=http://localhost:8100   # default; see README quickstarts
    export BRAVE_API_KEY=<key from brave.com/search/api>

Usage
-----
    python examples/brave_search_triage.py "My card was charged twice for order #4512. I want a refund."
    python examples/brave_search_triage.py --always-ground "<ticket>"
    python examples/brave_search_triage.py --no-search "<ticket>"   # triage only, no key needed

Upgrade path: swap BRAVE_WEB_ENDPOINT for the LLM Context endpoint
(api-dashboard.search.brave.com/documentation/services/llm-context) when you
want results pre-packaged for model consumption instead of human snippets.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

BRAVE_WEB_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def http_json(url, payload=None, headers=None, timeout=30):
    """Minimal JSON-over-HTTP helper (stdlib only)."""
    data = None
    hdrs = {"Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def triage(ticket, base_url, timeout=1800.0):
    """One Jeviathan pass: typed decisions over the ticket state.

    Generous default timeout on purpose: the dev-tier profile sets
    timeout_s=1800 because laptop-GPU generation runs ~1-2 tok/s and the
    worst case is a full max_tokens eot_id loop; 30 s would time out most
    tickets. Use --timeout to tune per machine.
    """
    payload = {
        "state": 'Customer ticket: "' + ticket + '"',
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this ticket?",
                "criteria": {
                    "billing": "Charges, invoices, payment problems, refunds",
                    "returns": "Exchanges, wrong or damaged items",
                    "technical": "Product defects, account access, errors",
                },
            },
            "urgency": {
                "type": "noul",
                "instructions": "Is this ticket urgent (time-sensitive harm if delayed)?",
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "levels": [{"name": "calm"}, {"name": "frustrated"}, {"name": "very_frustrated"}],
            },
        },
    }
    return http_json(base_url.rstrip("/") + "/v1/systemone", payload, timeout=timeout)


def brave_web_search(query, api_key, count=5):
    """Grounding step: top web results for the ticket text."""
    params = urllib.parse.urlencode(
        {"q": query[:200], "count": str(count), "safesearch": "moderate"}
    )
    url = BRAVE_WEB_ENDPOINT + "?" + params
    data = http_json(url, headers={"X-Subscription-Token": api_key})
    results = data.get("web", {}).get("results") or []
    out = []
    for r in results[:count]:
        out.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("description", ""),
            }
        )
    return out


def fmt_conf(v):
    return "n/a" if v is None else format(v, ".3f")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ticket", help="customer ticket text")
    ap.add_argument("--base-url", default=os.environ.get("JEVIATHAN_BASE_URL", "http://localhost:8100"))
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="ground when department confidence is below this (default 0.7)")
    ap.add_argument("--always-ground", action="store_true",
                    help="call Brave Search API regardless of confidence")
    ap.add_argument("--no-search", action="store_true",
                    help="triage only; no BRAVE_API_KEY needed")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="HTTP timeout for the triage call in seconds; dev-tier "
                         "profiles allow up to 30 min (default 1800)")
    args = ap.parse_args()

    # 1) triage (local, ~$0)
    try:
        resp = triage(args.ticket, args.base_url, args.timeout)
    except urllib.error.URLError as e:
        sys.exit("Jeviathan unreachable at " + args.base_url + " (" + str(e.reason) + "). "
                 "Is it running? See README quickstarts.")

    answers = resp.get("answers", {})
    dept = answers.get("department") or {}
    urg = answers.get("urgency") or {}
    frus = answers.get("frustration") or {}

    print("=" * 72)
    print("TICKET")
    print(args.ticket)
    print("-" * 72)
    print("TRIAGE (Jeviathan, model=" + str(resp.get("model", "?")) + ")")
    print("  department : " + str(dept.get("choice", "?"))
          + " (confidence " + fmt_conf(dept.get("confidence")) + ")")
    print("  urgency    : noul " + fmt_conf(urg.get("noul")))
    print("  frustration: score " + fmt_conf(frus.get("score"))
          + " (confidence " + fmt_conf(frus.get("confidence")) + ")")

    # 2) grounding decision
    conf = dept.get("confidence")
    should_ground = args.always_ground or (conf is not None and conf < args.threshold)
    if args.no_search:
        should_ground = False

    print("-" * 72)
    if not should_ground:
        reason = "triage only (--no-search)" if args.no_search else \
            "confidence " + fmt_conf(conf) + " >= threshold " + format(args.threshold, ".2f")
        print("GROUNDING: skipped (" + reason + ")")
        return 0

    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        sys.exit("BRAVE_API_KEY is not set. Get a key at brave.com/search/api "
                 "($5 free credits/month). Or run with --no-search.")

    print("GROUNDING (Brave Search API)")
    try:
        results = brave_web_search(args.ticket, api_key, args.count)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            sys.exit("Brave rejected the key (HTTP 401). Check BRAVE_API_KEY.")
        raise

    if not results:
        print("  no results returned")
        return 0

    for i, r in enumerate(results, 1):
        print("  [" + str(i) + "] " + r["title"])
        print("      " + r["url"])
        if r["description"]:
            print("      " + r["description"][:200])

    # 3) report
    print("-" * 72)
    print("SUGGESTED NEXT STEP")
    print("  Route to " + str(dept.get("choice", "?")).upper()
          + "; attach citations [1..n] above as grounding context.")
    if urg.get("noul") is not None and urg["noul"] >= 0.8:
        print("  Urgency noul=" + fmt_conf(urg["noul"])
              + " -> handle same-day; do not auto-reply without review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
