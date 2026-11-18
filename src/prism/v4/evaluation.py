"""Local-only Judge transport; V4 packets require a new v4-*.sqlite ledger.

No paid request is made without --execute and a new explicit --budget-usd.
"""
from prism.v3.evaluation import main

if __name__ == "__main__":
    main()
