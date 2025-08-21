#!/usr/bin/env python3
import json
from nsa.core.flags import execution_routing_summary


def main():
    info = execution_routing_summary()
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

