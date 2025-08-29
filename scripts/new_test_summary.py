#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path

TEMPLATES = {
    "test": "Documentation/Reports/templates/Test-Engineer-Template.md",
    "core": "Documentation/Reports/templates/Core-Engineer-Template.md",
}


def main():
    ap = argparse.ArgumentParser(description="Create a new report from template")
    ap.add_argument("--role", choices=["test", "core"], required=True)
    ap.add_argument("--subject", required=True, help="Report subject")
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()

    today = datetime.utcnow().strftime("%Y-%m-%d")
    role_name = "Test Engineer" if args.role == "test" else "Core Engineer"
    name = f"{today} {role_name} Report - {args.subject} {args.version}.md"
    out_dir = Path("Documentation/Reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / name

    tpl = Path(TEMPLATES[args.role])
    content = tpl.read_text()
    content = content.replace("<yyyy-mm-dd>", today)
    content = content.replace("<Subject>", args.subject)
    content = content.replace("v1", args.version)

    dst.write_text(content)
    print(dst)


if __name__ == "__main__":
    main()
