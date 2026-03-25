import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HF dataset project CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    create_project = sub.add_parser("create-project", help="Create project scaffold in HF dataset")
    create_project.add_argument("--project-slug", required=True)
    create_project.add_argument("--dataset-repo", required=True)

    sub.add_parser("verify-project", help="Verify project integrity")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "create-project":
        print(f"[TODO] create-project: {args.project_slug} in {args.dataset_repo}")
        return

    if args.command == "verify-project":
        print("[TODO] verify-project")
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
