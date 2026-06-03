"""Run a minimal two-account permission proof against an organization HF Bucket.

The script uses only the active Hugging Face credential available to
``huggingface_hub``. It never prints or stores tokens and writes only tiny
diagnostic JSON files in the chosen test Bucket.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import (
    HfApi,
    batch_bucket_files,
    create_bucket,
    delete_bucket,
    download_bucket_files,
    list_bucket_tree,
)


DEFAULT_BUCKET_ID = "ppbio-rabeca/birdnet-validator-permission-spike"
DIAGNOSTIC_PREFIX = "diagnostics/organization-permission-spike"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _authenticated_username() -> str:
    profile = HfApi().whoami(token=True)
    username = str(profile.get("name") or profile.get("fullname") or "").strip()
    if not username:
        raise RuntimeError("The active Hugging Face credential does not contain a username.")
    return username


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return cleaned or "unknown-user"


def _payload(*, actor: str, action: str, role_label: str) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "diagnostic": "birdnet_validator_org_bucket_permission_spike",
                "actor": actor,
                "action": action,
                "role_label": role_label,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "contains_project_data": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_marker(bucket_id: str, *, action: str, role_label: str) -> str:
    username = _authenticated_username()
    path = f"{DIAGNOSTIC_PREFIX}/{_safe_name(username)}-{_safe_name(role_label)}-{_timestamp()}.json"
    batch_bucket_files(
        bucket_id,
        add=[(_payload(actor=username, action=action, role_label=role_label), path)],
        token=True,
    )
    print(f"PASS: {username} wrote {path} to private Bucket {bucket_id}.")
    return path


def command_identity(_: argparse.Namespace) -> int:
    username = _authenticated_username()
    print(f"Authenticated Hugging Face account: {username}")
    return 0


def command_create(args: argparse.Namespace) -> int:
    username = _authenticated_username()
    bucket_url = create_bucket(args.bucket_id, private=True, exist_ok=True, token=True)
    print(f"Private test Bucket ready: {bucket_url.bucket_id} (created/confirmed by {username}).")
    _write_marker(args.bucket_id, action="bucket_initialized", role_label="admin")
    return 0


def command_write(args: argparse.Namespace) -> int:
    _write_marker(args.bucket_id, action="permission_write", role_label=args.role_label)
    return 0


def command_read(args: argparse.Namespace) -> int:
    username = _authenticated_username()
    entries = [
        item
        for item in list_bucket_tree(
            args.bucket_id,
            prefix=DIAGNOSTIC_PREFIX,
            recursive=True,
            token=True,
        )
        if getattr(item, "type", "file") == "file"
    ]
    if not entries:
        raise RuntimeError(f"No diagnostic marker was visible in {args.bucket_id}.")

    latest = sorted(entries, key=lambda item: str(item.path))[-1]
    with TemporaryDirectory(prefix="birdnet-hf-bucket-proof-") as temp_dir:
        destination = Path(temp_dir) / "marker.json"
        download_bucket_files(
            args.bucket_id,
            files=[(str(latest.path), destination)],
            raise_on_missing_files=True,
            token=True,
        )
        payload = json.loads(destination.read_text(encoding="utf-8"))
    print(
        "PASS: "
        f"{username} read {latest.path}; marker actor={payload.get('actor', 'unknown')}, "
        f"role_label={payload.get('role_label', 'unknown')}."
    )
    return 0


def command_list(args: argparse.Namespace) -> int:
    username = _authenticated_username()
    entries = list(
        list_bucket_tree(
            args.bucket_id,
            prefix=DIAGNOSTIC_PREFIX,
            recursive=True,
            token=True,
        )
    )
    print(f"Visible diagnostic entries for {username}: {len(entries)}")
    for item in entries:
        print(f"- {item.path}")
    return 0


def command_delete(args: argparse.Namespace) -> int:
    _authenticated_username()
    if args.confirm_delete != args.bucket_id:
        raise RuntimeError(
            "Deletion refused. Pass --confirm-delete with the exact Bucket id to remove only the test Bucket."
        )
    delete_bucket(args.bucket_id, missing_ok=True, token=True)
    print(f"Deleted test Bucket: {args.bucket_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe private Hugging Face organization Bucket access with the currently logged-in account."
    )
    parser.add_argument(
        "--bucket-id",
        default=DEFAULT_BUCKET_ID,
        help=f"Private diagnostic Bucket id (default: {DEFAULT_BUCKET_ID}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("identity", help="Show the account supplied by local HF authentication.").set_defaults(
        func=command_identity
    )
    subparsers.add_parser("create", help="Create/confirm the private Bucket and write the admin marker.").set_defaults(
        func=command_create
    )
    write_parser = subparsers.add_parser("write", help="Write one small permission diagnostic marker.")
    write_parser.add_argument(
        "--role-label",
        required=True,
        choices=["admin", "contributor", "write", "oauth-admin", "oauth-validator"],
        help="Role under test, recorded in the marker only.",
    )
    write_parser.set_defaults(func=command_write)
    subparsers.add_parser("read", help="Read the latest visible diagnostic marker.").set_defaults(func=command_read)
    subparsers.add_parser("list", help="List visible diagnostic markers.").set_defaults(func=command_list)
    delete_parser = subparsers.add_parser("delete", help="Delete the diagnostic Bucket after testing.")
    delete_parser.add_argument("--confirm-delete", required=True)
    delete_parser.set_defaults(func=command_delete)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
