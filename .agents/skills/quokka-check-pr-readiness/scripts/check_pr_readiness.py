"""Print whether a Quokka pull request is ready for review, as a table of checks.

Usage: python3 check_pr_readiness.py <pr-number> [--repo quokka-astro/quokka]
Exits 0 when every check passes, and 1 otherwise.
"""

import argparse
import json
import subprocess
import sys

QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      state
      isDraft
      body
      commits(last: 1) {
        nodes {
          commit {
            oid
            statusCheckRollup {
              contexts(first: 100) {
                nodes {
                  __typename
                  ... on CheckRun { name status conclusion }
                  ... on StatusContext { context state }
                }
              }
            }
          }
        }
      }
      reviewThreads(first: 100) { nodes { isResolved } }
      comments(last: 100) { nodes { body } }
    }
  }
}
"""

TEMPLATE_PLACEHOLDER = "Please write a short paragraph describing what changes this pull request makes"
FAILED_CONCLUSIONS = ("FAILURE", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED", "STARTUP_FAILURE")
FAILED_STATES = ("FAILURE", "ERROR")


def read_pull_request(
    *,
    repo: str,
    pr_number: int,
) -> dict:
    owner, name = repo.split("/")
    completed_process = subprocess.run(
        [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={QUERY}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"number={pr_number}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed_process.stdout)["data"]["repository"]["pullRequest"]


def check_state(
    pr_data: dict,
) -> tuple[bool, str]:
    if pr_data["state"] != "OPEN":
        return False, f"the PR is {pr_data['state'].lower()}"
    elif pr_data["isDraft"]:
        return False, "the PR is a draft; mark it ready for review"
    else:
        return True, "open and out of draft"


def check_ci(
    pr_data: dict,
) -> tuple[bool, str]:
    commit_data = pr_data["commits"]["nodes"][0]["commit"]
    rollup_data = commit_data["statusCheckRollup"]
    if rollup_data is None:
        return False, f"no checks have run on `{commit_data['oid'][:9]}`"
    failed_names = []
    pending_names = []
    for context_data in rollup_data["contexts"]["nodes"]:
        if context_data["__typename"] == "CheckRun":
            check_name = context_data["name"]
            check_is_done = (context_data["status"] == "COMPLETED")
            check_failed = (context_data["conclusion"] in FAILED_CONCLUSIONS)
        else:
            check_name = context_data["context"]
            check_is_done = (context_data["state"] != "PENDING")
            check_failed = (context_data["state"] in FAILED_STATES)
        if check_failed:
            failed_names.append(check_name)
        elif not check_is_done:
            pending_names.append(check_name)
    if len(failed_names) > 0:
        return False, f"failed: {', '.join(sorted(set(failed_names)))}"
    elif len(pending_names) > 0:
        return False, f"still running: {', '.join(sorted(set(pending_names)))}"
    else:
        return True, f"every check passed or was skipped on `{commit_data['oid'][:9]}`"


def check_description(
    pr_data: dict,
) -> tuple[bool, str]:
    body_text = pr_data["body"] or ""
    unticked_lines = [
        _line.strip() for _line in body_text.splitlines()
        if _line.strip().startswith("- [ ]") and ("~~" not in _line)
    ]
    if TEMPLATE_PLACEHOLDER in body_text:
        return False, "the description still has the template placeholder text"
    elif len(unticked_lines) > 0:
        return False, f"{len(unticked_lines)} checklist items are neither ticked nor struck through as not applicable"
    else:
        return True, "description written and checklist complete"


def check_gpu_tests(
    pr_data: dict,
) -> tuple[bool, str]:
    azp_commands = sorted({
        _comment_data["body"].strip().splitlines()[0]
        for _comment_data in pr_data["comments"]["nodes"]
        if _comment_data["body"].strip().startswith("/azp run")
    })
    gpu_checklist_lines = [_line for _line in (pr_data["body"] or "").splitlines() if "/azp run" in _line]
    gpu_tests_are_not_applicable = any("~~" in _line for _line in gpu_checklist_lines)
    if (len(azp_commands) == 0) and gpu_tests_are_not_applicable:
        return True, "not needed: the author marked GPU tests as not applicable"
    elif len(azp_commands) == 0:
        return False, "no `/azp run` comment; see the PR template for which command applies"
    else:
        return True, f"triggered with {', '.join(f'`{_command}`' for _command in azp_commands)}"


def check_review_threads(
    pr_data: dict,
) -> tuple[bool, str]:
    num_unresolved = sum(1 for _thread_data in pr_data["reviewThreads"]["nodes"] if not _thread_data["isResolved"])
    if num_unresolved > 0:
        return False, f"{num_unresolved} review threads are unresolved"
    else:
        return True, "no unresolved review threads"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pr_number", type=int)
    parser.add_argument("--repo", type=str, default="quokka-astro/quokka")
    parsed_args = parser.parse_args()
    pr_data = read_pull_request(
        repo=parsed_args.repo,
        pr_number=parsed_args.pr_number,
    )
    checks = (
        ("State", check_state(pr_data)),
        ("CI", check_ci(pr_data)),
        ("Description", check_description(pr_data)),
        ("GPU tests", check_gpu_tests(pr_data)),
        ("Review threads", check_review_threads(pr_data)),
    )
    print("| Check | Passed | Detail |")
    print("|---|---|---|")
    for check_name, (check_passed, detail_text) in checks:
        if check_passed:
            passed_text = "yes"
        else:
            passed_text = "no"
        print(f"| {check_name} | {passed_text} | {detail_text} |")
    pr_is_ready = all(_check_passed for _, (_check_passed, _) in checks)
    if pr_is_ready:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
