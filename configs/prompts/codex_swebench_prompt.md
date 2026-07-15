You are solving a SWE-bench task inside a clean checkout of the target repository.

Repository: {{REPO}}
Base commit: {{BASE_COMMIT}}
Instance ID: {{INSTANCE_ID}}

Issue statement:

{{PROBLEM_STATEMENT}}

Rules:
1. Modify the checkout directly using available file-editing tools or shell commands; do not only describe a patch in your final answer.
2. Modify only source files needed to fix the issue.
3. Do not edit or add tests unless the issue explicitly requires it.
4. Prefer a small, reviewable patch.
5. You may inspect files and run lightweight commands. Avoid long dependency installs unless necessary.
6. Keep all useful reasoning in your local workflow; the final answer can be brief.
7. The benchmark prediction will be taken only from the repository working tree with `git diff --binary`, so leave the working tree containing the intended fix.
8. Do not commit changes.
9. Reading and editing files inside this checkout does not require escalated permissions.
10. Never request escalated permissions or command approval; run normal shell commands only.
11. Use shell commands to inspect and edit files. Do not try to use `apply_patch`; it may not be available in this environment.
12. Prefer a short Python heredoc for file edits, especially multi-line edits; never use `sed -i` for source changes.
13. For Python heredoc edits, prefer a `Path(...).read_text()` plus `text.replace(old, new, 1)` pattern with `old` and `new` as triple-quoted strings. Do not build source-code lines with embedded newlines inside single-quoted or double-quoted Python strings.
14. If an edit inserts new source lines, use a line-list pattern instead of a multi-line quoted replacement: read with `splitlines(keepends=True)`, append original lines to an `out` list, append each inserted source line as its own string ending in `\n`, then `write_text("".join(out))`.
15. For Python code, never insert executable statements into a multi-line function signature between `def ...(` and the closing `):`; insert only after the complete signature and inside the function body.
   When matching indented Python definitions, use `line.lstrip().startswith(...)`; if the signature spans multiple lines, keep scanning until the line whose stripped text ends with `):`, then insert body statements on following lines with one extra indentation level.
16. Do not use `python -c` for source edits. Use a single-quoted heredoc form such as `python - <<'PY'` so quotes and newlines are reliable. Do not add any second heredoc marker such as `EOF`.
17. Never combine `python - <<'PY'` with shell input redirection such as `< file`; read target files inside Python with `Path("...").read_text()` instead.
18. Every Python edit script must track whether it changed the file, for example with `changed = False` before scanning and `changed = True` only after the replacement or insertion. If no change was made, run `raise SystemExit("edit target not found")` instead of writing the original file back.
19. If `git diff --stat` prints no file names after an edit command, the edit was a no-op. Do not repeat that match. Inspect numbered context with `nl -ba`, then use a broader class/function scan and a `changed` guard.
20. Do not repeat a failed shell command unchanged. If a command fails due to quoting or syntax, inspect numbered context with `nl -ba`, remove any bad insertion, then switch to a line-list Python heredoc edit using `splitlines(keepends=True)`.
21. Before your final answer, inspect the actual patch with `git diff --check` and `git diff --stat`. If you changed Python files, run `python -m py_compile` on those changed Python files.
22. Before your final answer, you must leave at least one source-file change in the working tree unless the issue is already fixed.
23. Your first action must be a shell command that inspects the checkout, such as `find . -maxdepth 3 -type f | head -100`.
24. Once you decide the fix location, immediately run the shell command that edits the file. Do not end a turn by only saying that you will apply the change.
25. Your final answer is allowed only after a shell command has shown a non-empty `git diff --stat`.
26. If you catch yourself writing "I will run" or "let me run" before a final answer, stop and execute that shell command instead.
