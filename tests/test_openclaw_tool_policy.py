import json
import subprocess
import sys

from harness_eval.benchmarks.openclaw import OpenClawBenchmark
from harness_eval.harnesses.native_claw import miniharness_initial_messages


def test_prepare_task_exports_only_official_allowed_tool_policy(tmp_path):
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "T900_policy_gate"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        """
task_id: T900_policy_gate
task_name: Policy Gate Fixture
category: finance
prompt:
  text: Find the answer using the official tools.
  language: en
services:
  - name: web_real
    command: python mock_services/web_real/server.py
    port: 9114
    health_check: http://localhost:9114/web/health
    health_check_method: GET
    reset_endpoint: http://localhost:9114/web/reset
tools:
  - name: web_search
    description: Search the public web.
    input_schema:
      type: object
      properties:
        query:
          type: string
      required: [query]
  - name: crm_update
    description: Update a CRM record.
    input_schema:
      type: object
      properties:
        record_id:
          type: string
      required: [record_id]
tool_endpoints:
  - tool_name: web_search
    url: http://localhost:9114/web/search
    method: POST
  - tool_name: crm_update
    url: http://localhost:9114/crm/update
    method: POST
  - tool_name: hidden_admin
    url: http://localhost:9114/admin
    method: POST

environment:
  timeout_seconds: 1800
  max_turns: 50
""".strip()
        + "\n",
        encoding="utf-8",
    )
    prompt_template = tmp_path / "prompt.md"
    prompt_template.write_text(
        "Task {{TASK_ID}}\n{{QUERY}}\n{{CLAW_EVAL_TOOLS}}\n",
        encoding="utf-8",
    )
    cfg = {
        "project": {"root": str(tmp_path)},
        "benchmark": {
            "tasks_dir": str(tasks_dir),
            "fixture_root": str(tmp_path / "fixtures"),
            "prompt_template": str(prompt_template),
            "live_tool_bridge": False,
        },
    }

    benchmark = OpenClawBenchmark(cfg, tmp_path / "run")
    task = benchmark.prepare_task(
        {
            "task_id": "T900_policy_gate",
            "query": "Which tools are available?",
            "category": "finance",
            "language": "en",
        }
    )

    policy = json.loads((task.output_dir / "claw_eval_tool_policy.json").read_text(encoding="utf-8"))
    workspace_policy = json.loads((task.workspace / "claw_eval_tool_policy.json").read_text(encoding="utf-8"))

    # Preserve YAML tool order; this is also the model-visible OpenAI tools order.
    assert policy["allowed_tools"] == ["web_search", "crm_update"]
    assert [spec["name"] for spec in policy["allowed_tool_specs"]] == ["web_search", "crm_update"]
    assert [ep["tool_name"] for ep in policy["exposed_tool_endpoints"]] == ["web_search", "crm_update"]
    assert policy["environment"] == {"max_turns": 50, "timeout_seconds": 1800}
    assert "hidden_admin" not in json.dumps(policy, ensure_ascii=False)
    assert workspace_policy["policy_sha256"] == policy["policy_sha256"]
    assert workspace_policy["allowed_tools"] == policy["allowed_tools"]
    assert (task.workspace / "claw_tool").exists()
    assert (task.workspace / "claw_web_search").exists()
    assert set(policy["helper_files"]) == {"claw_tool", "claw_web_search"}
    assert task.metadata["tool_policy_path"] == str(task.output_dir / "claw_eval_tool_policy.json")
    assert task.metadata["workspace_tool_policy_path"] == str(task.workspace / "claw_eval_tool_policy.json")
    assert task.metadata["timeout_seconds"] == 1800
    assert task.metadata["max_turns"] == 50
    assert "Official Claw-Eval tool policy" in task.prompt
    assert "timeout_seconds=1800" in task.prompt
    assert "max_turns=50" in task.prompt
    assert "crm_update" in task.prompt
    assert "hidden_admin" not in task.prompt
    assert "@payload.json" in task.prompt
    helper_probe = subprocess.run(
        [sys.executable, str(task.workspace / "claw_tool"), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert helper_probe.returncode == 2
    assert "crm_update" in helper_probe.stderr
    assert "web_search" in helper_probe.stderr


def test_live_prepare_task_uses_miniharness_user_prompt_without_bridge_contract(tmp_path):
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "T902_live_prompt"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        """
task_id: T902_live_prompt
task_name: Live Prompt Fixture
prompt:
  text: Use the available tools to answer the original task.
  language: en
tools:
  - name: web_fetch
    description: Fetch full webpage content for a URL.
    input_schema:
      type: object
      properties:
        url:
          type: string
      required: [url]
tool_endpoints:
  - tool_name: web_fetch
    url: http://localhost:9114/web/fetch
    method: POST
environment:
  timeout_seconds: 1800
  max_turns: 50
""".strip()
        + "\n",
        encoding="utf-8",
    )
    prompt_template = tmp_path / "prompt.md"
    prompt_template.write_text(
        "Task {{TASK_ID}}\n{{QUERY}}\n{{CLAW_EVAL_TOOLS}}\n",
        encoding="utf-8",
    )
    cfg = {
        "project": {"root": str(tmp_path)},
        "benchmark": {
            "tasks_dir": str(tasks_dir),
            "fixture_root": str(tmp_path / "fixtures"),
            "prompt_template": str(prompt_template),
            "live_tool_bridge": True,
        },
    }

    benchmark = OpenClawBenchmark(cfg, tmp_path / "run")
    task = benchmark.prepare_task(
        {
            "task_id": "T902_live_prompt",
            "query": "stale dataset query should not override task.yaml",
            "language": "en",
        }
    )

    assert task.prompt == "Use the available tools to answer the original task."
    assert "Task T902_live_prompt" not in task.prompt
    assert "Official Claw-Eval tool policy" not in task.prompt
    assert "claw_tool" not in task.prompt
    assert "live bridge" not in task.prompt.lower()
    assert task.row["query"] == "Use the available tools to answer the original task."
    assert "Bash" in task.metadata["sandbox_tools"]
    bash_spec = next(spec for spec in task.metadata["allowed_tool_specs"] if spec["name"] == "Bash")
    assert bash_spec["description"].startswith("Executes a given bash command and returns its output.")
    assert "run_in_background" in bash_spec["input_schema"]["properties"]
    system_text = miniharness_initial_messages(task)[0].text
    assert "- web_fetch: Fetch full webpage content for a URL." in system_text
    assert "- Bash: Executes a given bash command and returns its output." in system_text
    assert "claw_tool" not in system_text
    assert "live bridge" not in system_text.lower()


def test_prepare_task_isolates_service_ports_for_parallel_rows(tmp_path):
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "T901_parallel_ports"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        """
task_id: T901_parallel_ports
task_name: Parallel Port Fixture
prompt:
  text: Search with the official tool.
  language: en
services:
  - name: web_real
    command: python mock_services/web_real/server.py
    port: 9114
    health_check: http://localhost:9114/web/health
    health_check_method: GET
    reset_endpoint: http://localhost:9114/web/reset
tools:
  - name: web_search
    description: Search the public web.
    input_schema:
      type: object
      properties:
        query:
          type: string
      required: [query]
tool_endpoints:
  - tool_name: web_search
    url: http://localhost:9114/web/search
    method: POST
environment:
  timeout_seconds: 1800
  max_turns: 50
""".strip()
        + "\n",
        encoding="utf-8",
    )
    prompt_template = tmp_path / "prompt.md"
    prompt_template.write_text("Task {{TASK_ID}}\n{{CLAW_EVAL_TOOLS}}\n", encoding="utf-8")
    cfg = {
        "project": {"root": str(tmp_path)},
        "benchmark": {
            "tasks_dir": str(tasks_dir),
            "fixture_root": str(tmp_path / "fixtures"),
            "prompt_template": str(prompt_template),
            "live_tool_bridge": False,
            "isolate_service_ports": True,
            "service_port_base": 19000,
            "service_port_stride": 10,
        },
    }

    benchmark = OpenClawBenchmark(cfg, tmp_path / "run")
    task = benchmark.prepare_task(
        {
            "task_id": "T901_parallel_ports",
            "query": "Find the evidence.",
            "language": "en",
            "_harness_eval_row_index": 2,
        }
    )

    policy = json.loads((task.output_dir / "claw_eval_tool_policy.json").read_text(encoding="utf-8"))

    assert task.metadata["services"][0]["port"] == 19020
    assert task.metadata["services"][0]["health_check"] == "http://localhost:19020/web/health"
    assert task.metadata["services"][0]["reset_endpoint"] == "http://localhost:19020/web/reset"
    assert task.metadata["tool_endpoints"][0]["url"] == "http://localhost:19020/web/search"
    assert policy["service_port_isolation"] == {
        "enabled": True,
        "row_index": 2,
        "port_map": {"9114": 19020},
    }
