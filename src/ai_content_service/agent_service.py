"""Install the provisioning agent as an image-style script plus supervisor conf.

Supervisord's ``environment=`` parser cannot safely carry arbitrary shell
values: escaped double quotes are rejected and apostrophes may be silently
truncated.  The Vast template instead stores real variables in a shell script
and leaves only ``PROC_NAME`` in the conf.  This module follows that stable
convention without importing the image's private provisioner implementation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Final

import structlog

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .config import Settings

AGENT_PROGRAM_NAME: Final = "aisha-agent"
AGENT_SCRIPT_PATH: Final = Path("/opt/supervisor-scripts/aisha-agent.sh")
AGENT_CONF_PATH: Final = Path("/etc/supervisor/conf.d/aisha-agent.conf")
log = structlog.get_logger()

_SECRET_ENV_KEY = re.compile(r"(?:_TOKEN|_SECRET|_KEY)")

# Deploy-time one-shots are meaningful for the bootstrap invocation only, not
# a long-lived command agent. Keep every exclusion justified beside the name.
ENV_DENYLIST: frozenset[str] = frozenset(
    {
        "ACS_APEX_OPERATION_ID",  # Bootstrap operation id must not leak into later commands.
        "ACS_BUNDLE",  # Bootstrap bundle selection must not become an agent default.
        "ACS_BUNDLE_VERSION",  # Bootstrap bundle version must not become an agent default.
        "ACS_MODELS_ONLY",  # Bootstrap mode is selected by the one-shot shell script.
        "ACS_NO_VERIFY",  # Bootstrap verification policy is not a command policy.
        "ACS_DRY_RUN",  # A one-shot diagnostic must never make queued work dry-run.
    }
)


def shell_escape(value: str) -> str:
    """Escape a value for insertion into a double-quoted shell string."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")


def render_startup_script(*, acs_bin: Path, workdir: Path, environment: Mapping[str, str]) -> str:
    """Render the secret-bearing startup script; callers must never log it."""
    exports = "\n".join(
        f'export {key}="{shell_escape(value)}"'
        for key, value in sorted(environment.items())
        if key.startswith("ACS_") and key not in ENV_DENYLIST
    )
    exports_block = exports or "# No ACS_* variables were present at installation time."
    return f"""#!/bin/bash
utils=/opt/supervisor-scripts/utils
[ -d \"${{utils}}\" ] && {{ . \"${{utils}}/logging.sh\"; . \"${{utils}}/environment.sh\"; }}

# Wait for provisioning to complete before this process can claim work.
while [ -f \"/.provisioning\" ]; do
    echo \"$PROC_NAME startup paused until instance provisioning has completed (/.provisioning present)\"
    sleep 5
done

# Set environment variables
{exports_block}

# Launch application
cd \"{shell_escape(str(workdir))}\"
exec \"{shell_escape(str(acs_bin))}\" agent run
"""


def render_supervisor_conf(*, script_path: Path) -> str:
    """Render the non-secret supervisord program declaration."""
    return f"""[program:{AGENT_PROGRAM_NAME}]
environment=PROC_NAME=\"%(program_name)s\"
command={script_path}
autostart=true
autorestart=true
exitcodes=0
startsecs=5
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=30
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
"""


def install_agent_service(
    settings: Settings, *, dry_run: bool = False, show_secrets: bool = False
) -> tuple[Path, Path]:
    """Create the private startup script and public supervisor conf.

    The script is opened with mode ``0o700`` from its first creation syscall;
    it is never created world-readable before its callback token is written.
    """
    script_path = settings.agent_script_path
    conf_path = settings.agent_supervisor_conf_path
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("ACS_") and key not in ENV_DENYLIST
    }
    script = render_startup_script(
        acs_bin=settings.agent_acs_bin,
        workdir=settings.agent_workdir,
        environment=environment,
    )
    conf = render_supervisor_conf(script_path=script_path)
    log.info("agent.service.installing", acs_bin=str(settings.agent_acs_bin))
    if dry_run:
        dry_run_environment = environment if show_secrets else _redact_environment(environment)
        print(
            render_startup_script(
                acs_bin=settings.agent_acs_bin,
                workdir=settings.agent_workdir,
                environment=dry_run_environment,
            ),
            end="",
        )
        print(conf, end="")
        return script_path, conf_path

    script_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(script_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o700)
    try:
        os.fchmod(fd, 0o700)
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as script_file:
            script_file.write(script)
    finally:
        os.close(fd)
    conf_path.write_text(conf, encoding="utf-8")
    conf_path.chmod(0o644)
    return script_path, conf_path


def _redact_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return a display-only environment with credentials hidden."""
    return {
        key: "***redacted***" if _SECRET_ENV_KEY.search(key) else value
        for key, value in environment.items()
    }
