"""Long-lived, one-in-flight provisioning command agent."""

from __future__ import annotations

import asyncio
import random
import signal
import socket
from typing import TYPE_CHECKING

import httpx
import structlog

from .agent_contract import CommandParseError, parse_command
from .callback_client import CallbackClient
from .command_executor import CommandExecutor
from .provisioning_reporter import ProvisioningReporter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .config import Settings

log = structlog.get_logger()


class ProvisioningAgent:
    """Claim and execute Apex commands serially for one GPU session."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: CallbackClient | None = None,
        reporter: ProvisioningReporter | None = None,
        executor: CommandExecutor | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or CallbackClient.from_settings(settings)
        self._reporter = reporter or ProvisioningReporter(
            self._client,
            session_id=settings.apex_session_id,
        )
        self._owns_reporter = reporter is None
        self._client_is_owned_by_reporter = reporter is None
        self._executor = executor or CommandExecutor(settings, reporter=self._reporter)
        self._stop_requested = False
        self._stop_event: asyncio.Event | None = None
        self._random = random.SystemRandom()
        self.agent_id = settings.agent_id or f"{settings.apex_session_id}:{socket.gethostname()}"

    async def run(self) -> None:
        """Run until a signal requests a graceful stop between commands."""
        self._stop_event = asyncio.Event()
        if self._stop_requested:
            self._stop_event.set()
        backoff = self._settings.agent_poll_interval_seconds
        if self._owns_reporter:
            async with self._reporter:
                await self._claim_commands(backoff)
        else:
            await self._claim_commands(backoff)
        if not self._client_is_owned_by_reporter:
            await self._client.aclose()

    async def _claim_commands(self, backoff: float) -> None:
        """Claim and dispatch commands until stopping between completed commands."""
        while not self._stop_requested:
            status, body = await self._client.claim_command(
                self._settings.apex_session_id,
                self.agent_id,
            )
            if status == httpx.codes.OK and body is not None:
                # Apex has now assigned this work to us.  A stop request that
                # races the claim must wait until the assigned command finishes.
                backoff = self._settings.agent_poll_interval_seconds
                await self._execute_claim(body)
                if self._stop_requested:
                    break
                # Claim immediately after every command; batches drain at work speed.
                continue
            if self._stop_requested:
                break
            if status == httpx.codes.NO_CONTENT:
                backoff = self._settings.agent_poll_interval_seconds
                await self._sleep_with_jitter(backoff)
                continue
            if httpx.codes.BAD_REQUEST <= status < httpx.codes.INTERNAL_SERVER_ERROR:
                log.error("agent.claim.rejected", status=status)
            elif status != 0:
                log.warning("agent.claim.unexpected_status", status=status)
            await self._sleep_with_jitter(backoff)
            backoff = min(backoff * 2, self._settings.agent_max_backoff_seconds)

    def request_stop(self) -> None:
        """Stop claiming after the currently executing command finishes."""
        self._stop_requested = True
        if self._stop_event is not None:
            self._stop_event.set()

    def install_signal_handlers(self) -> None:
        """Register graceful SIGTERM/SIGINT handling when an event loop allows it."""
        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGTERM, self.request_stop)
            loop.add_signal_handler(signal.SIGINT, self.request_stop)
        except (NotImplementedError, RuntimeError, ValueError):
            log.debug("agent.signal_handlers.unavailable")

    async def _execute_claim(self, body: Mapping[str, object]) -> None:
        """Parse a claimed envelope and report only correctly attributable failures.

        A terminal event needs both the operation id and a known command kind.
        If either is malformed, logging is deliberate: inventing an
        ``operation_kind`` would make the event disagree with Apex's envelope.
        """
        try:
            command = parse_command(body)
        except CommandParseError as exc:
            if exc.operation_id is None or exc.kind is None:
                log.error("agent.command.unparseable", error=str(exc), exc_info=True)
                return
            log.error(
                "agent.command.unparseable",
                operation_id=exc.operation_id,
                error=str(exc),
                exc_info=True,
            )
            await self._executor.report_unparseable(
                operation_id=exc.operation_id,
                kind=exc.kind,
                detail=str(exc),
            )
            return
        await self._executor.execute(command)

    async def _sleep_with_jitter(self, interval: float) -> None:
        """Sleep at least half an interval to avoid synchronized fleet polling."""
        delay = interval / 2 + self._random.uniform(0, interval / 2)
        if self._stop_event is None:
            await asyncio.sleep(delay)
            return
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        except TimeoutError:
            return
