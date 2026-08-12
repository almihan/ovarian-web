"""Thin, lazy Modal client wrapper.

The Railway image imports this module without importing CUDA, PyTorch, or the
CellExLink inference stack.  Modal is contacted only when Stage 2 is submitted
or an existing remote call is reconciled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from backend.config import settings


@dataclass(slots=True, frozen=True)
class ModalPollResult:
    state: str
    result: dict[str, Any] | None = None
    error: str | None = None


class ModalAnnotationExecutor:
    def _function(self):
        import modal

        kwargs: dict[str, Any] = {}
        if settings.modal_environment:
            kwargs["environment_name"] = settings.modal_environment
        return modal.Function.from_name(
            settings.modal_app_name,
            settings.modal_function_name,
            **kwargs,
        )

    def submit(self, payload: Mapping[str, Any]) -> str:
        call = self._function().spawn(dict(payload))
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        if not call_id:
            raise RuntimeError("Modal did not return a function-call ID.")
        return str(call_id)

    def poll(self, call_id: str) -> ModalPollResult:
        import modal

        try:
            call = modal.FunctionCall.from_id(str(call_id))
            result = call.get(timeout=0)
        except TimeoutError:
            return ModalPollResult(state="running")
        except Exception as exc:
            exception_module = getattr(modal, "exception", None)
            output_expired = getattr(exception_module, "OutputExpiredError", ())
            if output_expired and isinstance(exc, output_expired):
                return ModalPollResult(
                    state="expired",
                    error="Modal no longer retains this function result.",
                )

            # Function execution failures are terminal. Authentication, transport,
            # or temporary Modal API failures must not incorrectly fail a GPU job;
            # the signed callback can still complete it after Railway recovers.
            terminal_names = {"RemoteError", "UserCodeException", "ExecutionError"}
            terminal_types = tuple(
                candidate
                for name in terminal_names
                if isinstance((candidate := getattr(exception_module, name, None)), type)
            )
            if terminal_types and isinstance(exc, terminal_types):
                return ModalPollResult(state="failed", error=str(exc))
            return ModalPollResult(state="unavailable", error=str(exc))

        if result is None:
            return ModalPollResult(state="completed", result={})
        if not isinstance(result, dict):
            return ModalPollResult(
                state="failed",
                error="Modal returned an unexpected result type.",
            )
        return ModalPollResult(state="completed", result=dict(result))


modal_executor = ModalAnnotationExecutor()

__all__ = ["ModalAnnotationExecutor", "ModalPollResult", "modal_executor"]
