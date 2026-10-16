"""ADA Runtime OS (Feature 1) — control plane for the assistant.

The control plane (lifecycle, policy, events) lives here; the runtime plane
(the actual tools and subsystems) stays where it was. Subsystems are wrapped
in thin :class:`~runtime.services.Service` adapters — delegated, never
rewritten — registered in a :class:`~runtime.services.ServiceRegistry`, and
run under a :class:`~runtime.supervisor.Supervisor` that restarts crashed
services with backoff and ticks health checks onto the
:class:`~runtime.bus.EventBus`.

Kept deliberately import-light: stdlib + sqlite3 only, so importing this
package can never break the rest of the backend.
"""
