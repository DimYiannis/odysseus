"""image_gen is spawned on its first tool call, exactly once, instead of at boot."""

from __future__ import annotations

import asyncio

from src.mcp_manager import McpManager


class _Session:
    pass


def _manager(monkeypatch):
    mgr = McpManager()
    spawns = []

    async def fake_reconnect(server_id):
        spawns.append(server_id)
        await asyncio.sleep(0.01)
        mgr._sessions[server_id] = _Session()
        return True

    async def fake_do_call(session, tool_name, arguments):
        return {"stdout": tool_name, "stderr": "", "exit_code": 0}

    monkeypatch.setattr(mgr, "_reconnect_builtin", fake_reconnect)
    monkeypatch.setattr(mgr, "_do_call", fake_do_call)
    return mgr, spawns


async def test_concurrent_first_calls_spawn_lazy_builtin_once(monkeypatch):
    mgr, spawns = _manager(monkeypatch)

    results = await asyncio.gather(
        mgr.call_tool("mcp__image_gen__generate_image", {"prompt": "a"}),
        mgr.call_tool("mcp__image_gen__generate_image", {"prompt": "b"}),
    )

    assert spawns == ["image_gen"]
    assert [r["exit_code"] for r in results] == [0, 0]


async def test_eager_or_unknown_servers_are_not_spawned_on_call(monkeypatch):
    mgr, spawns = _manager(monkeypatch)

    email = await mgr.call_tool("mcp__email__list_emails", {})
    unknown = await mgr.call_tool("mcp__someuser__tool", {})

    assert spawns == []
    assert "not connected" in email["error"]
    assert "not connected" in unknown["error"]
