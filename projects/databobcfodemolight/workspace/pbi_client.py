"""
pbi_client.py — Power BI Desktop MCP client.

Provides:
  list_pbi_instances()  — discover all open PBI Desktop models (no prior connection needed)
  PBIClient             — persistent MCP session for running DAX queries
"""

import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import POWERBI_EXE


# ── Instance discovery ────────────────────────────────────────────────────────

async def list_pbi_instances() -> list[dict]:
    """
    Start a temporary MCP session to discover all open Power BI Desktop models.

    Calls ListLocalInstances to find running Desktop processes, then for each
    connects and calls database_operations List to resolve the database GUID.

    Returns a list of dicts:
        display_name      — PBI Desktop window title, e.g. "FINANCE - Semantic Model"
        connection_string — e.g. "Data Source=localhost:58966;..."
        database          — database GUID, e.g. "270fbf16-..."
        port              — TCP port number (int)
    """
    params    = StdioServerParameters(command=POWERBI_EXE, args=["--start"], env={})
    transport = None
    session   = None
    result: list[dict] = []

    try:
        transport = stdio_client(params)
        r, w      = await transport.__aenter__()
        session   = ClientSession(r, w)
        await session.__aenter__()
        await session.initialize()

        # Step 1 — list open PBI Desktop processes
        raw   = await session.call_tool("connection_operations",
                    {"request": {"operation": "ListLocalInstances"}})
        data  = json.loads(raw.content[0].text if raw.content else "{}")
        insts = data.get("data", [])
        print(f"[PBI] {len(insts)} Desktop instance(s) found")

        # Step 2 — for each instance resolve database GUID
        for inst in insts:
            conn_str = inst.get("connectionString", "")
            title    = inst.get("parentWindowTitle", "Power BI Model")
            port     = inst.get("port", 0)

            try:
                # Connect without initialCatalog — Desktop always has exactly
                # one model loaded, so the MCP server auto-selects it.
                await session.call_tool("connection_operations",
                    {"request": {"operation": "Connect",
                                 "connectionString": conn_str}})

                db_raw  = await session.call_tool("database_operations",
                              {"request": {"operation": "List"}})
                db_data = json.loads(db_raw.content[0].text if db_raw.content else "{}")

                for db in db_data.get("data", []):
                    db_id = db.get("id", db.get("name", ""))
                    result.append({
                        "display_name":      title,
                        "connection_string": conn_str,
                        "database":          db_id,
                        "port":              port,
                    })
                    print(f"[PBI]  ↳ '{title}'  port={port}  db={db_id[:8]}...")

            except Exception as e:
                print(f"[PBI] Could not get DB for '{title}': {e}")
                result.append({
                    "display_name":      title,
                    "connection_string": conn_str,
                    "database":          "",
                    "port":              port,
                })

    except Exception as e:
        print(f"[PBI] Discovery error: {e}")

    finally:
        try:
            if session:   await session.__aexit__(None, None, None)
        except Exception: pass
        try:
            if transport: await transport.__aexit__(None, None, None)
        except Exception: pass

    return result


# ── Persistent MCP session ────────────────────────────────────────────────────

class PBIClient:
    """
    Persistent MCP session connected to a specific Power BI Desktop model.

    Usage:
        pbi = PBIClient()
        await pbi.connect(conn_str, db_guid)
        result = await pbi.dax("EVALUATE ...")
        await pbi.disconnect()
    """

    def __init__(self):
        self._session          = None
        self._transport        = None
        self.connection_string = ""
        self.database_guid     = ""

    async def connect(self, conn_str: str, db_guid: str):
        """Open an MCP session and connect to the given model."""
        params          = StdioServerParameters(command=POWERBI_EXE, args=["--start"], env={})
        self._transport = stdio_client(params)
        r, w            = await self._transport.__aenter__()
        self._session   = ClientSession(r, w)
        await self._session.__aenter__()
        await self._session.initialize()
        res = await self._session.call_tool("connection_operations", {"request": {
            "operation":       "Connect",
            "connectionString": conn_str,
            "initialCatalog":   db_guid,
        }})
        self.connection_string = conn_str
        self.database_guid     = db_guid
        print(f"[PBI] Connected: {res.content[0].text[:80] if res.content else 'ok'}")

    async def dax(self, query: str) -> dict:
        """Execute a DAX query and return the parsed JSON response."""
        res = await self._session.call_tool("dax_query_operations",
                  {"request": {"operation": "Execute", "query": query}})
        raw = res.content[0].text if res.content else "{}"
        return json.loads(raw)

    async def disconnect(self):
        """Close the MCP session gracefully, swallowing cleanup errors."""
        try:
            if self._session:   await self._session.__aexit__(None, None, None)
            if self._transport: await self._transport.__aexit__(None, None, None)
        except Exception:
            pass
        self._session   = None
        self._transport = None
