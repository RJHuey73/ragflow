#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Regression tests for the ExeSQL agent tool's variable-substitution fix.

ExeSQL builds SQL from a flow-designer-authored template plus canvas/flow
variables (whose value can be raw, attacker-influenced end-user chat text --
the tool's own default binding for its ``sql`` parameter is ``{sys.query}``).
Before the fix, a variable's value was spliced into the SQL text as a raw,
unescaped string via naive text substitution, then the *substituted* text was
split on ";" and each piece run with ``cursor.execute(single_sql)`` -- meaning
a value like ``"'; DROP TABLE orders--"`` could break out of its surrounding
SQL, splice in an extra statement, or comment out the rest of the query.

These tests assert directly on what actually reaches ``cursor.execute`` for
a benign value and for an injection attempt, rather than on any network/DB
behavior -- exercising the real `agent.tools.exesql.ExeSQL` class end to end
via a fake DB-API connection/cursor.
"""

import sys
import types


def _install_tiktoken_stub_if_unavailable():
    """`common/token_utils.py` calls `tiktoken.get_encoding("cl100k_base")` at
    *import* time, which downloads the encoding from OpenAI's blob storage on
    a cache miss. That network call is unrelated to anything ExeSQL does, and
    is unreachable in a network-restricted test sandbox -- stub it out the
    same way `test_browser_use_component.py` stubs `cv2` for an analogous
    "heavy, irrelevant, real-import-time side effect" situation.
    """
    try:
        import tiktoken

        tiktoken.get_encoding("cl100k_base")
        return
    except Exception:
        pass

    stub = types.ModuleType("tiktoken")

    class _FakeEncoding:
        def encode(self, text):
            return list(text.encode("utf-8", errors="ignore"))

    stub.get_encoding = lambda _name: _FakeEncoding()
    sys.modules["tiktoken"] = stub


def _install_rag_prompts_stub_if_unavailable():
    """`agent/tools/base.py` imports `rag.prompts.generator.kb_prompt` so
    *other* tools can build knowledge-base prompts -- `ExeSQL` never calls
    it. The real module transitively pulls in `rag.nlp.rag_tokenizer`'s
    Chinese/English NLP tokenizer stack, unrelated to this fix. Stub it, same
    rationale/pattern as the tiktoken stub above.
    """
    try:
        import rag.prompts.generator  # noqa: F401
        return
    except Exception:
        pass

    fake_rag = sys.modules.get("rag") or types.ModuleType("rag")
    fake_prompts = types.ModuleType("rag.prompts")
    fake_generator = types.ModuleType("rag.prompts.generator")
    fake_generator.kb_prompt = lambda *_args, **_kwargs: ""
    fake_prompts.generator = fake_generator
    fake_rag.prompts = fake_prompts
    sys.modules["rag"] = fake_rag
    sys.modules["rag.prompts"] = fake_prompts
    sys.modules["rag.prompts.generator"] = fake_generator


_install_tiktoken_stub_if_unavailable()
_install_rag_prompts_stub_if_unavailable()

from agent.tools.exesql import ExeSQL, ExeSQLParam  # noqa: E402


class _FakeCanvas:
    """Minimal stand-in for `agent.canvas.Canvas`: only what
    `get_input_elements_from_text`/`check_if_canceled` touch.
    """

    def __init__(self, values):
        self._values = values

    def is_canceled(self):
        return False

    def get_variable_value(self, token):
        return self._values.get(token)

    def get_component_name(self, cpn_id):
        return cpn_id


class _FakeCursor:
    """Records every `execute(sql, params)` call and simulates one row of
    a `("id", "customer_id")`-shaped result set, matching what the mysql/
    postgres branch of `ExeSQL._invoke` expects from `cursor.description`/
    `cursor.fetchmany`.
    """

    def __init__(self):
        self.calls: list[tuple[str, object]] = []
        self.description = [("id",), ("customer_id",)]
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        self.rowcount = 1

    def fetchmany(self, _n):
        return [(1, "42")]

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.rollback_calls = 0

    def cursor(self):
        return self._cursor

    def rollback(self):
        self.rollback_calls += 1

    def close(self):
        pass


def _build_exesql(monkeypatch, canvas_values, db_type="mysql"):
    import agent.tools.exesql as exesql_module

    component = ExeSQL.__new__(ExeSQL)
    component._canvas = _FakeCanvas(canvas_values)
    component._param = ExeSQLParam()
    component._param.db_type = db_type
    component._param.database = "testdb"
    component._param.username = "root"
    component._param.host = "127.0.0.1"
    component._param.port = 3306
    component._param.password = "secret"
    component._param.max_records = 100

    cursor = _FakeCursor()
    connection = _FakeConnection(cursor)
    monkeypatch.setattr(exesql_module.pymysql, "connect", lambda **_kwargs: connection)
    return component, cursor


# --- (a) a benign parameterized query still executes correctly ------------


def test_value_substitution_uses_bound_parameter_not_raw_splice(monkeypatch):
    component, cursor = _build_exesql(monkeypatch, {"sys.query": "42"})

    result = component._invoke(sql="SELECT * FROM orders WHERE customer_id = {sys.query}")

    assert len(cursor.calls) == 1
    executed_sql, params = cursor.calls[0]
    # The customer-provided value is bound as a parameter -- never spliced
    # into the SQL text -- and the template is otherwise untouched.
    assert executed_sql == "SELECT * FROM orders WHERE customer_id = %(sys.query)s"
    assert params == {"sys.query": "42"}
    # And the query still actually runs and returns the (fake) row (one
    # entry in `json` per executed statement, each itself the statement's
    # list of row-dicts -- unchanged output shape from before the fix).
    assert component.output("json") == [[{"id": 1, "customer_id": "42"}]]
    assert "42" in result


def test_repeated_variable_reference_binds_once_and_reuses_it(monkeypatch):
    component, cursor = _build_exesql(monkeypatch, {"sys.query": "7"})

    component._invoke(sql="SELECT * FROM orders WHERE customer_id = {sys.query} OR referred_by = {sys.query}")

    executed_sql, params = cursor.calls[0]
    assert executed_sql == "SELECT * FROM orders WHERE customer_id = %(sys.query)s OR referred_by = %(sys.query)s"
    assert params == {"sys.query": "7"}


# --- (b) an injection attempt cannot alter the executed SQL's structure ----


def test_quote_breakout_and_stacked_query_payload_is_confined_to_a_parameter(monkeypatch):
    payload = "1'; DROP TABLE orders; SELECT * FROM users WHERE '1'='1"
    component, cursor = _build_exesql(monkeypatch, {"sys.query": payload})

    component._invoke(sql="SELECT * FROM orders WHERE customer_id = {sys.query}")

    # Exactly one statement is ever sent to the driver: the payload's own
    # ";"s did not smuggle in extra top-level statements (splitting happens
    # on the *template*, before substitution).
    assert len(cursor.calls) == 1
    executed_sql, params = cursor.calls[0]
    # The SQL structure is byte-identical to the benign case above --
    # attacker-controlled content changed only the *value* of a parameter,
    # never the shape of the query.
    assert executed_sql == "SELECT * FROM orders WHERE customer_id = %(sys.query)s"
    assert params == {"sys.query": payload}
    # In particular, the raw payload text is *not* present in the executed
    # SQL string itself -- only inside the params dict passed alongside it.
    assert "DROP TABLE" not in executed_sql
    assert "'1'='1" not in executed_sql


def test_like_wildcard_percent_in_template_is_preserved(monkeypatch):
    """PyMySQL/psycopg2 interpolate "%(name)s" placeholders through Python's
    own "%" string-formatting operator, so a literal "%" already in the
    template (e.g. a LIKE wildcard) must be doubled to "%%" first, or the
    driver would raise on it / misinterpret it. Confirm the doubled form is
    exactly what is sent to the driver, and that formatting it the same way
    the driver would collapses back to the original, single-"%" wildcard --
    the value is still bound, never concatenated into the string.
    """
    component, cursor = _build_exesql(monkeypatch, {"sys.query": "smith"})

    component._invoke(sql="SELECT * FROM orders WHERE name LIKE '%{sys.query}%'")

    executed_sql, params = cursor.calls[0]
    assert executed_sql == "SELECT * FROM orders WHERE name LIKE '%%%(sys.query)s%%'"
    assert params == {"sys.query": "smith"}
    assert (executed_sql % params) == "SELECT * FROM orders WHERE name LIKE '%smith%'"


def test_blocklist_still_rejects_mutating_statements(monkeypatch):
    component, cursor = _build_exesql(monkeypatch, {"sys.query": "1"})

    component._invoke(sql="DELETE FROM orders WHERE customer_id = {sys.query}")

    assert cursor.calls == []
    assert "not supported" in component.output("formalized_content")


def test_widened_blocklist_also_rejects_drop(monkeypatch):
    component, cursor = _build_exesql(monkeypatch, {"sys.query": "orders"})

    component._invoke(sql="DROP TABLE orders")

    assert cursor.calls == []
    assert "not supported" in component.output("formalized_content")


# --- whole-statement-is-one-variable ("text-to-SQL agent") is unchanged ----


def test_whole_statement_variable_still_runs_as_raw_sql(monkeypatch):
    """The officially shipped `text2sql_data_expert.json` template binds the
    *entire* `sql` field to a single upstream-agent-produced variable, e.g.
    `"sql": "{Agent:WickedGoatsDivide@content}"`. That value IS a complete
    query (or several ";"-separated ones); quoting/binding it as a scalar
    would break it rather than secure it, so this exact shape must keep
    running unescaped, exactly as before the fix.
    """
    component, cursor = _build_exesql(monkeypatch, {"Agent:X@content": "SELECT * FROM orders"})

    component._invoke(sql="{Agent:X@content}")

    assert len(cursor.calls) == 1
    executed_sql, params = cursor.calls[0]
    assert executed_sql == "SELECT * FROM orders"
    assert params is None


# --- pure-function coverage for the non-pymysql/psycopg2 dialect paths ----


def test_quote_sql_literal_escapes_quotes_and_strips_nul_bytes():
    assert ExeSQL._quote_sql_literal("O'Reilly") == "'O''Reilly'"
    assert ExeSQL._quote_sql_literal("a\x00b") == "'ab'"
    assert ExeSQL._quote_sql_literal(None) == "NULL"


def test_mssql_uses_qmark_positional_binding(monkeypatch):
    component, _cursor = _build_exesql(monkeypatch, {"sys.query": "1'); DROP TABLE orders--"}, db_type="mssql")

    resolved = component._resolve_sql_statement("SELECT * FROM orders WHERE customer_id = {sys.query}")

    assert resolved == [("SELECT * FROM orders WHERE customer_id = ?", ["1'); DROP TABLE orders--"])]


def test_trino_falls_back_to_correct_literal_quoting(monkeypatch):
    component, _cursor = _build_exesql(monkeypatch, {"sys.query": "1'; DROP TABLE orders--"}, db_type="trino")

    resolved = component._resolve_sql_statement("SELECT * FROM orders WHERE customer_id = {sys.query}")

    assert resolved == [("SELECT * FROM orders WHERE customer_id = '1''; DROP TABLE orders--'", None)]


def test_statement_split_happens_before_substitution(monkeypatch):
    """A template with two real statements, the first carrying a variable
    whose value itself contains ";" -- must resolve to exactly two
    statements (the template's own two), not three (which would mean the
    value's ";" was treated as a statement separator).
    """
    component, _cursor = _build_exesql(monkeypatch, {"sys.query": "x'; DROP TABLE orders; --"})

    resolved = []
    for raw in component._split_raw_statements(
        "SELECT * FROM orders WHERE note = {sys.query}; SELECT * FROM accounts"
    ):
        resolved.extend(component._resolve_sql_statement(raw))

    assert len(resolved) == 2
    first_sql, first_params = resolved[0]
    assert first_sql == "SELECT * FROM orders WHERE note = %(sys.query)s"
    assert first_params == {"sys.query": "x'; DROP TABLE orders; --"}
    assert resolved[1] == ("SELECT * FROM accounts", None)
