#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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
import contextlib
import json
import os
import re
from abc import ABC
from typing import Any
import pandas as pd
import pymysql
import psycopg2
import pyodbc
from agent.tools.base import ToolParamBase, ToolBase, ToolMeta
from common.connection_utils import timeout


class ExeSQLParam(ToolParamBase):
    """
    Define the ExeSQL component parameters.
    """

    def __init__(self):
        self.meta:ToolMeta = {
            "name": "execute_sql",
            "description": "This is a tool that can execute SQL.",
            "parameters": {
                "sql": {
                    "type": "string",
                    "description": "The SQL needs to be executed.",
                    "default": "{sys.query}",
                    "required": True
                }
            }
        }
        super().__init__()
        self.db_type = "mysql"
        self.database = ""
        self.username = ""
        self.host = ""
        self.port = 3306
        self.password = ""
        self.max_records = 1024

    def check(self):
        self.check_valid_value(self.db_type, "Choose DB type", ['mysql', 'postgres', 'mariadb', 'mssql', 'IBM DB2', 'trino', 'oceanbase'])
        self.check_empty(self.database, "Database name")
        self.check_empty(self.username, "database username")
        self.check_empty(self.host, "IP Address")
        self.check_positive_integer(self.port, "IP Port")
        if self.db_type != "trino":
            self.check_empty(self.password, "Database password")
        self.check_positive_integer(self.max_records, "Maximum number of records")
        if self.database == "rag_flow":
            if self.host == "ragflow-mysql":
                raise ValueError("For the security reason, it does not support database named rag_flow.")
            if self.password == "infini_rag_flow":
                raise ValueError("For the security reason, it does not support database named rag_flow.")

    def get_input_form(self) -> dict[str, dict]:
        return {
            "sql": {
                "name": "SQL",
                "type": "line"
            }
        }


class ExeSQL(ToolBase, ABC):
    component_name = "ExeSQL"

    # Statements that mutate or alter schema/data/permissions are blocked
    # outright, regardless of how their variables are substituted. This is
    # defense-in-depth on top of (never a substitute for) the per-value
    # escaping/parameterization in `_resolve_sql_statement`: it is the only
    # protection available for the one case that mechanism deliberately
    # does not touch -- a whole statement bound to a single variable whose
    # value is meant to BE a complete SQL statement (see the "whole
    # statement is one variable" branch below).
    _BLOCKED_STATEMENT_PREFIX = re.compile(
        r"^(insert|update|delete|drop|alter|truncate|create|grant|revoke|replace|exec(?:ute)?|call|merge)\b",
        flags=re.IGNORECASE,
    )

    @staticmethod
    def _split_raw_statements(sql: str) -> list[str]:
        """Split *sql* on ";" into individual statement strings, applying the
        cleanup this tool has always applied per statement (stripping
        Markdown code fences and "[ID:n]" citation tags, dropping blanks).

        This must always run on text that has *not* had any canvas/flow
        variable substituted into it yet (see `_resolve_sql_statement`) --
        otherwise a substituted value containing its own ";" could
        terminate the current statement and smuggle in an extra one that
        runs as its own top-level statement.
        """
        stmts = []
        for chunk in sql.split(";"):
            stmt = chunk.replace("```", "").strip()
            if not stmt:
                continue
            stmt = re.sub(r"\[ID:[0-9]+\]", "", stmt)
            stmts.append(stmt)
        return stmts

    def _resolve_sql_statement(self, raw_stmt: str) -> list[tuple[str, Any]]:
        """Resolve the canvas/flow variables referenced by one
        *unsubstituted* SQL statement into ready-to-run (sql_text, params)
        pairs.

        `params` is:
          - `None` -- `sql_text` is complete SQL, run as `cursor.execute(sql_text)`.
          - a `dict` -- `sql_text` contains "%(name)s" placeholders for
            pyformat-paramstyle drivers (PyMySQL, psycopg2); run as
            `cursor.execute(sql_text, params)`.
          - a `list` -- `sql_text` contains "?" placeholders, one per list
            entry in left-to-right order, for qmark-paramstyle drivers
            (pyodbc); run the same way.

        A substituted value is never spliced into `sql_text` as raw,
        unescaped text -- it is always bound as a parameter, or (for
        dialects whose driver this code does not bind parameters through
        directly) substituted only after correct dialect string-literal
        escaping. The one narrow exception is the "whole statement is one
        variable" branch below, which does not weaken this guarantee for
        variables used as values.
        """
        var_refs = self.get_input_elements_from_text(raw_stmt)
        args: dict[str, str] = {}
        for k, o in var_refs.items():
            v = o["value"]
            if not isinstance(v, str):
                try:
                    v = json.dumps(v, ensure_ascii=False)
                except Exception:
                    v = str(v)
            args[k] = v
            self.set_input_value(k, v)

        if not var_refs:
            return [(raw_stmt, None)]

        # Whole-statement case: the statement is *exactly* one variable
        # reference and nothing else, e.g. `sql: "{Agent:xyz@content}"`.
        # This is the documented text-to-SQL-agent pattern (an upstream
        # LLM/agent component emits a complete query, or several ";"-
        # separated ones, and ExeSQL just runs it) -- the "value" here IS
        # the SQL, not a value to quote/bind into it, so treating it as a
        # scalar would break the query rather than secure it. This is
        # unchanged from the tool's pre-fix behavior for this exact shape,
        # including re-splitting the substituted text on ";" the same way
        # the outer statement list is built; it relies on the DDL/DML/etc.
        # blocklist as its (pre-existing) defense-in-depth.
        if len(var_refs) == 1:
            (only_key,) = var_refs.keys()
            if raw_stmt.strip() == "{%s}" % only_key:
                substituted = self.string_format(raw_stmt, {only_key: args[only_key]})
                return [(s, None) for s in self._split_raw_statements(substituted)]

        # Every other case: the statement is fixed SQL text (written by the
        # flow designer, or an agent-authored template) with one or more
        # values spliced in -- e.g. `SELECT * FROM orders WHERE customer_id
        # = {customer_id}`. Every such value is treated strictly as a VALUE:
        # it is bound through the target driver's own parameter mechanism
        # so it can never terminate the token it sits in, start a new
        # statement, or comment out the remainder of the query, no matter
        # what characters it contains.
        #
        # NOTE on identifier positions: a bind parameter (like a quoted
        # literal) can only ever stand in for a *value*, never for an
        # identifier such as a table or column name -- no SQL dialect's
        # placeholder syntax supports that. No shipped ExeSQL template or
        # test in this repository uses a variable for an identifier; if a
        # flow did, this now produces a SQL syntax error instead of
        # executing unescaped, attacker-influenced text as an identifier,
        # which is the safe direction for that trade-off to fail in.
        if self._param.db_type in ("mysql", "mariadb", "oceanbase", "postgres"):
            # Both PyMySQL and psycopg2 use Python "%"-style pyformat
            # placeholders and interpolate the query through "%"
            # internally, so any literal "%" already in the template (e.g.
            # `LIKE '%foo%'`) must be doubled first. Values themselves need
            # no such treatment -- the driver inserts them, already
            # escaped, after this step, so a "%" inside a *value* is never
            # re-interpreted.
            templated = raw_stmt.replace("%", "%%")
            alternation = "|".join(re.escape(k) for k in var_refs)
            templated = re.sub(r"\{(%s)\}" % alternation, lambda m: "%%(%s)s" % m.group(1), templated)
            return [(templated, dict(args))]

        if self._param.db_type == "mssql":
            params: list[str] = []

            def _qmark(m):
                params.append(args[m.group(1)])
                return "?"

            alternation = "|".join(re.escape(k) for k in var_refs)
            templated = re.sub(r"\{(%s)\}" % alternation, _qmark, raw_stmt)
            return [(templated, params)]

        # Trino and IBM DB2: this tool does not drive either driver's
        # bind-parameter API directly, so fall back to correct dialect
        # string-literal escaping instead (both use plain ANSI-SQL string
        # literals: wrap in single quotes, double any embedded single
        # quote, no backslash escapes).
        templated = raw_stmt
        for k, v in args.items():
            templated = re.sub(r"\{%s\}" % re.escape(k), lambda _m, val=v: ExeSQL._quote_sql_literal(val), templated)
        return [(templated, None)]

    @staticmethod
    def _quote_sql_literal(value: str) -> str:
        """Render *value* as a safely-escaped ANSI-SQL string literal.

        Used only for dialects (Trino, IBM DB2) whose Python driver this
        tool does not bind parameters through directly. NUL bytes are
        stripped: they are not valid inside a SQL string literal in any
        supported dialect, and some engines/drivers silently truncate on
        them, which could otherwise hide the tail of an injected payload
        from anyone reading the executed SQL back.
        """
        if value is None:
            return "NULL"
        return "'" + value.replace("\x00", "").replace("'", "''") + "'"

    @timeout(int(os.environ.get("COMPONENT_EXEC_TIMEOUT", 60)))
    def _invoke(self, **kwargs):
        if self.check_if_canceled("ExeSQL processing"):
            return

        def convert_decimals(obj):
            from decimal import Decimal
            import math
            if isinstance(obj, float):
                # Handle NaN and Infinity which are not valid JSON values
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            if isinstance(obj, Decimal):
                return float(obj)  # 或 str(obj)
            elif isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals(item) for item in obj]
            return obj

        sql = kwargs.get("sql")
        if not sql:
            raise Exception("SQL for `ExeSQL` MUST not be empty.")

        if self.check_if_canceled("ExeSQL processing"):
            return

        # Security: split into individual statements *before* substituting
        # any canvas/flow variable, and never splice a substituted value
        # into the SQL text as a raw, unescaped string. See
        # `_resolve_sql_statement` for the full rationale. Splitting first
        # is what stops a substituted value containing its own ";" from
        # being able to smuggle in an extra top-level statement; escaping/
        # binding every value is what stops it from breaking out of the
        # token (string literal, comparison, etc.) it was substituted into.
        statements: list[tuple[str, Any]] = []
        for raw_stmt in self._split_raw_statements(sql):
            statements.extend(self._resolve_sql_statement(raw_stmt))

        if self.check_if_canceled("ExeSQL processing"):
            return
        if self._param.db_type in ["mysql", "mariadb"]:
            db = pymysql.connect(db=self._param.database, user=self._param.username, host=self._param.host,
                                 port=self._param.port, password=self._param.password)
        elif self._param.db_type == 'oceanbase':
            db = pymysql.connect(db=self._param.database, user=self._param.username, host=self._param.host,
                                 port=self._param.port, password=self._param.password, charset='utf8mb4')
        elif self._param.db_type == 'postgres':
            db = psycopg2.connect(dbname=self._param.database, user=self._param.username, host=self._param.host,
                                  port=self._param.port, password=self._param.password)
        elif self._param.db_type == 'mssql':
            conn_str = (
                    r'DRIVER={ODBC Driver 17 for SQL Server};'
                    r'SERVER=' + self._param.host + ',' + str(self._param.port) + ';'
                    r'DATABASE=' + self._param.database + ';'
                    r'UID=' + self._param.username + ';'
                    r'PWD=' + self._param.password
            )
            db = pyodbc.connect(conn_str)
        elif self._param.db_type == 'trino':
            try:
                import trino
                from trino.auth import BasicAuthentication
            except Exception:
                raise Exception("Missing dependency 'trino'. Please install: pip install trino")

            def _parse_catalog_schema(db: str):
                if not db:
                    return None, None
                if "." in db:
                    c, s = db.split(".", 1)
                elif "/" in db:
                    c, s = db.split("/", 1)
                else:
                    c, s = db, "default"
                return c, s

            catalog, schema = _parse_catalog_schema(self._param.database)
            if not catalog:
                raise Exception("For Trino, `database` must be 'catalog.schema' or at least 'catalog'.")

            http_scheme = "https" if os.environ.get("TRINO_USE_TLS", "0") == "1" else "http"
            auth = None
            if http_scheme == "https" and self._param.password:
                auth = BasicAuthentication(self._param.username, self._param.password)

            try:
                db = trino.dbapi.connect(
                    host=self._param.host,
                    port=int(self._param.port or 8080),
                    user=self._param.username or "ragflow",
                    catalog=catalog,
                    schema=schema or "default",
                    http_scheme=http_scheme,
                    auth=auth
                )
            except Exception as e:
                raise Exception("Database Connection Failed! \n" + str(e))
        elif self._param.db_type == 'IBM DB2':
            import ibm_db
            conn_str = (
                f"DATABASE={self._param.database};"
                f"HOSTNAME={self._param.host};"
                f"PORT={self._param.port};"
                f"PROTOCOL=TCPIP;"
                f"UID={self._param.username};"
                f"PWD={self._param.password};"
            )
            try:
                conn = ibm_db.connect(conn_str, "", "")
            except Exception as e:
                raise Exception("Database Connection Failed! \n" + str(e))

            try:
                sql_res = []
                formalized_content = []
                for single_sql, _params in statements:
                    if self.check_if_canceled("ExeSQL processing"):
                        return

                    if self._BLOCKED_STATEMENT_PREFIX.match(single_sql):
                        msg = "For security reasons, this type of statement is not supported."
                        sql_res.append({"content": msg})
                        formalized_content.append(msg)
                        continue

                    try:
                        stmt = ibm_db.exec_immediate(conn, single_sql)
                        rows = []
                        row = ibm_db.fetch_assoc(stmt)
                        while row and len(rows) < self._param.max_records:
                            if self.check_if_canceled("ExeSQL processing"):
                                return
                            rows.append(row)
                            row = ibm_db.fetch_assoc(stmt)

                        if not rows:
                            sql_res.append({"content": "No record in the database!"})
                            continue

                        df = pd.DataFrame(rows)
                        for col in df.columns:
                            if pd.api.types.is_datetime64_any_dtype(df[col]):
                                df[col] = df[col].dt.strftime("%Y-%m-%d")

                        df = df.where(pd.notnull(df), None)

                        sql_res.append(convert_decimals(df.to_dict(orient="records")))
                        formalized_content.append(df.to_markdown(index=False, floatfmt=".6f"))
                    except Exception as e:
                        # Keep the node alive on a bad statement: report and continue.
                        with contextlib.suppress(Exception):
                            ibm_db.rollback(conn)
                        msg = f"SQL Execution Failed: {single_sql}\n{str(e)}"
                        sql_res.append({"content": msg})
                        formalized_content.append(msg)
                        continue
            finally:
                with contextlib.suppress(Exception):
                    ibm_db.close(conn)

            self.set_output("json", sql_res)
            self.set_output("formalized_content", "\n\n".join(formalized_content))
            return self.output("formalized_content")
        try:
            cursor = db.cursor()
        except Exception as e:
            with contextlib.suppress(Exception):
                db.close()
            raise Exception("Database Connection Failed! \n" + str(e))

        try:
            sql_res = []
            formalized_content = []
            for single_sql, params in statements:
                if self.check_if_canceled("ExeSQL processing"):
                    return

                if self._BLOCKED_STATEMENT_PREFIX.match(single_sql):
                    msg = "For security reasons, this type of statement is not supported."
                    sql_res.append({"content": msg})
                    formalized_content.append(msg)
                    continue
                try:
                    if params is None:
                        cursor.execute(single_sql)
                    else:
                        cursor.execute(single_sql, params)
                    if cursor.rowcount == 0:
                        sql_res.append({"content": "No record in the database!"})
                        break
                    if self._param.db_type == 'mssql':
                        single_res = pd.DataFrame.from_records(cursor.fetchmany(self._param.max_records),
                                                               columns=[desc[0] for desc in cursor.description])
                    else:
                        single_res = pd.DataFrame([i for i in cursor.fetchmany(self._param.max_records)])
                        single_res.columns = [i[0] for i in cursor.description]

                    for col in single_res.columns:
                        if pd.api.types.is_datetime64_any_dtype(single_res[col]):
                            single_res[col] = single_res[col].dt.strftime('%Y-%m-%d')

                    single_res = single_res.where(pd.notnull(single_res), None)

                    sql_res.append(convert_decimals(single_res.to_dict(orient='records')))
                    formalized_content.append(single_res.to_markdown(index=False, floatfmt=".6f"))
                except Exception as e:
                    # A failing statement must not abort the node: report it and keep
                    # going so earlier results survive and later statements still run.
                    # The rollback clears PostgreSQL's aborted-transaction state, which
                    # would otherwise make every subsequent statement fail too.
                    with contextlib.suppress(Exception):
                        db.rollback()
                    msg = f"SQL Execution Failed: {single_sql}\n{str(e)}"
                    sql_res.append({"content": msg})
                    formalized_content.append(msg)
                    continue
        finally:
            with contextlib.suppress(Exception):
                cursor.close()
            with contextlib.suppress(Exception):
                db.close()

        self.set_output("json", sql_res)
        self.set_output("formalized_content", "\n\n".join(formalized_content))
        return self.output("formalized_content")

    def thoughts(self) -> str:
        return "Query sent—waiting for the data."
