import sys, json, subprocess, psycopg2, string, dataclasses, json
from typing import Union, TypeVar, Generic, Callable, Optional
from ivm_types import *
from expect_test import expect, test_case
from psycopg2.extras import LogicalReplicationConnection, StopReplication


@dataclasses.dataclass
class EmptyTable:
    pass


def assert_valid_id(s):
    for ch in s:
        if not (ch in string.ascii_letters or ch in string.digits or ch in "_"):
            raise Exception(
                "invalid identifier, expected only letters, digits and _ (%r)" % s
            )


@dataclasses.dataclass
class TableMeta:
    sql_for_table: dict[Table, str]
    column_names: dict[Table, list[str]]


def make_sql(table_expr: TableExpr, table_meta) -> str:
    if isinstance(table_expr, Table):
        return table_meta.sql_for_table[table_expr]
    elif isinstance(table_expr, Select):
        src = make_sql(table_expr.table, table_meta)
        exprs = ",".join(
            named_expr.expr + " as " + named_expr.name
            for named_expr in table_expr.exprs
        )
        return f"select {exprs}, _ivm_count from {src} as tbl where {table_expr.condition_expr}"
    else:
        raise Exception("unsupported table_expr (%r)" % table_expr)


def union_with_counts(a : Union[str, EmptyTable], b : Union[str, EmptyTable], names, multiplier=1):
    if isinstance(a, EmptyTable): return b
    if isinstance(b, EmptyTable): return a

    name_exprs = ", ".join(names)
    name_join = ", ".join(f"left.{name} = right.{name}" for name in names)
    return f"""
select {name_exprs}, coalesce(left._ivm_count, 0) + coalesce(right._ivm_count, 0)*{multiplier} 
from {a} as left full join {b} as right
on {name_join}
where coalesce(left._ivm_count, 0) + coalesce(right._ivm_count, 0) <> 0"""

def diff_with_counts(a, b, names):
    return union_with_counts(a, b, names, -1)

def union_with_counts_3(a, b, c, names):
    return union_with_counts(union_with_counts(a, b, names), c, names)

def make_derivative_sql(
    table_expr: TableExpr, table_var: Optional[Table], table_meta
) -> Union[str, EmptyTable]:
    if isinstance(table_expr, Table):
        if table_expr == table_var:
            return prefix_table_and_quote(prefix="_ivm_diff", name=table_expr.name)
        else:
            return EmptyTable()
    elif isinstance(table_expr, Select):
        src = make_derivative_sql(table_expr.table, table_var, table_meta)
        if isinstance(src, EmptyTable):
            return EmptyTable()

        exprs = ",".join(
            named_expr.expr + " as " + named_expr.name
            for named_expr in table_expr.exprs
        )
        return f"select {exprs}, _ivm_count from {src} as tbl where {table_expr.condition_expr}"
    elif isinstance(table_expr, InnerJoin):
        # ((A+dA) x (B+dB)) = A x B + A x dB + dA x B + dA x dB 
        left_diff = make_derivative_sql(table_expr.left.expr, table_var, table_meta)
        left = make_sql(table_expr.left.expr, table_meta)
        right_diff = make_derivative_sql(table_expr.right.expr, table_var, table_meta)
        right = make_sql(table_expr.right.expr, table_meta)
        keys = ", ".join(table_column_names(table_expr, table_meta))

        def make_one_side(left_sql, right_sql):
            return f"""
select {keys}, {table_expr.left.name}._ivm_count * {table_expr.right.name}._ivm_count as _ivm_count
from {left_sql} as {table_expr.left.name} join {right_sql} as {table_expr.right.name}
on {table_expr.condition}
"""

        return union_with_counts_3(
            make_one_side(left_diff, right),
            make_one_side(left, right_diff),
            make_one_side(left_diff, right_diff),
            table_column_names(table_expr, table_meta),
        )
    else:
        raise Exception("unsupported table_expr (%r)" % table_expr)


@test_case
def derivative_sql_cases():
    table_meta = TableMeta(
        sql_for_table={Table("tbl1"): "tbl1", Table("tbl2"): "tbl2"},
        column_names={
            Table("tbl1"): ["col11", "col12"],
            Table("tbl2"): ["col21", "col22"],
        },
    )
    table_var = Table("tbl1")
    print(make_derivative_sql(Table("tbl1"), table_var, table_meta))
    expect(r"""_ivm_diff_tbl1 """)
    print(make_derivative_sql(Table("tbl2"), table_var, table_meta))
    expect(r"""EmptyTable() """)
    print(
        make_derivative_sql(
            Select(
                table=Table("tbl1"),
                exprs=[WithAlias(name="x", expr="col11+1")],
                condition_expr="foo(col11)",
            ),
            table_var,
            table_meta,
        )
    )
    expect(
        r"""select col11+1 as x, _ivm_count from _ivm_diff_tbl1 as tbl where foo(col11) """
    )


def table_column_names(table_expr: TableExpr, table_meta: TableMeta):
    if isinstance(table_expr, Select):
        return [e.name for e in table_expr.exprs]
    if isinstance(table_expr, Table):
        return table_meta.column_names[table_expr]
    else:
        raise TypeError("unsupported expression %r" % table_expr)


def make_update_stmt(
    table_expr: TopLevelTableExpr,
    table_var: Optional[Table],
    table_meta: TableMeta,
    target_table: str,
):
    if isinstance(table_expr, GroupBy):
        updated_keys = "..."
        raise Exception("not impl")
    #         return f'''
    # with relevant_rows as materialized (
    #   select {all_table_fields}
    #   from {table} as _ivm_tbl
    #   join {table_diff} as _ivm_tbl_diff
    #   on {group_by_key_as_join_cond}
    #   where _ivm_tbl_diff._ivm_count <> 0
    # ),
    # diff_keys as materialized (
    #   select distinct {group_by_keys} from table_diff
    # ),
    # deletions as (
    #   delete from {target_table}
    #   where ({group_by_keys}) in diff_keys
    # ),
    # new_grouped_rows as (
    #   select {group_by_keys}, {group_values} from {table}
    # )
    # insert into {target_table}
    # select {group_by_keys}
    # from new_grouped_rows
    # '''
    else:
        keys = ", ".join(table_column_names(table_expr, table_meta))
        table_diff = make_derivative_sql(table_expr, table_var, table_meta)
        if isinstance(table_diff, EmptyTable):
            return ""
        return f"""
with updated as (insert into {target_table}
select * from {table_diff} _ivm_diff
on conflict ({keys}) 
do update set _ivm_count = {target_table}._ivm_count + EXCLUDED._ivm_count
returning *)
delete from {target_table} where ({keys}) in (select {keys} from updated where _ivm_count = 0)
"""


def prepare(conn, slot_name):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_create_logical_replication_slot(%s, 'wal2json')", (slot_name,)
        )
        cur.execute("CREATE TABLE ivm_snapshot_{} (id text);".format(slot_name))
        cur.execute("CREATE TABLE ivm_live_queries_{} (table_name text, query text);".format(slot_name))


def prepare_if_needed(conn, slot_name):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT EXISTS (
   SELECT FROM information_schema.tables 
   WHERE  table_schema = 'public'
   AND    table_name   = %s
   )""",
            ("ivm_snapshot_" + slot_name,),
        )
        [(exists,)] = cur

    if not exists:
        print("creating replication slot")
        prepare(conn, slot_name)

def lsn_to_int(x):
    a,b = x.split('/')
    return (int(a, 16)<<32) + int(b, 16)
        
def all_used_tables(t) -> set[Table]:
    r = set()
    def f(t):
        if isinstance(t, Table):
            r.add(t)
    walk_table_expr(t, f)
    return r

def quote_name(x):
    return '"%s"' % (x.replace('"', '""'))

def prefix_table_and_quote(name, prefix):
    schema, name = split_schema(name)
    return quote_name(schema) + '.' + quote_name(prefix + '_' + name)

def split_schema(table_name : str):
    if '.' not in table_name:
        raise Exception('table name is missing schema (%r)' % table_name)
    return table_name.split('.', 1)

def quote_name_with_schema(table_name):
    schema, name = split_schema(table_name)
    return quote_name(schema) + '.' + quote_name(name)

def select_from_table_with_one_count(table_name : str, column_names):
    exprs = ", ".join(column_names)

    return f"select {exprs}, 1 as _ivm_count from {quote_name_with_schema(table_name)}"

def handle_query(slot_name, cur, target_table_name, query):
    query_str = str(query)
    cur.execute(f'select from ivm_live_queries_{slot_name} where table_name = %s and query = %s', (target_table_name, query_str))
    already_exists = list(cur)

    dependency_tables = all_used_tables(query)
    
    dependency_column_names: dict[Table, list[str]] = {}
    for table in dependency_tables:
        cur.execute(f'select column_name from information_schema.columns where table_schema = %s and table_name = %s', split_schema(table.name))
        dependency_column_names[table] = list( name for (name,) in cur )
        
    if not already_exists:
        cur.execute(f'drop table if exists {quote_name_with_schema(target_table_name)}')

        table_meta = TableMeta(
            sql_for_table={
                table: select_from_table_with_one_count(table, dependency_column_names[table])
                for table in dependency_tables },
            column_names=dependency_column_names)
        
        sql = make_sql(query, table_meta)
        print('creating initial table', target_table_name)
        cur.execute(f'create table {quote_name_with_schema(target_table_name)} as {sql}')
        column_names = ', '.join(table_column_names(query, table_meta))
        cur.execute(f'create unique index on {quote_name_with_schema(target_table_name)} ({column_names}) include (_ivm_count)')
        cur.execute(f'insert into ivm_live_queries_{slot_name} values (%s, %s)', (target_table_name, query_str))
    else:
        already_processed_dependencies = set()

        for dependency_name in dependency_tables:
            # - act as if deltas where applied to tables one by one
            # - where applying changes to current table, we need to pass original version to the derivative
            sql_for_table: dict[Table, str] = {}

            for table in dependency_tables:
                sql = select_from_table_with_one_count(
                    table.name, dependency_column_names[table])

                if table not in already_processed_dependencies:
                    sql = diff_with_counts(
                        sql,
                        prefix_table_and_quote(prefix="_ivm_diff_", name=table.name),
                        dependency_column_names[table])

                assert not isinstance(sql, EmptyTable)
                sql_for_table[table] = sql
            
            already_processed_dependencies.add(dependency_name)

            table_meta = TableMeta(
                sql_for_table={
                    
                },
                column_names=dependency_column_names)

            # derivative_sql = make_derivative_sql(query, dependency_name, table_meta)

            update_sql = make_update_stmt(
                query,
                dependency_name,
                table_meta,
                target_table_name,
            )
            print(update_sql)
            
        raise Exception('diff!')
        
def main(dsn, slot_name, queries : dict[str, TopLevelTableExpr]):
    watched_tables = set()
    for q in queries.values(): watched_tables |= all_used_tables(q)

    def callback():
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                for table_name, query in queries.items():
                    handle_query(slot_name, cur, table_name, query)

    replication_loop(dsn, slot_name, watched_tables, callback)

def replication_loop(dsn, slot_name, watched_tables, callback):
    assert_valid_id(slot_name)
    with psycopg2.connect(dsn) as conn:
        prepare_if_needed(conn, slot_name)

    with psycopg2.connect(
        dsn, connection_factory=LogicalReplicationConnection
    ) as replication_conn:
        # TODO: we should have loop here
        replication_cur = replication_conn.cursor()
        replication_cur.start_replication(
            slot_name=slot_name,
            options={
                "format-version": "2",
                "include-lsn": "1",
            },
        )

        # TODO: pass this to wal2json
        # ['public.ivm_snapshot_' + slot_name] + 
        for table in watched_tables:
            if '.' not in table.name:
                raise Exception('invalid table name %r (missing schema)' % table)
        
        already_created_tables = set()

        def create_table_if_needed(cur, table_name, columns):
            if table_name not in already_created_tables:
                columns_s = ', '.join( '%s %s' % (quote_name(col['name']), quote_name(col['type'])) for col in columns )
                cur.execute('create table if not exists %s (%s, _ivm_count bigint)' % (prefix_table_and_quote(table_name, '_ivm_diff'), columns_s))

        def insert_row(table_name, columns, count_diff):
            # todo: batch and use COPY
            table_name = prefix_table_and_quote(table_name, '_ivm_diff')
            column_names = ', '.join( quote_name(col['name']) for col in columns )
            placeholders = ', '.join( '%s' for col in columns )
            # prepend '\x' to bytea here, TODO: other types as well?
            cur.execute(f'insert into {table_name} ({column_names}, _ivm_count) values ({placeholders}, %s)',
                        tuple( ('\\x' + col['value'] if col['type'] == 'bytea' else col['value']) for col in columns ) + (count_diff,))

        final_lsn = None
            
        with psycopg2.connect(dsn) as conn:
            with psycopg2.connect(dsn) as snapshot_conn:
                with snapshot_conn.cursor() as cur:
                    cur.execute("SELECT pg_export_snapshot()")
                    [(snapshot_id,)] = cur
                    cur.execute(f"DELETE FROM ivm_snapshot_{slot_name}")
                    cur.execute(
                        f"INSERT INTO ivm_snapshot_{slot_name} VALUES (%s)",
                        (snapshot_id,),
                    )
                with conn.cursor() as cur:
                    cur.execute("SET TRANSACTION SNAPSHOT %s", (snapshot_id,))

            with conn.cursor() as cur:
                def on_msg(msg):
                    msg = json.loads(msg.payload)
                    # print(msg)

                    if msg['action'] == 'I' or msg['action'] == 'D':
                        count_diff = 1 if msg['action'] == 'I' else -1
                        table_name = msg['schema'] + '.' + msg['table']

                        # only works for REPLICA IDENTITY FULL 
                        columns = msg['columns'] if msg['action'] == 'I' else msg['identity']
                        if table_name in watched_tables:
                            create_table_if_needed(cur, table_name, columns)
                            insert_row(table_name, columns, count_diff)
                    
                    if (
                        msg["action"] == "I"
                        and msg["schema"] == "public"
                        and msg["table"] == "ivm_snapshot_" + slot_name
                        and msg["columns"][0]["value"] == snapshot_id
                    ):
                        print("caught up with replication")
                        nonlocal final_lsn
                        final_lsn = msg['lsn']
                        raise StopReplication()
                    

                try:
                    replication_cur.consume_stream(on_msg)
                except StopReplication:
                    pass

        replication_cur.send_feedback(reply=True, force=True, flush_lsn=lsn_to_int(final_lsn))
        callback()

    # TODO: run analyze on temporary tables


# if data['action'] == 'I' and data['schema'] == 'public' and data['table'] == 'ivm_snapshot_' + slot_name:
#                     print(data['columns'][0]['value'], snapshot_id)
