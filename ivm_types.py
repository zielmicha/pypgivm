import dataclasses
from typing import Union, TypeVar, Generic

T = TypeVar('T')
            
@dataclasses.dataclass(frozen=True)
class WithAlias(Generic[T]):
    expr: T
    name: str
            
@dataclasses.dataclass(frozen=True)
class Select:
    table: 'TableExpr'
    condition_expr: str
    exprs: list[WithAlias[str]] 

@dataclasses.dataclass(frozen=True)
class Table:
    name: str

        
    
@dataclasses.dataclass(frozen=True)
class InnerJoin:
    left: WithAlias['TableExpr']
    right: WithAlias['TableExpr']
    condition: str

@dataclasses.dataclass(frozen=True)
class AggregationFunc:
    name: str
    expr: str

@dataclasses.dataclass(frozen=True)
class CountFunc:
    pass
    
@dataclasses.dataclass(frozen=True)
class GroupBy:
    group_by: list[str]
    table: 'TableExpr'
    exprs: list[WithAlias[Union[str, AggregationFunc, CountFunc]]]

TableExpr = Union[Select, Table, InnerJoin]
TopLevelTableExpr = Union[TableExpr, GroupBy]

def walk_table_expr(e : TopLevelTableExpr, f):
    f(e)
    if isinstance(e, Select):
        walk_table_expr(e.table, f)
    elif isinstance(e, InnerJoin):
        walk_table_expr(e.left.expr, f)
        walk_table_expr(e.right.expr, f)
    elif isinstance(e, GroupBy):
        walk_table_expr(e.table, f)
    elif isinstance(e, Table):
        pass
    else:
        raise Exception('unknown type %r' % e)
