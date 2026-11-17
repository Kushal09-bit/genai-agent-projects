from dataclasses import dataclass, field
from sqlalchemy import create_engine, inspect, text

@dataclass
class ColumnMeta:
    name: str
    type: str
    sample_values: list = field(default_factory=list)

@dataclass
class TableMeta:
    name: str
    row_count: int
    columns: list
    foreign_keys: list

class MetadataExtractor:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.inspector = inspect(self.engine)

    def extract_all(self) -> dict:
        tables = {}
        for table_name in self.inspector.get_table_names(schema="public"):
            tables[table_name] = self._extract_table(table_name)
        return tables

    def _extract_table(self, table_name: str) -> TableMeta:
        columns = []
        for col in self.inspector.get_columns(table_name):
            col_type = str(col["type"])
            samples = []
            if any(t in col_type.lower() for t in ["char", "text", "bool"]):
                samples = self._sample_values(table_name, col["name"])
            columns.append(ColumnMeta(name=col["name"], type=col_type, sample_values=samples))

        row_count = self._row_count(table_name)
        fks = self._foreign_keys(table_name)
        return TableMeta(name=table_name, row_count=row_count, columns=columns, foreign_keys=fks)

    def _row_count(self, table_name: str) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
            return result.scalar()

    def _sample_values(self, table_name: str, column_name: str, limit: int = 5) -> list:
        with self.engine.connect() as conn:
            query = text(
                f'SELECT DISTINCT "{column_name}" FROM "{table_name}" '
                f'WHERE "{column_name}" IS NOT NULL LIMIT :limit'
            )
            result = conn.execute(query, {"limit": limit})
            return [row[0] for row in result]

    def _foreign_keys(self, table_name: str) -> list:
        fks = []
        for fk in self.inspector.get_foreign_keys(table_name):
            fks.append({
                "column": fk["constrained_columns"],
                "references_table": fk["referred_table"],
                "references_column": fk["referred_columns"],
            })
        return fks

    def close(self):
        self.engine.dispose()


if __name__ == "__main__":
    extractor = MetadataExtractor("postgresql://postgres:postgres@localhost:5432/northwind")
    metadata = extractor.extract_all()

    for table_name, meta in metadata.items():
        print(f"\n{table_name} ({meta.row_count} rows)")
        for col in meta.columns:
            samples = f" samples={col.sample_values}" if col.sample_values else ""
            print(f"  - {col.name}: {col.type}{samples}")
        for fk in meta.foreign_keys:
            print(f"  FK: {fk['column']} -> {fk['references_table']}.{fk['references_column']}")

    extractor.close()