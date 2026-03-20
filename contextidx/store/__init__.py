from contextidx.store.base import Store
from contextidx.store.sqlite_store import SQLiteStore
from contextidx.store.backend_metadata_store import BackendMetadataStore

__all__ = ["Store", "SQLiteStore", "BackendMetadataStore"]

try:
    from contextidx.store.postgres_store import PostgresStore
    __all__.append("PostgresStore")
except ImportError:
    pass

try:
    from contextidx.store.migrations import (
        MigrationReport,
        migrate_sqlite_to_postgres,
        validate_migration,
    )
    __all__.extend(["migrate_sqlite_to_postgres", "validate_migration", "MigrationReport"])
except ImportError:
    pass
