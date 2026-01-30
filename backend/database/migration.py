from .db import init_db

def migrate():
    print("Running DB Migration...")
    init_db()
    print("Migration Completed")

if __name__ == "__main__":
    migrate()
