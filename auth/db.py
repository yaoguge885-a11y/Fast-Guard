import os
import time
import sqlite3
import hashlib


class UserDB:
    def __init__(self, db_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = db_path or os.path.join(data_dir, "users.db")
        self._init_db()

    def _hash_password(self, password: str) -> str:
        salt = "FastGuard@2026"
        return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        self.ensure_admin()

    def ensure_admin(self):
        admin_user = "admin"
        admin_pass = "Admin123"
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT username FROM users WHERE username=?", (admin_user,))
            if cur.fetchone() is None:
                cur.execute(
                    "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                    (admin_user, self._hash_password(admin_pass), "admin", time.strftime("%Y-%m-%d %H:%M:%S"))
                )
                conn.commit()

    def verify_user(self, username: str, password: str):
        pwd_hash = self._hash_password(password)
        return self.verify_user_hash(username, pwd_hash)

    def verify_user_hash(self, username: str, password_hash: str):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT role FROM users WHERE username=? AND password_hash=?", (username, password_hash))
            row = cur.fetchone()
            return row[0] if row else None


    def user_exists(self, username: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM users WHERE username=?", (username,))
            return cur.fetchone() is not None

    def create_user(self, username: str, password: str, role: str = "user") -> bool:
        if self.user_exists(username):
            return False
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (username, self._hash_password(password), role, time.strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
        return True

    def list_users(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT username, role, created_at FROM users ORDER BY created_at DESC")
            return cur.fetchall()

    def delete_user(self, username: str) -> bool:
        if username == "admin":
            return False
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM users WHERE username=?", (username,))
            conn.commit()
            return cur.rowcount > 0


class LogDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    category TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add_log(self, username: str, level: str, message: str, category: str = "system"):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO logs (username, level, message, created_at, category) VALUES (?, ?, ?, ?, ?)",
                (username, level, message, time.strftime("%Y-%m-%d %H:%M:%S"), category)
            )
            conn.commit()

    def list_logs(self, username: str = None, limit: int = 500):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            if username:
                cur.execute(
                    "SELECT id, username, level, message, created_at, category FROM logs WHERE username=? ORDER BY id DESC LIMIT ?",
                    (username, limit)
                )
            else:
                cur.execute(
                    "SELECT id, username, level, message, created_at, category FROM logs ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
            return cur.fetchall()

    def clear_logs(self, username: str = None):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            if username:
                cur.execute("DELETE FROM logs WHERE username=?", (username,))
            else:
                cur.execute("DELETE FROM logs")
            conn.commit()

    def delete_logs_for_user(self, username: str):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM logs WHERE username=?", (username,))
            conn.commit()

