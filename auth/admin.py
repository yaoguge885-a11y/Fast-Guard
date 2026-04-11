from PyQt5 import QtWidgets, QtCore

from .db import UserDB, LogDB


class AdminPanel(QtWidgets.QDialog):
    def __init__(self, user_db: UserDB, log_db: LogDB, parent=None):
        super().__init__(parent)
        self.user_db = user_db
        self.log_db = log_db
        self.setWindowTitle("后台管理")
        self.resize(900, 600)
        self.setStyleSheet(
            """
            QDialog { background-color: #09090b; color: #e5e7eb; font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; }
            QLabel { color: #e5e7eb; }
            QTabWidget::pane { border: 1px solid #27272a; border-radius: 4px; }
            QTabBar::tab { background: #18181b; color: #a1a1aa; padding: 8px 16px; border: 1px solid #27272a; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #27272a; color: #ffffff; border-color: #3f3f46; }
            QHeaderView::section { background: #18181b; color: #e5e7eb; padding: 6px; border: 1px solid #27272a; font-weight: bold; }
            QTableWidget { background: #111111; color: #e5e7eb; gridline-color: #27272a; border: 1px solid #27272a; }
            QPushButton { background: #27272a; color: #ffffff; border: 1px solid #3f3f46; border-radius: 8px; padding: 6px 12px; }
            QPushButton:hover { background: #3f3f46; border: 1px solid #6366f1; }
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        # Users tab
        users_tab = QtWidgets.QWidget()
        users_layout = QtWidgets.QVBoxLayout(users_tab)

        self.users_table = QtWidgets.QTableWidget(0, 3)
        self.users_table.setHorizontalHeaderLabels(["用户名", "角色", "创建时间"])
        self.users_table.horizontalHeader().setStretchLastSection(True)
        self.users_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.users_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        users_layout.addWidget(self.users_table)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_refresh_users = QtWidgets.QPushButton("刷新用户")
        self.btn_delete_user = QtWidgets.QPushButton("删除用户")
        btn_row.addWidget(self.btn_refresh_users)
        btn_row.addWidget(self.btn_delete_user)
        btn_row.addStretch()
        users_layout.addLayout(btn_row)

        tabs.addTab(users_tab, "用户管理")

        # Logs tab
        logs_tab = QtWidgets.QWidget()
        logs_layout = QtWidgets.QVBoxLayout(logs_tab)

        self.logs_table = QtWidgets.QTableWidget(0, 5)
        self.logs_table.setHorizontalHeaderLabels(["时间", "用户", "等级", "类别", "内容"])
        self.logs_table.horizontalHeader().setStretchLastSection(True)
        self.logs_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.logs_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        logs_layout.addWidget(self.logs_table)

        log_btn_row = QtWidgets.QHBoxLayout()
        self.btn_refresh_logs = QtWidgets.QPushButton("刷新日志")
        self.btn_clear_logs = QtWidgets.QPushButton("清空日志")
        log_btn_row.addWidget(self.btn_refresh_logs)
        log_btn_row.addWidget(self.btn_clear_logs)
        log_btn_row.addStretch()
        logs_layout.addLayout(log_btn_row)

        tabs.addTab(logs_tab, "日志管理")

        self.btn_refresh_users.clicked.connect(self.load_users)
        self.btn_delete_user.clicked.connect(self.delete_user)
        self.btn_refresh_logs.clicked.connect(self.load_logs)
        self.btn_clear_logs.clicked.connect(self.clear_logs)

        self.load_users()
        self.load_logs()

    def load_users(self):
        users = self.user_db.list_users()
        self.users_table.setRowCount(0)
        for row in users:
            r = self.users_table.rowCount()
            self.users_table.insertRow(r)
            for c, val in enumerate(row):
                self.users_table.setItem(r, c, QtWidgets.QTableWidgetItem(str(val)))

    def delete_user(self):
        row = self.users_table.currentRow()
        if row < 0:
            return
        username = self.users_table.item(row, 0).text()
        if username == "admin":
            QtWidgets.QMessageBox.warning(self, "提示", "管理员账号不可删除")
            return
        if QtWidgets.QMessageBox.question(self, "确认", f"确认删除用户 {username} 吗？") != QtWidgets.QMessageBox.Yes:
            return
        if self.user_db.delete_user(username):
            self.log_db.delete_logs_for_user(username)
            self.load_users()
            self.load_logs()

    def load_logs(self):
        logs = self.log_db.list_logs(None, limit=500)
        self.logs_table.setRowCount(0)
        for _, username, level, message, created_at, category in logs:
            r = self.logs_table.rowCount()
            self.logs_table.insertRow(r)
            self.logs_table.setItem(r, 0, QtWidgets.QTableWidgetItem(created_at))
            self.logs_table.setItem(r, 1, QtWidgets.QTableWidgetItem(username))
            self.logs_table.setItem(r, 2, QtWidgets.QTableWidgetItem(level))
            self.logs_table.setItem(r, 3, QtWidgets.QTableWidgetItem(category))
            self.logs_table.setItem(r, 4, QtWidgets.QTableWidgetItem(message))

    def clear_logs(self):
        if QtWidgets.QMessageBox.question(self, "确认", "确认清空所有日志吗？") != QtWidgets.QMessageBox.Yes:
            return
        self.log_db.clear_logs()
        self.load_logs()
