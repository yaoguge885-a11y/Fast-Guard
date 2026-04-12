import os
import json
from typing import Optional
from PyQt5 import QtWidgets, QtCore, QtGui


from .db import UserDB


class LoginDialog(QtWidgets.QDialog):
    def _apply_theme(self):
        self.setStyleSheet(
            """
            QFrame#dialogBg {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #09090b, stop:0.5 #18181b, stop:1 #27272a);
                border-radius: 24px;
                border: 2px solid #3f3f46;
            }
            QLabel {
                color: #ffffff;
                background: transparent;
            }
            QLineEdit {
                background-color: #111111;
                border: 1px solid #3f3f46;
                border-radius: 8px;
                padding: 12px 14px;
                color: #ffffff;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #6366f1;
                background-color: #18181b;
            }
            QLineEdit#regInput:focus {
                border: 1px solid #10b981;
            }
            QCheckBox {
                color: #a1a1aa;
                spacing: 8px;
                background: transparent;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #3f3f46;
                background-color: #111111;
            }
            QCheckBox::indicator:checked {
                background-color: #6366f1;
                border: 1px solid #6366f1;
            }
            QPushButton {
                background-color: #27272a;
                color: #ffffff;
                border: 1px solid #3f3f46;
                border-radius: 8px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3f3f46;
                border: 1px solid #6366f1;
            }
            QPushButton#btnCancel:hover {
                border: 1px solid #ef4444;
            }
            QPushButton:pressed {
                background-color: #18181b;
            }
            QPushButton#primaryBtn {
                background-color: #6366f1;
                border: none;
            }
            QPushButton#primaryBtn:hover {
                background-color: #4f46e5;
            }
            QPushButton#regPrimaryBtn {
                background-color: #10b981;
                border: none;
            }
            QPushButton#regPrimaryBtn:hover {
                background-color: #059669;
            }
            QPushButton#switchBtn {
                background-color: transparent;
                border: none;
                color: #a1a1aa;
                font-weight: normal;
                text-decoration: underline;
                padding: 4px;
            }
            QPushButton#switchBtn:hover {
                color: #ffffff;
            }
            """
        )

    def __init__(self, user_db: UserDB, parent=None):
        super().__init__(parent)
        self.user_db = user_db
        self.username = ""
        self.role = ""
        self.setWindowTitle("FastGuard 登录与注册")
        self.setFixedSize(480, 560)
        self.setModal(True)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self._remember_path = self._get_remember_path()
        self._remember_data = {}

        # 居中显示
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.bg_frame = QtWidgets.QFrame()
        self.bg_frame.setObjectName("dialogBg")
        main_layout.addWidget(self.bg_frame)

        self._apply_theme()

        self.stack = QtWidgets.QStackedWidget(self.bg_frame)
        stack_layout = QtWidgets.QVBoxLayout(self.bg_frame)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        stack_layout.addWidget(self.stack)

        self._setup_login_page()
        self._setup_register_page()

        self.stack.addWidget(self.login_page)
        self.stack.addWidget(self.register_page)
        self.stack.setCurrentWidget(self.login_page)

        self._load_remember()

    def _setup_login_page(self):
        self.login_page = QtWidgets.QWidget(self.stack)
        layout = QtWidgets.QVBoxLayout(self.login_page)
        layout.setContentsMargins(45, 50, 45, 40)
        layout.setSpacing(16)

        # Logo and Title
        logo_label = QtWidgets.QLabel("❖")
        logo_label.setStyleSheet("font-size: 64px; color: #6366f1; margin-bottom: 0px;")
        logo_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(logo_label)

        title = QtWidgets.QLabel("FastGuard")
        title.setStyleSheet("font-size: 28px; font-weight: 900; letter-spacing: 2px;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel("欢迎回来，请登录您的账号")
        subtitle.setStyleSheet("color: #a1a1aa; font-size: 14px; margin-bottom: 10px;")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(subtitle)

        form = QtWidgets.QVBoxLayout()
        form.setSpacing(14)

        self.input_user = QtWidgets.QLineEdit()
        self.input_user.setPlaceholderText("用户名")
        self.input_pass = QtWidgets.QLineEdit()
        self.input_pass.setPlaceholderText("密码")
        self.input_pass.setEchoMode(QtWidgets.QLineEdit.Password)

        form.addWidget(self.input_user)
        form.addWidget(self.input_pass)
        layout.addLayout(form)

        self.checkbox_remember = QtWidgets.QCheckBox("保持登录状态")
        self.checkbox_remember_password = QtWidgets.QCheckBox("记住密码")
        remember_layout = QtWidgets.QHBoxLayout()
        remember_layout.setSpacing(16)
        remember_layout.addStretch()
        remember_layout.addWidget(self.checkbox_remember)
        remember_layout.addWidget(self.checkbox_remember_password)
        remember_layout.addStretch()
        layout.addLayout(remember_layout)


        layout.addStretch()

        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setSpacing(12)

        self.btn_login = QtWidgets.QPushButton("登 录")
        self.btn_login.setObjectName("primaryBtn")
        self.btn_login.setFixedHeight(46)
        
        self.btn_cancel_login = QtWidgets.QPushButton("退 出 系 统")
        self.btn_cancel_login.setObjectName("btnCancel")
        self.btn_cancel_login.setFixedHeight(46)
        
        self.btn_go_register = QtWidgets.QPushButton("没有账号？立即注册 ➔")
        self.btn_go_register.setObjectName("switchBtn")
        self.btn_go_register.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        
        btn_layout.addWidget(self.btn_login)
        btn_layout.addWidget(self.btn_cancel_login)
        
        layout.addLayout(btn_layout)
        layout.addSpacing(8)
        layout.addWidget(self.btn_go_register, alignment=QtCore.Qt.AlignCenter)

        self.btn_login.clicked.connect(self.handle_login)
        self.btn_cancel_login.clicked.connect(self.reject)
        self.btn_go_register.clicked.connect(self.switch_to_register)
        self.checkbox_remember_password.toggled.connect(self._on_remember_password_toggled)



    def _setup_register_page(self):
        self.register_page = QtWidgets.QWidget(self.stack)
        layout = QtWidgets.QVBoxLayout(self.register_page)
        layout.setContentsMargins(45, 40, 45, 40)
        layout.setSpacing(14)

        # Logo and Title
        logo_label = QtWidgets.QLabel("✧")
        logo_label.setStyleSheet("font-size: 64px; color: #10b981; margin-bottom: 0px;")
        logo_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(logo_label)

        title = QtWidgets.QLabel("创建新账号")
        title.setStyleSheet("font-size: 26px; font-weight: 800; letter-spacing: 2px;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        form = QtWidgets.QVBoxLayout()
        form.setSpacing(14)

        self.reg_input_user = QtWidgets.QLineEdit()
        self.reg_input_user.setObjectName("regInput")
        self.reg_input_user.setPlaceholderText("用户名 (至少3位)")
        
        self.reg_input_pass = QtWidgets.QLineEdit()
        self.reg_input_pass.setObjectName("regInput")
        self.reg_input_pass.setPlaceholderText("密码 (至少6位)")
        self.reg_input_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        
        self.reg_input_confirm = QtWidgets.QLineEdit()
        self.reg_input_confirm.setObjectName("regInput")
        self.reg_input_confirm.setPlaceholderText("确认密码")
        self.reg_input_confirm.setEchoMode(QtWidgets.QLineEdit.Password)

        form.addWidget(self.reg_input_user)
        form.addWidget(self.reg_input_pass)
        form.addWidget(self.reg_input_confirm)
        layout.addLayout(form)

        hint = QtWidgets.QLabel("提示：建议使用字母+数字组合。")
        hint.setStyleSheet("color: #9ca3af; font-size: 12px;")
        hint.setAlignment(QtCore.Qt.AlignLeft)
        layout.addWidget(hint)

        layout.addStretch()

        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setSpacing(12)

        self.btn_create = QtWidgets.QPushButton("立 即 注 册")
        self.btn_create.setObjectName("regPrimaryBtn")
        self.btn_create.setFixedHeight(46)
        
        self.btn_go_login = QtWidgets.QPushButton("⬅ 返回登录")
        self.btn_go_login.setObjectName("switchBtn")
        self.btn_go_login.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        
        btn_layout.addWidget(self.btn_create)
        
        layout.addLayout(btn_layout)
        layout.addSpacing(8)
        layout.addWidget(self.btn_go_login, alignment=QtCore.Qt.AlignCenter)

        self.btn_create.clicked.connect(self.handle_create)
        self.btn_go_login.clicked.connect(self.switch_to_login)

    def switch_to_register(self):
        self.reg_input_user.clear()
        self.reg_input_pass.clear()
        self.reg_input_confirm.clear()

        self.stack.setCurrentWidget(self.register_page)

    def switch_to_login(self):
        self.stack.setCurrentWidget(self.login_page)

    def _get_remember_path(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, "remember.json")

    def _load_remember(self):
        if not os.path.exists(self._remember_path):
            return
        try:
            with open(self._remember_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            self._remember_data = data
            if data.get("username"):
                self.input_user.setText(str(data.get("username")))
            remember = bool(data.get("remember"))
            remember_password = bool(data.get("remember_password"))
            if remember:
                self.checkbox_remember.setChecked(True)
            if remember_password:
                self.checkbox_remember_password.setChecked(True)
                if data.get("password"):
                    self.input_pass.setText(str(data.get("password")))
            else:
                self.input_pass.clear()
            if remember and remember_password and data.get("password"):
                self._try_auto_login()
        except Exception:
            self._remember_data = {}

    def _save_remember(
        self,
        username: str,
        remember: bool = False,
        password: Optional[str] = None,
        remember_password: bool = False
    ):
        data = {
            "remember": bool(remember),
            "username": username,
            "remember_password": bool(remember_password)
        }
        if remember_password and password:
            data["password"] = password
        with open(self._remember_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)



    def _clear_remember(self):
        if os.path.exists(self._remember_path):
            try:
                os.remove(self._remember_path)
            except Exception:
                pass

    def _on_remember_password_toggled(self, checked: bool):
        if checked and not self.checkbox_remember.isChecked():
            self.checkbox_remember.setChecked(True)

    def _try_auto_login(self):
        username = self.input_user.text().strip()
        password = self.input_pass.text().strip()
        if not username or not password:
            return
        role = self.user_db.verify_user(username, password)
        if role:
            self.username = username
            self.role = role
            self.accept()

    def _show_msg(self, msg_type, title, text):
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        if msg_type == "warning":
            msg.setIcon(QtWidgets.QMessageBox.Warning)
        elif msg_type == "info":
            msg.setIcon(QtWidgets.QMessageBox.Information)
        elif msg_type == "critical":
            msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.WindowStaysOnTopHint)
        msg.exec_()


    def handle_login(self):
        username = self.input_user.text().strip()
        password = self.input_pass.text().strip()
        if not username or not password:
            self._show_msg("warning", "提示", "请输入用户名和密码")
            return
        role = self.user_db.verify_user(username, password)
        if role:
            self.username = username
            self.role = role
            if self.checkbox_remember.isChecked() or self.checkbox_remember_password.isChecked():
                self._save_remember(
                    username,
                    remember=self.checkbox_remember.isChecked(),
                    password=password if self.checkbox_remember_password.isChecked() else None,
                    remember_password=self.checkbox_remember_password.isChecked()
                )
            else:
                self._clear_remember()


            self.accept()
        else:
            self._show_msg("critical", "登录失败", "用户名或密码错误")

    def handle_create(self):
        username = self.reg_input_user.text().strip()
        password = self.reg_input_pass.text().strip()
        confirm = self.reg_input_confirm.text().strip()
        
        if len(username) < 3 or len(password) < 6:
            self._show_msg("warning", "提示", "用户名至少3位，密码至少6位")
            return
        if password != confirm:
            self._show_msg("warning", "提示", "两次输入的密码不一致")
            return
        if self.user_db.user_exists(username):
            self._show_msg("warning", "提示", "用户名已存在")
            return
            
        if self.user_db.create_user(username, password):
            self._show_msg("info", "成功", "注册成功，请登录")
            self.input_user.setText(username)
            self.input_pass.clear()
            self.switch_to_login()
        else:
            self._show_msg("critical", "失败", "注册失败，请重试")
