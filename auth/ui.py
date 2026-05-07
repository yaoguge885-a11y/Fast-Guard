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
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0c0a09, stop:0.5 #1c1917, stop:1 #292524);
                border-radius: 24px;
                border: 2px solid #44403c;
            }
            QLabel {
                color: #ffffff;
                background: transparent;
            }
            QLineEdit {
                background-color: #111111;
                border: 1px solid #3f3f46;
                border-radius: 8px;
                padding: 24px 28px;
                color: #ffffff;
                font-size: 42px;
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
                spacing: 12px;
                background: transparent;
                font-size: 28px;
            }
            QCheckBox::indicator {
                width: 32px;
                height: 32px;
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
                padding: 20px 28px;
                font-size: 36px;
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
        self.exit_program = False  # 退出系统标志
        self.setWindowTitle("FastGuard 登录与注册")
        self.setFixedSize(900, 1000)
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
        logo_label = QtWidgets.QLabel("🛡️")
        logo_label.setStyleSheet("font-size: 120px; color: #1e40af; margin-bottom: 0px;")
        logo_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(logo_label)

        title = QtWidgets.QLabel("FastGuard")
        title.setStyleSheet("font-size: 56px; font-weight: 900; letter-spacing: 2px;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel("欢迎回来，请登录您的账号")
        subtitle.setStyleSheet("color: #a1a1aa; font-size: 28px; margin-bottom: 10px;")
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

        self.checkbox_remember_password = QtWidgets.QCheckBox("记住密码")
        self.checkbox_remember_password.setStyleSheet("font-size: 28px;")
        remember_layout = QtWidgets.QHBoxLayout()
        remember_layout.setSpacing(16)
        remember_layout.addStretch()
        remember_layout.addWidget(self.checkbox_remember_password)
        remember_layout.addStretch()
        layout.addLayout(remember_layout)

        layout.addSpacing(20)

        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setSpacing(12)

        self.btn_login = QtWidgets.QPushButton("登 录")
        self.btn_login.setObjectName("primaryBtn")
        self.btn_login.setFixedHeight(80)
        
        self.btn_cancel_login = QtWidgets.QPushButton("退 出 系 统")
        self.btn_cancel_login.setObjectName("btnCancel")
        self.btn_cancel_login.setFixedHeight(80)
        
        self.btn_go_register = QtWidgets.QPushButton("没有账号？立即注册 ➔")
        self.btn_go_register.setObjectName("switchBtn")
        self.btn_go_register.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        
        btn_layout.addWidget(self.btn_login)
        btn_layout.addWidget(self.btn_cancel_login)
        
        layout.addLayout(btn_layout)
        layout.addSpacing(8)
        layout.addWidget(self.btn_go_register, alignment=QtCore.Qt.AlignCenter)

        self.btn_login.clicked.connect(self.handle_login)
        self.btn_cancel_login.clicked.connect(self.handle_exit)
        self.btn_go_register.clicked.connect(self.switch_to_register)



    def _setup_register_page(self):
        self.register_page = QtWidgets.QWidget(self.stack)
        layout = QtWidgets.QVBoxLayout(self.register_page)
        layout.setContentsMargins(45, 40, 45, 40)
        layout.setSpacing(14)

        # Logo and Title
        logo_label = QtWidgets.QLabel("🛡️")
        logo_label.setStyleSheet("font-size: 120px; color: #1e40af; margin-bottom: 0px;")
        logo_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(logo_label)

        title = QtWidgets.QLabel("创建新账号")
        title.setStyleSheet("font-size: 54px; font-weight: 800; letter-spacing: 2px;")
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
        hint.setStyleSheet("color: #9ca3af; font-size: 24px;")
        hint.setAlignment(QtCore.Qt.AlignLeft)
        layout.addWidget(hint)

        layout.addStretch()

        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setSpacing(12)

        self.btn_create = QtWidgets.QPushButton("立 即 注 册")
        self.btn_create.setObjectName("regPrimaryBtn")
        self.btn_create.setFixedHeight(80)
        
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
            remember_password = bool(data.get("remember_password"))
            if remember_password:
                self.checkbox_remember_password.setChecked(True)
                if data.get("password"):
                    self.input_pass.setText(str(data.get("password")))
            else:
                self.input_pass.clear()
            if remember_password and data.get("password"):
                self._try_auto_login()
        except Exception:
            self._remember_data = {}

    def _save_remember(
        self,
        username: str,
        password: Optional[str] = None,
        remember_password: bool = False
    ):
        data = {
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
        # 此方法已不再需要，因为移除了保持登录状态按钮
        pass

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
        
        # 设置样式，确保文字清晰可见
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #ffffff;
                color: #000000;
                min-width: 800px;
                min-height: 350px;
            }
            QMessageBox QLabel {
                color: #000000;
                font-size: 32px;
                font-family: 'Microsoft YaHei';
                qproperty-alignment: 'AlignCenter';
                padding: 30px;
            }
            QPushButton {
                background-color: #f0f0f0;
                color: #000000;
                border: 2px solid #cccccc;
                border-radius: 10px;
                padding: 20px 50px;
                font-size: 40px;
                font-weight: bold;
                min-width: 200px;
                min-height: 70px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        
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
            
            # 安全逻辑：管理员账户不保存密码，即使勾选了记住密码
            if self.checkbox_remember_password.isChecked():
                if role == "admin":
                    # 管理员账户：只保存用户名，不保存密码
                    self._save_remember(
                        username,
                        password=None,  # 不保存管理员密码
                        remember_password=False  # 强制设置为False
                    )
                    self._show_msg("info", "安全提示", "管理员账户已登录，出于安全考虑，密码不会被保存。")
                else:
                    # 普通用户：正常保存密码
                    self._save_remember(
                        username,
                        password=password,
                        remember_password=True
                    )
            else:
                self._clear_remember()

            self.accept()
        else:
            self._show_msg("critical", "登录失败", "用户名或密码错误")

    def handle_exit(self):
        """处理退出系统按钮点击"""
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("确认退出")
        msg_box.setText("确定要退出FastGuard系统吗？")
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msg_box.setDefaultButton(QtWidgets.QMessageBox.No)
    
        # 设置样式，覆盖全局样式
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #ffffff;
                color: #000000;
                min-width: 800px;
                min-height: 350px;
            }
            QMessageBox QLabel {
                color: #000000;
                font-size: 32px;
                font-family: 'Microsoft YaHei';
                qproperty-alignment: 'AlignCenter';
                padding: 30px;
            }
            QPushButton {
                background-color: #f0f0f0;
                color: #000000;
                border: 2px solid #cccccc;
                border-radius: 10px;
                padding: 20px 50px;
                font-size: 40px;
                font-weight: bold;
                min-width: 200px;
                min-height: 70px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
    
        reply = msg_box.exec_()
    
        if reply == QtWidgets.QMessageBox.Yes:
            # 设置退出标志
            self.exit_program = True
            self.reject()  # 关闭登录窗口

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