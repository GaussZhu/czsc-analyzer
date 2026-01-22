document.addEventListener('DOMContentLoaded', function() {
    // 标签切换功能
    initTabs();
    
    // 密码可见性切换
    initPasswordToggles();
    
    // 验证码按钮功能
    initVerificationCodeButtons();
    
    // 表单验证
    initFormValidation();
    
    // 微信登录/注册按钮
    initWechatAuth();
});

// 标签切换功能
function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // 移除所有标签的活动状态
            tabs.forEach(t => t.classList.remove('active'));
            
            // 隐藏所有表单
            const forms = document.querySelectorAll('.login-form');
            forms.forEach(form => form.classList.remove('active'));
            
            // 激活当前标签和表单
            tab.classList.add('active');
            const tabId = tab.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// 密码可见性切换
function initPasswordToggles() {
    const toggles = document.querySelectorAll('.toggle-password');
    toggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            const input = this.previousElementSibling;
            const type = input.getAttribute('type') === 'password' ? 'text' : 'password';
            input.setAttribute('type', type);
            this.textContent = type === 'password' ? '显示' : '隐藏';
        });
    });
}

// 验证码按钮功能
function initVerificationCodeButtons() {
    const codeButtons = document.querySelectorAll('.code-btn');
    
    codeButtons.forEach(button => {
        button.addEventListener('click', async function() {
            const buttonId = this.id;
            let phone, email;
            
            // 获取对应输入框的值
            if (buttonId === 'send-phone-code') {
                phone = document.getElementById('phone').value;
                if (!validatePhone(phone)) {
                    showToast('请输入有效的手机号码');
                    return;
                }
                await sendVerificationCode('phone', phone);
            } else if (buttonId === 'send-reg-phone-code') {
                phone = document.getElementById('reg-phone').value;
                if (!validatePhone(phone)) {
                    showToast('请输入有效的手机号码');
                    return;
                }
                await sendVerificationCode('phone', phone);
            }
            
            // 启动倒计时
            startCountdown(this);
        });
    });
}

// 发送验证码
async function sendVerificationCode(type, value) {
    try {
        const response = await fetch('/api/send-verification-code', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: type,
                value: value
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('验证码已发送');
        } else {
            showToast(data.message || '发送验证码失败，请稍后重试');
        }
    } catch (error) {
        console.error('发送验证码出错:', error);
        showToast('网络错误，请稍后重试');
    }
}

// 验证码按钮倒计时
function startCountdown(button) {
    let seconds = 60;
    const originalText = button.textContent;
    button.disabled = true;
    
    const countdown = setInterval(() => {
        seconds--;
        button.textContent = `重新获取(${seconds}s)`;
        
        if (seconds <= 0) {
            clearInterval(countdown);
            button.textContent = originalText;
            button.disabled = false;
        }
    }, 1000);
}

// 表单验证
function initFormValidation() {
    // 手机登录表单验证
    const phoneLoginForm = document.getElementById('phone-login-form');
    if (phoneLoginForm) {
        phoneLoginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const phone = document.getElementById('phone').value;
            const code = document.getElementById('phone-code').value;
            
            if (!validatePhone(phone)) {
                showToast('请输入有效的手机号码');
                return;
            }
            
            if (!code) {
                showToast('请输入验证码');
                return;
            }
            
            await submitForm(this);
        });
    }
    
    // 邮箱登录表单验证
    const emailLoginForm = document.getElementById('email-login-form');
    if (emailLoginForm) {
        emailLoginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            
            if (!validateEmail(email)) {
                showToast('请输入有效的邮箱地址');
                return;
            }
            
            if (!password) {
                showToast('请输入密码');
                return;
            }
            
            await submitForm(this);
        });
    }
    
    // 手机注册表单验证
    const phoneRegisterForm = document.getElementById('phone-register-form');
    if (phoneRegisterForm) {
        phoneRegisterForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const phone = document.getElementById('reg-phone').value;
            const code = document.getElementById('reg-phone-code').value;
            const nickname = document.getElementById('reg-nickname').value;
            const agreement = this.querySelector('input[name="agreement"]').checked;
            
            if (!validatePhone(phone)) {
                showToast('请输入有效的手机号码');
                return;
            }
            
            if (!code) {
                showToast('请输入验证码');
                return;
            }
            
            if (!nickname) {
                showToast('请设置您的昵称');
                return;
            }
            
            if (!agreement) {
                showToast('请阅读并同意服务条款和隐私政策');
                return;
            }
            
            await submitForm(this);
        });
    }
    
    // 邮箱注册表单验证
    const emailRegisterForm = document.getElementById('email-register-form');
    if (emailRegisterForm) {
        emailRegisterForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const email = document.getElementById('reg-email').value;
            const password = document.getElementById('reg-password').value;
            const confirmPassword = document.getElementById('reg-confirm-password').value;
            const nickname = document.getElementById('reg-email-nickname').value;
            const agreement = this.querySelector('input[name="agreement"]').checked;
            
            if (!validateEmail(email)) {
                showToast('请输入有效的邮箱地址');
                return;
            }
            
            if (password.length < 8) {
                showToast('密码长度至少为8位');
                return;
            }
            
            if (password !== confirmPassword) {
                showToast('两次输入的密码不一致');
                return;
            }
            
            if (!nickname) {
                showToast('请设置您的昵称');
                return;
            }
            
            if (!agreement) {
                showToast('请阅读并同意服务条款和隐私政策');
                return;
            }
            
            await submitForm(this);
        });
    }
}

// 表单提交
async function submitForm(form) {
    const submitButton = form.querySelector('button[type="submit"]');
    const originalText = submitButton.textContent;
    
    try {
        submitButton.disabled = true;
        submitButton.textContent = '处理中...';
        
        const formData = new FormData(form);
        const data = {};
        
        formData.forEach((value, key) => {
            data[key] = value;
        });
        
        const response = await fetch(form.action, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showToast(result.message || '操作成功');
            
            // 登录/注册成功后重定向
            if (result.redirect) {
                setTimeout(() => {
                    window.location.href = result.redirect;
                }, 1000);
            }
        } else {
            showToast(result.message || '操作失败，请稍后重试');
        }
    } catch (error) {
        console.error('提交表单出错:', error);
        showToast('网络错误，请稍后重试');
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = originalText;
    }
}

// 微信登录/注册功能
function initWechatAuth() {
    const wechatLoginBtn = document.getElementById('wechat-login');
    const wechatRegisterBtn = document.getElementById('wechat-register');
    
    if (wechatLoginBtn) {
        wechatLoginBtn.addEventListener('click', () => {
            initiateWechatAuth('login');
        });
    }
    
    if (wechatRegisterBtn) {
        wechatRegisterBtn.addEventListener('click', () => {
            initiateWechatAuth('register');
        });
    }
}

// 发起微信认证
async function initiateWechatAuth(action) {
    try {
        const response = await fetch(`/api/wechat-auth/${action}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success && data.qrUrl) {
            showWechatQrCode(data.qrUrl, action);
        } else {
            showToast(data.message || '获取微信二维码失败，请稍后重试');
        }
    } catch (error) {
        console.error('微信认证出错:', error);
        showToast('网络错误，请稍后重试');
    }
}

// 显示微信二维码
function showWechatQrCode(qrUrl, action) {
    const title = action === 'login' ? '微信登录' : '微信注册';
    
    // 创建模态框
    const modal = document.createElement('div');
    modal.className = 'wechat-modal';
    
    modal.innerHTML = `
        <div class="wechat-modal-content">
            <span class="close-modal">&times;</span>
            <h3>${title}</h3>
            <div class="qr-container">
                <img src="${qrUrl}" alt="微信二维码">
            </div>
            <p>请使用微信扫描二维码${action === 'login' ? '登录' : '注册'}</p>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // 关闭模态框
    const closeBtn = modal.querySelector('.close-modal');
    closeBtn.addEventListener('click', () => {
        document.body.removeChild(modal);
    });
    
    // 点击模态框外部关闭
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            document.body.removeChild(modal);
        }
    });
    
    // 启动轮询检查认证状态
    const checkInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/check-wechat-auth-status');
            const data = await response.json();
            
            if (data.success && data.authenticated) {
                clearInterval(checkInterval);
                document.body.removeChild(modal);
                showToast(data.message || '认证成功');
                
                // 重定向
                if (data.redirect) {
                    window.location.href = data.redirect;
                }
            }
        } catch (error) {
            console.error('检查认证状态出错:', error);
        }
    }, 2000);
    
    // 5分钟后停止轮询
    setTimeout(() => {
        clearInterval(checkInterval);
    }, 300000);
}

// 工具函数：显示提示消息
function showToast(message) {
    // 检查是否已经存在toast元素
    const existingToast = document.querySelector('.toast');
    if (existingToast) {
        document.body.removeChild(existingToast);
    }
    
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    // 显示动画
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    // 3秒后隐藏
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            if (document.body.contains(toast)) {
                document.body.removeChild(toast);
            }
        }, 300);
    }, 3000);
}

// 工具函数：验证手机号
function validatePhone(phone) {
    return /^1[3-9]\d{9}$/.test(phone);
}

// 工具函数：验证邮箱
function validateEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

// 添加Toast样式
const style = document.createElement('style');
style.textContent = `
    .toast {
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%) translateY(-100px);
        background-color: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 12px 24px;
        border-radius: 4px;
        z-index: 10000;
        transition: transform 0.3s ease;
    }
    
    .toast.show {
        transform: translateX(-50%) translateY(0);
    }
    
    .wechat-modal {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    }
    
    .wechat-modal-content {
        background-color: white;
        padding: 30px;
        border-radius: 8px;
        text-align: center;
        position: relative;
        max-width: 90%;
        width: 320px;
    }
    
    .close-modal {
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 24px;
        cursor: pointer;
        color: #999;
    }
    
    .qr-container {
        margin: 20px 0;
    }
    
    .qr-container img {
        max-width: 100%;
        height: auto;
    }
`;

document.head.appendChild(style);