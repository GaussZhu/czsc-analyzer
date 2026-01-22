// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化标签切换功能
    initTabs();
    
    // 生成示例K线图
    generateSampleKline();
    
    // 生成雷达图
    generateRadarChart();
    
    // 生成英雄区域的动态图表
    generateHeroChart();
    
    // 导航滚动效果
    initSmoothScroll();
});

// 标签切换功能
function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // 移除所有标签的活动状态
            tabs.forEach(t => t.classList.remove('active'));
            
            // 隐藏所有标签内容
            const tabPanes = document.querySelectorAll('.tab-pane');
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // 激活当前标签和内容
            tab.classList.add('active');
            const tabId = tab.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// 生成示例K线图
function generateSampleKline() {
    // 示例数据
    const dates = [];
    const opens = [];
    const highs = [];
    const lows = [];
    const closes = [];
    const volumes = [];
    
    // 生成30天的随机数据
    const basePrice = 15.0;
    let currentPrice = basePrice;
    
    for (let i = 0; i < 30; i++) {
        const date = new Date();
        date.setDate(date.getDate() - (30 - i));
        dates.push(date.toISOString().split('T')[0]);
        
        // 随机波动
        const change = (Math.random() - 0.5) * 0.5;
        currentPrice = Math.max(5, currentPrice + change);
        
        const open = currentPrice;
        const close = currentPrice + (Math.random() - 0.5) * 0.3;
        const high = Math.max(open, close) + Math.random() * 0.2;
        const low = Math.min(open, close) - Math.random() * 0.2;
        const volume = Math.floor(Math.random() * 10000) + 5000;
        
        opens.push(open);
        highs.push(high);
        lows.push(low);
        closes.push(close);
        volumes.push(volume);
        
        currentPrice = close;
    }
    
    // 绘制K线图
    const trace = {
        x: dates,
        close: closes,
        high: highs,
        low: lows,
        open: opens,
        
        // 自定义颜色 - 红涨绿跌
        increasing: {line: {color: '#b22222'}},
        decreasing: {line: {color: '#2e8b57'}},
        
        type: 'candlestick',
        name: '股价'
    };
    
    // 添加买点示例
    const buyPoints = {
        x: [dates[5], dates[12], dates[22]],
        y: [lows[5] - 0.2, lows[12] - 0.2, lows[22] - 0.2],
        mode: 'markers+text',
        marker: {
            symbol: 'triangle-up',
            size: 15,
            color: '#b22222',
            line: {color: 'white', width: 1}
        },
        text: ['买1', '买2', '买3'],
        textposition: 'bottom center',
        name: '买点'
    };
    
    // 中枢区域
    const centers = {
        x: [dates[14], dates[14], dates[20], dates[20], dates[14]],
        y: [highs[14] + 0.3, lows[14] - 0.3, lows[20] - 0.3, highs[20] + 0.3, highs[14] + 0.3],
        fill: 'toself',
        fillcolor: 'rgba(178, 34, 34, 0.1)',
        line: {color: 'rgba(178, 34, 34, 0.5)', width: 1, dash: 'dot'},
        name: '中枢区域'
    };
    
    const data = [trace, buyPoints, centers];
    
    const layout = {
        title: '',
        showlegend: false,
        xaxis: {
            rangeslider: {visible: false},
            gridcolor: '#f5f5f5',
            showgrid: true
        },
        yaxis: {
            autorange: true,
            gridcolor: '#f5f5f5',
            showgrid: true
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        margin: {l: 40, r: 10, b: 40, t: 10}
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('sample-kline', data, layout, config);
}

// 生成雷达图
function generateRadarChart() {
    const ctx = document.getElementById('strategy-radar');
    
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: [
                '买点精度',
                '盈利能力',
                '回撤控制',
                '胜率',
                '盈亏比',
                '抗风险性'
            ],
            datasets: [{
                label: '缠论买卖点',
                data: [85, 82, 78, 80, 90, 75],
                backgroundColor: 'rgba(178, 34, 34, 0.2)',
                borderColor: '#b22222',
                pointBackgroundColor: '#b22222',
                pointBorderColor: '#fff'
            }, {
                label: '中枢形态',
                data: [75, 88, 85, 81, 76, 89],
                backgroundColor: 'rgba(44, 62, 80, 0.2)',
                borderColor: '#2c3e50',
                pointBackgroundColor: '#2c3e50',
                pointBorderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            elements: {
                line: {
                    borderWidth: 2
                }
            },
            scales: {
                r: {
                    angleLines: {
                        color: '#e0e0e0'
                    },
                    grid: {
                        color: '#f0f0f0'
                    },
                    pointLabels: {
                        font: {
                            size: 12
                        }
                    },
                    suggestedMin: 50,
                    suggestedMax: 100
                }
            }
        }
    });
}

// 生成英雄区域图表
function generateHeroChart() {
    const dates = [];
    const values1 = [];
    const values2 = [];
    
    // 生成示例数据
    for (let i = 0; i < 60; i++) {
        const date = new Date();
        date.setDate(date.getDate() - (60 - i));
        dates.push(date.toISOString().split('T')[0]);
        
        // 模拟对比策略
        values1.push(1000 * (1 + 0.002 * i + Math.sin(i/10) * 0.05));
        values2.push(1000 * (1 + 0.001 * i));
    }
    
    const trace1 = {
        x: dates,
        y: values1,
        type: 'scatter',
        mode: 'lines',
        name: '策略收益',
        line: {
            color: '#b22222',
            width: 3
        }
    };
    
    const trace2 = {
        x: dates,
        y: values2,
        type: 'scatter',
        mode: 'lines',
        name: '基准收益',
        line: {
            color: '#2c3e50',
            width: 2,
            dash: 'dot'
        }
    };
    
    const data = [trace1, trace2];
    
    const layout = {
        title: '',
        showlegend: false,
        xaxis: {
            showgrid: false,
            showticklabels: false
        },
        yaxis: {
            showgrid: false,
            showticklabels: false
        },
        plot_bgcolor: 'transparent',
        paper_bgcolor: 'transparent',
        margin: {l: 0, r: 0, b: 0, t: 0}
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('hero-chart', data, layout, config);
}

// 平滑滚动效果
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (!targetElement) return;
            
            window.scrollTo({
                top: targetElement.offsetTop - 80,
                behavior: 'smooth'
            });
        });
    });
}