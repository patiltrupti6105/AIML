// Trading Simulator Frontend
const API_BASE = window.location.origin;
const WS_BASE = window.location.origin.replace('http', 'ws');
let ws = null;
let isConnected = false;

// DOM Elements
const symbolSelect = document.getElementById("symbolSelect");
const speedSlider = document.getElementById("speedSlider");
const speedLabel = document.getElementById("speedLabel");
const useModelCheck = document.getElementById("useModelCheck");
const connectBtn = document.getElementById("connectBtn");
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const stepBtn = document.getElementById("stepBtn");
const logBox = document.getElementById("logBox");
const actionsLog = document.getElementById("actionsLog");
const netWorthEl = document.getElementById("netWorth");
const cashEl = document.getElementById("cash");
const uplEl = document.getElementById("upl");
const totalReturnEl = document.getElementById("totalReturn");
const currentStepEl = document.getElementById("currentStep");
const cumRewardEl = document.getElementById("cumReward");
const statusText = document.getElementById("statusText");
const progressText = document.getElementById("progressText");

// Data
let candles = [], closes = [];
let volumes = []; // Track volume data
let candleTimestamps = []; // Track timestamps for auto-scroll
let buyMarkers = {x: [], y: []}; // Buy markers
let sellMarkers = {x: [], y: []}; // Sell markers
let portfolioX = [], portfolioY = [];
let indicatorTimestamps = []; // Track timestamps for indicators
let rsiData = [], macdData = [], macdSignalData = []; // Indicator data
let totalBars = 0, currentBar = 0;
let chartsInitialized = false;
const INITIAL_CAPITAL = 100000; // Must match backend config.INITIAL_CAPITAL

// Logging
function log(msg, type = 'info') {
    console.log(`[${type}] ${msg}`);
    const div = document.createElement("div");
    div.className = `log-entry ${type}`;
    div.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logBox.insertBefore(div, logBox.firstChild);
    while (logBox.children.length > 50) logBox.removeChild(logBox.lastChild);
}

function logAction(action, price, timestamp, sharesSold = 0) {
    const div = document.createElement("div");
    div.className = `action-item ${action}`;
    let icon = '⸱', text = 'HOLD';
    if (action === 'buy') { 
        icon = '🟢'; 
        text = 'BUY 1 share'; 
    }
    else if (action === 'sell') { 
        icon = '🔴'; 
        if (sharesSold > 1) {
            text = `SELL ${sharesSold} shares`;
        } else {
            text = 'SELL 1 share';
        }
    }
    div.innerHTML = `<strong>${icon} ${text}</strong> @ $${price.toFixed(2)} <span style="color: var(--muted); font-size: 11px;"></span>`;
    actionsLog.insertBefore(div, actionsLog.firstChild);
    while (actionsLog.children.length > 20) actionsLog.removeChild(actionsLog.lastChild);
}

// Indicators
function calcSMA20() {
    if (closes.length < 20) return null;
    return closes.slice(-20).reduce((a,b) => a+b, 0) / 20;
}

let lastEMA = null;
function calcEMA20(close) {
    const k = 2 / 21;
    if (lastEMA == null) { lastEMA = close; return lastEMA; }
    lastEMA = (close - lastEMA) * k + lastEMA;
    return lastEMA;
}

function calcRSI() {
    if (closes.length < 15) return null;
    let gains = 0, losses = 0;
    for (let i = closes.length - 14; i < closes.length; i++) {
        const diff = closes[i] - closes[i-1];
        if (diff > 0) gains += diff; else losses -= diff;
    }
    const rs = gains / (losses || 1);
        return 100 - 100/(1+rs);
}

// Update indicators chart
function updateIndicators(ts, indicators) {
    if (!chartsInitialized || !indicators) {
        return;
    }
    
    try {
        // Parse and validate timestamp
        let timestamp = ts;
        if (!timestamp || timestamp === null || timestamp === undefined || timestamp === 'null' || timestamp === 'undefined') {
            timestamp = new Date().toISOString();
        } else if (typeof timestamp === 'string') {
            if (timestamp.trim() === '') {
                timestamp = new Date().toISOString();
            }
        } else if (timestamp instanceof Date) {
            timestamp = timestamp.toISOString();
        } else {
            timestamp = new Date().toISOString();
        }
        
        // Update indicator data arrays
        if (indicators.rsi !== undefined) {
            indicatorTimestamps.push(timestamp);
            rsiData.push(Number(indicators.rsi));
        }
        if (indicators.macd !== undefined) {
            macdData.push(Number(indicators.macd));
        }
        if (indicators.macd_signal !== undefined) {
            macdSignalData.push(Number(indicators.macd_signal));
        }
        
        // Keep only last 200 points for performance
        const maxPoints = 200;
        if (indicatorTimestamps.length > maxPoints) {
            indicatorTimestamps = indicatorTimestamps.slice(-maxPoints);
            rsiData = rsiData.slice(-maxPoints);
            macdData = macdData.slice(-maxPoints);
            macdSignalData = macdSignalData.slice(-maxPoints);
        }
        
        // Update chart
        if (indicatorTimestamps.length > 0) {
            Plotly.restyle("indicatorsChart", {
                x: [indicatorTimestamps],
                y: [rsiData]
            }, [0]); // RSI trace
            
            if (macdData.length > 0) {
                Plotly.restyle("indicatorsChart", {
                    x: [indicatorTimestamps],
                    y: [macdData]
                }, [1]); // MACD trace
            }
            
            if (macdSignalData.length > 0) {
                Plotly.restyle("indicatorsChart", {
                    x: [indicatorTimestamps],
                    y: [macdSignalData]
                }, [2]); // Signal trace
            }
            
            // Auto-scroll to show latest data
            if (indicatorTimestamps.length > 0) {
                const visiblePoints = 100;
                const startIdx = Math.max(0, indicatorTimestamps.length - visiblePoints);
                const startDate = new Date(indicatorTimestamps[startIdx]);
                const endDate = new Date(timestamp);
                
                if (!isNaN(startDate.getTime()) && !isNaN(endDate.getTime())) {
                    Plotly.relayout("indicatorsChart", {
                        'xaxis.range': [startDate.toISOString(), endDate.toISOString()]
                    });
                }
            }
        }
    } catch (error) {
        console.error("Error updating indicators:", error);
    }
}

// Charts
function initCharts() {
    log("Initializing charts...");
    
    try {
        const candleEl = document.getElementById("candles");
        const portfolioEl = document.getElementById("portfolioChart");
        const indicatorsEl = document.getElementById("indicatorsChart");
        
        if (!candleEl || !portfolioEl || !indicatorsEl) {
            log("Chart elements not found!", "error");
            return false;
        }
        
        log("Chart elements found, creating Plotly charts...");
        
        // Candlestick chart with buy/sell markers
        Plotly.newPlot("candles", [
            { 
                x: [], open: [], high: [], low: [], close: [], 
                type: "candlestick", name: "Price",
                increasing: {line: {color: '#26a69a', width: 1}, fillcolor: '#26a69a'}, 
                decreasing: {line: {color: '#ef5350', width: 1}, fillcolor: '#ef5350'},
                xaxis: 'x',
                yaxis: 'y'
            },
            { x: [], y: [], name: "BUY", mode: "markers", type: "scatter", 
              marker: {symbol: "triangle-up", size: 14, color: '#16e624ff', line: {color: '#ffffff', width: 1.5}} },
            { x: [], y: [], name: "SELL", mode: "markers", type: "scatter",
              marker: {symbol: "triangle-down", size: 14, color: '#f6100cff', line: {color: '#ffffff', width: 1.5}} }
        ], {
            margin: {t:30, b:10, l:60, r:20}, 
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)', 
            font: {color: '#00d4ff', size: 12},
            xaxis: {
                gridcolor: 'rgba(255,255,255,0.05)', 
                type: 'date',
                showgrid: true,
                gridwidth: 1,
                side: 'bottom'
            }, 
            yaxis: {
                gridcolor: 'rgba(255,255,255,0.05)', 
                title: 'Price ($)',
                side: 'right',
                showgrid: true,
                gridwidth: 1
            },
            legend: {x: 0, y: 1, orientation: 'h', bgcolor: 'rgba(255,255,255,0.8)'},
            dragmode: 'pan'
        });
        log("Candlestick chart created");

        // Portfolio chart with better visibility
        Plotly.newPlot("portfolioChart", [
            { 
                x: [], y: [], mode: "lines", name: "Net Worth", 
                type: "scatter",
                line: {color: '#00ff9d', width: 3},
                fill: 'tozeroy', fillcolor: 'rgba(0, 255, 157, 0.2)'
            }
        ], {
            margin: {t:20, b:30, l:60, r:20}, 
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(255,255,255,0.02)', 
            font: {color: '#00d4ff', size: 11},
            xaxis: {gridcolor: 'rgba(255,255,255,0.05)', type: 'date', title: 'Time'}, 
            yaxis: {gridcolor: 'rgba(255,255,255,0.05)', title: 'Portfolio Value ($)', 
                    tickformat: '$,.0f', showgrid: true}
        });
        log("Portfolio chart created");

        // Indicators chart (RSI and MACD subplots)
        Plotly.newPlot("indicatorsChart", [
            {
                x: [],
                y: [],
                mode: "lines",
                name: "RSI",
                type: "scatter",
                line: {color: '#ff6b6b', width: 2},
                yaxis: 'y'
            },
            {
                x: [],
                y: [],
                mode: "lines",
                name: "MACD",
                type: "scatter",
                line: {color: '#4ecdc4', width: 2},
                yaxis: 'y2'
            },
            {
                x: [],
                y: [],
                mode: "lines",
                name: "Signal",
                type: "scatter",
                line: {color: '#ffe66d', width: 2, dash: 'dash'},
                yaxis: 'y2'
            }
        ], {
            margin: {t:30, b:50, l:60, r:60},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: '#ffffff',
            font: {color: '#00d4ff', size: 11},
            xaxis: {
                gridcolor: 'rgba(255,255,255,0.05)',
                type: 'date',
                showgrid: true,
                gridwidth: 1,
                side: 'bottom',
                title: 'Time'
            },
            yaxis: {
                gridcolor: 'rgba(255,255,255,0.05)',
                title: 'RSI',
                side: 'left',
                showgrid: true,
                gridwidth: 1,
                range: [0, 100],
                domain: [0.55, 1.0]
            },
            yaxis2: {
                gridcolor: 'rgba(255,255,255,0.05)',
                title: 'MACD',
                side: 'right',
                showgrid: true,
                gridwidth: 1,
                domain: [0, 0.45],
                overlaying: 'x'
            },
            legend: {x: 0, y: 1, orientation: 'h', bgcolor: 'rgba(255,255,255,0.8)'},
            dragmode: 'pan'
        });
        log("Indicators chart created");
        
        chartsInitialized = true;
        log("All charts initialized successfully", "success");
        return true;
        
    } catch (error) {
        log(`Error initializing charts: ${error.message}`, "error");
        console.error("Chart initialization error:", error);
        return false;
    }
}

function addCandle(c, isBatch = false) {
    if (!chartsInitialized) {
        log("Charts not initialized, skipping candle", "error");
        return;
    }
    
    try {
        candles.push(c); 
        closes.push(c.close);
        volumes.push(c.volume || 0);
        currentBar++;
        
        // For batch updates, we'll handle them separately
        if (!isBatch) {
            // Parse and validate timestamp - never pass null to Plotly
            let ts = c.timestamp;
            if (!ts || ts === null || ts === undefined) {
                // Generate a timestamp if missing
                ts = new Date().toISOString();
                log("Warning: Missing timestamp, using current time", "error");
            } else if (typeof ts === 'string') {
                // Validate it's a proper date string
                if (ts.trim() === '' || ts === 'null' || ts === 'undefined') {
                    ts = new Date().toISOString();
                }
            } else if (ts instanceof Date) {
                ts = ts.toISOString();
            } else {
                // Fallback to current time
                ts = new Date().toISOString();
            }
            
            // Only update if we have valid data
            if (ts && typeof c.open === 'number' && !isNaN(c.open)) {
                // Store timestamp for auto-scroll
                candleTimestamps.push(ts);
                
                // Update candlestick
                Plotly.extendTraces("candles", {
                    x: [[ts]],
                    open: [[c.open]], 
                    high: [[c.high]], 
                    low: [[c.low]], 
                    close: [[c.close]]
                }, [0]);
                
                // Add buy/sell markers if action occurred
                const action = c.action !== undefined ? c.action : 0;
                if (action === 1) { // Buy
                    buyMarkers.x.push(ts);
                    buyMarkers.y.push(c.low * 0.998); // Place marker slightly below the low
                    Plotly.extendTraces("candles", {
                        x: [[ts]],
                        y: [[c.low * 0.998]]
                    }, [1]); // Trace 1 is BUY markers
                } else if (action === 2) { // Sell
                    sellMarkers.x.push(ts);
                    sellMarkers.y.push(c.high * 1.002); // Place marker slightly above the high
                    Plotly.extendTraces("candles", {
                        x: [[ts]],
                        y: [[c.high * 1.002]]
                    }, [2]); // Trace 2 is SELL markers
                }
                
                // Auto-scroll charts to show latest data - calculate range from our data
                try {
                    if (candleTimestamps.length > 0) {
                        // Use our internal timestamps array to calculate range
                        const visiblePoints = 100; // Show last 100 points
                        const startIdx = Math.max(0, candleTimestamps.length - visiblePoints);
                        const startDate = new Date(candleTimestamps[startIdx]);
                        const endDate = new Date(ts);
                        
                        // Validate dates before using
                        if (!isNaN(startDate.getTime()) && !isNaN(endDate.getTime())) {
                            Plotly.relayout("candles", {
                                'xaxis.range': [startDate.toISOString(), endDate.toISOString()]
                            });
                        }
                    }
                } catch (relayoutError) {
                    // Silently fail - autorange will handle it
                }
            } else {
                log(`Skipping invalid candle data: timestamp=${ts}, open=${c.open}`, "error");
            }
        }
        
        if (totalBars > 0) {
            const progress = ((currentBar / totalBars) * 100).toFixed(1);
            progressText.textContent = `Progress: ${progress}% (${currentBar}/${totalBars})`;
        }
    } catch (error) {
        console.error("Error adding candle:", error);
        log(`Error adding candle: ${error.message}`, "error");
    }
}

function addInitialCandles(candleData) {
    if (!chartsInitialized) {
        log("Charts not initialized, cannot add initial candles", "error");
        return;
    }
    
    if (!candleData || candleData.length === 0) {
        log("No candle data provided", "error");
        return;
    }
    
    try {
        log(`Processing ${candleData.length} initial candles...`, "info");
        
        // Process all candles first to build arrays
        const timestamps = [];
        const opens = [];
        const highs = [];
        const lows = [];
        const closes_data = [];
        
        candleData.forEach((c, idx) => {
            // Validate candle data
            if (!c || typeof c.open !== 'number' || typeof c.close !== 'number') {
                log(`Invalid candle data at index ${idx}`, "error");
                return;
            }
            
            // Add to internal arrays
            candles.push(c); 
            closes.push(c.close);
            volumes.push(c.volume || 0);
            currentBar++;
            
            // Parse timestamp - ensure it's a valid date string, never null
            let ts = c.timestamp;
            if (!ts || ts === null || ts === undefined) {
                // Generate timestamp based on index if missing
                ts = new Date(2000 + Math.floor(idx / 252), (idx % 12), 1).toISOString();
                log(`Warning: Missing timestamp at index ${idx}, generated: ${ts}`, "error");
            } else if (typeof ts === 'string') {
                // Validate it's a proper date string
                if (ts.trim() === '' || ts === 'null' || ts === 'undefined') {
                    ts = new Date(2000 + Math.floor(idx / 252), (idx % 12), 1).toISOString();
                }
            } else if (ts instanceof Date) {
                ts = ts.toISOString();
            } else {
                // Fallback
                ts = new Date(2000 + Math.floor(idx / 252), (idx % 12), 1).toISOString();
            }
            timestamps.push(ts);
            candleTimestamps.push(ts); // Track for auto-scroll
            opens.push(Number(c.open));
            highs.push(Number(c.high));
            lows.push(Number(c.low));
            closes_data.push(Number(c.close));
        });
        
        log(`Built arrays: ${timestamps.length} timestamps, ${opens.length} prices`, "info");
        
        // Rebuild candlestick chart completely with all data (including buy/sell markers from initial data)
        const buyX = [], buyY = [], sellX = [], sellY = [];
        candleData.forEach((c, idx) => {
            if (c.action === 1) {
                buyX.push(timestamps[idx]);
                buyY.push(lows[idx] * 0.998);
            } else if (c.action === 2) {
                sellX.push(timestamps[idx]);
                sellY.push(highs[idx] * 1.002);
            }
        });
        
        if (timestamps.length > 0 && opens.length > 0) {
            // Candlestick chart
            Plotly.newPlot("candles", [
                { 
                    x: timestamps,
                    open: opens,
                    high: highs,
                    low: lows,
                    close: closes_data,
                    type: "candlestick",
                    name: "Price",
                    increasing: {line: {color: '#26a69a', width: 1}, fillcolor: '#26a69a'}, 
                    decreasing: {line: {color: '#ef5350', width: 1}, fillcolor: '#ef5350'},
                    xaxis: 'x',
                    yaxis: 'y'
                },
                { 
                    x: buyX,
                    y: buyY,
                    name: "BUY",
                    mode: "markers",
                    type: "scatter",
                    marker: {symbol: "triangle-up", size: 14, color: '#11f31cff', line: {color: '#ffffff', width: 1.5}}
                },
                { 
                    x: sellX,
                    y: sellY,
                    name: "SELL",
                    mode: "markers",
                    type: "scatter",
                    marker: {symbol: "triangle-down", size: 14, color: '#e3130fff', line: {color: '#ffffff', width: 1.5}}
                }
            ], {
                margin: {t:30, b:10, l:60, r:20}, 
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#ffffff', 
                font: {color: '#00d4ff', size: 12},
                xaxis: {
                    gridcolor: 'rgba(255,255,255,0.05)', 
                    type: 'date',
                    showgrid: true,
                    gridwidth: 1,
                    side: 'bottom'
                }, 
                yaxis: {
                    gridcolor: 'rgba(255,255,255,0.05)', 
                    title: 'Price ($)',
                    side: 'right',
                    showgrid: true,
                    gridwidth: 1
                },
                legend: {x: 0, y: 1, orientation: 'h', bgcolor: 'rgba(255,255,255,0.8)'},
                dragmode: 'pan'
            }, {responsive: true});
            
            log("Candlestick chart rebuilt successfully", "success");
        } else {
            log("No valid data to plot in charts", "error");
        }
        
        log(`Added ${candleData.length} initial candles to charts`, "success");
    } catch (error) {
        console.error("Error adding initial candles:", error);
        log(`Error adding initial candles: ${error.message}`, "error");
        console.error("Stack trace:", error.stack);
        // Fallback: try adding candles one by one
        log("Trying fallback method: adding candles individually", "info");
        try {
            // Reset state first
            candles = [];
            closes = [];
            volumes = [];
            candleTimestamps = [];
            currentBar = 0;
            candleData.forEach(c => addCandle(c, false));
        } catch (fallbackError) {
            log(`Fallback also failed: ${fallbackError.message}`, "error");
            console.error("Fallback error:", fallbackError);
        }
    }
}

function updatePortfolio(ts, p) {
    try {
        // Parse and validate timestamp - never pass null to Plotly
        let timestamp = ts;
        if (!timestamp || timestamp === null || timestamp === undefined || timestamp === 'null' || timestamp === 'undefined') {
            // Generate timestamp if missing
            timestamp = new Date().toISOString();
            log("Warning: Missing portfolio timestamp, using current time", "error");
        } else if (typeof timestamp === 'string') {
            if (timestamp.trim() === '') {
                timestamp = new Date().toISOString();
            }
        } else if (timestamp instanceof Date) {
            timestamp = timestamp.toISOString();
        } else {
            timestamp = new Date().toISOString();
        }
        
        const portfolioValue = Number(p.portfolio_value);
        if (isNaN(portfolioValue)) {
            log(`Warning: Invalid portfolio value: ${p.portfolio_value}`, "error");
            return;
        }
        
        portfolioX.push(timestamp); 
        portfolioY.push(portfolioValue);
        
        netWorthEl.textContent = `$${p.portfolio_value.toFixed(2)}`;
        cashEl.textContent = `$${p.cash.toFixed(2)}`;
        
        const upl = p.unrealized_pl;
        uplEl.textContent = `$${upl.toFixed(2)}`;
        uplEl.className = `value ${upl >= 0 ? 'positive' : 'negative'}`;
        
        // Update portfolio chart live
        if (chartsInitialized) {
            try {
                // Always use extendTraces - it works even on empty charts
                Plotly.extendTraces("portfolioChart", { 
                    x: [[timestamp]], 
                    y: [[Number(p.portfolio_value)]] 
                }, [0]);
                
                // Auto-scroll portfolio chart to show latest data and ensure proper Y-axis scaling
                try {
                    if (portfolioX.length > 0 && portfolioY.length > 0) {
                        // Use our internal portfolioX array to calculate range
                        const visiblePoints = 100; // Show last 100 points
                        const startIdx = Math.max(0, portfolioX.length - visiblePoints);
                        const startDate = new Date(portfolioX[startIdx]);
                        const endDate = new Date(timestamp);
                        
                        // Calculate Y-axis range to show growth properly
                        const visibleY = portfolioY.slice(startIdx);
                        const minY = Math.min(...visibleY, INITIAL_CAPITAL);
                        const maxY = Math.max(...visibleY, INITIAL_CAPITAL);
                        const yPadding = (maxY - minY) * 0.1 || INITIAL_CAPITAL * 0.05;
                        
                        Plotly.relayout("portfolioChart", {
                            'xaxis.range': [startDate.toISOString(), endDate.toISOString()],
                            'yaxis.range': [Math.max(0, minY - yPadding), maxY + yPadding]
                        });
                    } else {
                        // Fallback: just set autorange
                        Plotly.relayout("portfolioChart", {
                            'xaxis.autorange': true,
                            'yaxis.autorange': true
                        });
                    }
                } catch (relayoutError) {
                    // If relayout fails, just use autorange (no errors)
                    Plotly.relayout("portfolioChart", {
                        'xaxis.autorange': true,
                        'yaxis.autorange': true
                    });
                }
            } catch (extendError) {
                // If extend fails (empty chart), use restyle to set initial data
                log("Extend failed, using restyle for portfolio", "info");
                Plotly.restyle("portfolioChart", {
                    x: [portfolioX],
                    y: [portfolioY]
                }, [0]);
            }
        }
    } catch (error) {
        console.error("Error updating portfolio:", error);
        log(`Error updating portfolio: ${error.message}`, "error");
    }
}

// WebSocket
function connectWS() {
    if (ws) {
        log("Closing existing connection...");
        ws.close();
    }
    
    // Reset data
    candles = []; 
    closes = [];
    volumes = []; // Reset volume tracking
    candleTimestamps = []; // Reset timestamp tracking
    buyMarkers = {x: [], y: []}; // Reset buy markers
    sellMarkers = {x: [], y: []}; // Reset sell markers
    portfolioX = []; 
    portfolioY = [];
    indicatorTimestamps = []; // Reset indicator timestamps
    rsiData = []; // Reset RSI data
    macdData = []; // Reset MACD data
    macdSignalData = []; // Reset MACD signal data
    currentBar = 0; 
    totalBars = 0;
    logBox.innerHTML = ''; 
    actionsLog.innerHTML = '';
    
    // Initialize charts
    if (!initCharts()) {
        log("Failed to initialize charts", "error");
        return;
    }
    
    const symbol = symbolSelect.value;
    const speed = parseFloat(speedSlider.value);
    const useModel = useModelCheck.checked ? 1 : 0;
    
    if (!symbol) { 
        log("Please select symbol", "error"); 
        return; 
    }
    
    const wsUrl = `${WS_BASE}/ws/simulate/${symbol}/all?speed=${speed}&use_model=${useModel}`;
    log(`Connecting to: ${wsUrl}`);
    statusText.textContent = "Connecting...";
    
    try {
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            log("Connected successfully", "success");
            statusText.textContent = "Connected"; 
            statusText.className = "connected"; 
            isConnected = true;
            playBtn.disabled = false; 
            pauseBtn.disabled = false; 
            stepBtn.disabled = false;
            connectBtn.textContent = "🔌 Disconnect";
        };
        
        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                
                if (msg.type === "initial") {
                    totalBars = msg.total_bars;
                    log(`Received ${msg.data.length} initial candles`);
                    if (msg.data && msg.data.length > 0) {
                        addInitialCandles(msg.data);
                        log(`Loaded ${msg.data.length} initial candles`, "success");
                    }
                    progressText.textContent = `Ready (${totalBars} total bars)`;
                }
                else if (msg.type === "update") {
                    addCandle(msg);
                    updatePortfolio(msg.timestamp, msg.portfolio);
                    if (msg.indicators) {
                        updateIndicators(msg.timestamp, msg.indicators);
                    }
                    currentStepEl.textContent = msg.step;
                    cumRewardEl.textContent = msg.cumulative_reward.toFixed(2);
                    
                    const returnPct = ((msg.portfolio.portfolio_value / INITIAL_CAPITAL - 1) * 100).toFixed(2);
                    totalReturnEl.textContent = `${returnPct}%`;
                    totalReturnEl.className = returnPct >= 0 ? 'positive' : 'negative';
                    
                    const actionMap = {0: 'hold', 1: 'buy', 2: 'sell'};
                    const action = actionMap[msg.action] || 'hold';
                    const sharesSold = msg.shares_sold || 0;
                    
                    if (action !== 'hold') {
                        logAction(action, msg.close, msg.timestamp, sharesSold);
                        if (action === 'sell' && sharesSold > 1) {
                            log(`${action.toUpperCase()} ${sharesSold} shares @ $${msg.close.toFixed(2)}`);
                        } else {
                            log(`${action.toUpperCase()} @ $${msg.close.toFixed(2)}`);
                        }
                    }
                }
                else if (msg.type === "complete") {
                    log(`Simulation complete! Final value: $${msg.final_portfolio_value.toFixed(2)}`, "success");
                    log(`Total return: ${msg.total_return.toFixed(2)}%`, "success");
                    progressText.textContent = `Complete - Return: ${msg.total_return.toFixed(2)}%`;
                    playBtn.disabled = true; 
                    pauseBtn.disabled = true; 
                    stepBtn.disabled = true;
                }
                else if (msg.type === "error") {
                    log(`ERROR: ${msg.message}`, "error");
                }
            } catch (error) {
                console.error("Error processing message:", error);
                log(`Error processing message: ${error.message}`, "error");
            }
        };
        
        ws.onclose = () => {
            log("Disconnected"); 
            statusText.textContent = "Disconnected"; 
            statusText.className = ""; 
            isConnected = false;
            playBtn.disabled = true; 
            pauseBtn.disabled = true; 
            stepBtn.disabled = true; 
            connectBtn.textContent = "🔌 Connect";
        };
        
        ws.onerror = (error) => { 
            log("WebSocket connection error", "error"); 
            console.error("WebSocket error:", error); 
        };
        
    } catch (error) {
        log(`Failed to create WebSocket: ${error.message}`, "error");
        console.error("WebSocket creation error:", error);
    }
}

// Controls
connectBtn.onclick = () => { 
    if (isConnected && ws) {
        log("Disconnecting...");
        ws.close();
    } else {
        connectWS(); 
    }
};

playBtn.onclick = () => { 
    if (ws && ws.readyState === WebSocket.OPEN) { 
        ws.send(JSON.stringify({cmd: "resume"})); 
        log("Resumed"); 
    } 
};

pauseBtn.onclick = () => { 
    if (ws && ws.readyState === WebSocket.OPEN) { 
        ws.send(JSON.stringify({cmd: "pause"})); 
        log("Paused"); 
    } 
};

stepBtn.onclick = () => { 
    if (ws && ws.readyState === WebSocket.OPEN) { 
        ws.send(JSON.stringify({cmd: "step"})); 
        log("Step forward"); 
    } 
};

speedSlider.oninput = () => { 
    speedLabel.textContent = parseFloat(speedSlider.value).toFixed(2) + "s"; 
};

// Load data
async function loadSymbols() {
    try {
        log("Loading symbols...");
        const res = await fetch(`${API_BASE}/api/data`);
        
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        
        const data = await res.json();
        symbolSelect.innerHTML = "";
        
        if (data.symbols && data.symbols.length > 0) {
            data.symbols.forEach(symbol => {
                const opt = document.createElement("option");
                opt.value = symbol; 
                opt.textContent = symbol;
                symbolSelect.appendChild(opt);
            });
            log(`Loaded ${data.symbols.length} symbols`, "success");
        } else {
            symbolSelect.innerHTML = '<option value="">No data available</option>';
            log("No symbols found. Run: python run.py --mode setup", "error");
        }
    } catch (error) {
        log(`Failed to load symbols: ${error.message}`, "error");
        console.error("Load symbols error:", error);
        symbolSelect.innerHTML = '<option value="">Error loading symbols</option>';
    }
}


// Learning curve chart
function loadLearningCurve() {
    fetch(`${API_BASE}/api/learning-curve`)
        .then(res => {
            if (!res.ok) {
                // If 404 or other error, log and return early
                if (res.status === 404) {
                    log(`Learning curve data not found (evaluations.npz may not exist)`, "error");
                    return null;
                }
                throw new Error(`HTTP error! status: ${res.status}`);
            }
            return res.json();
        })
        .then(data => {
            // Handle case where res.json() returned null (404 case)
            if (!data) {
                return;
            }
            
            if (data.error) {
                log(`Could not load learning curve: ${data.error}`, "error");
                return;
            }
            
            // Validate data structure
            if (!data || !data.timesteps || !data.mean_rewards || !data.std_rewards) {
                log(`Invalid learning curve data structure`, "error");
                console.error("Received data:", data);
                return;
            }
            
            // Ensure arrays exist and are arrays
            if (!Array.isArray(data.timesteps) || !Array.isArray(data.mean_rewards) || !Array.isArray(data.std_rewards)) {
                log(`Learning curve data is not in array format`, "error");
                console.error("Data types:", {
                    timesteps: typeof data.timesteps,
                    mean_rewards: typeof data.mean_rewards,
                    std_rewards: typeof data.std_rewards
                });
                return;
            }
            
            const timesteps = data.timesteps;
            const meanRewards = data.mean_rewards;
            const stdRewards = data.std_rewards;
            
            // Validate arrays have same length
            if (timesteps.length !== meanRewards.length || timesteps.length !== stdRewards.length) {
                log(`Data length mismatch: timesteps=${timesteps.length}, mean=${meanRewards.length}, std=${stdRewards.length}`, "error");
                return;
            }
            
            // Validate arrays are not empty
            if (timesteps.length === 0) {
                log(`Learning curve data is empty`, "error");
                return;
            }
            
            // Calculate upper and lower bounds for variance band
            const upperBound = meanRewards.map((mean, i) => mean + stdRewards[i]);
            const lowerBound = meanRewards.map((mean, i) => mean - stdRewards[i]);
            
            // Create learning curve chart with variance band
            Plotly.newPlot("learningCurve", [
                // Variance band (filled area)
                {
                    x: timesteps.concat(timesteps.slice().reverse()),
                    y: upperBound.concat(lowerBound.slice().reverse()),
                    type: "scatter",
                    mode: "lines",
                    name: "±1 Std Dev",
                    fill: "toself",
                    fillcolor: "rgba(128, 0, 128, 0.2)",
                    line: {color: "transparent"},
                    showlegend: true,
                    hoverinfo: "skip"
                },
                // Mean reward line
                {
                    x: timesteps,
                    y: meanRewards,
                    mode: "lines",
                    name: "Mean Reward",
                    type: "scatter",
                    line: {color: "#800080", width: 3},
                    marker: {size: 6, color: "#800080"}
                }
            ], {
                margin: {t:40, b:50, l:60, r:20},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#ffffff',
                font: {color: '#424242', size: 12},
                xaxis: {
                    title: 'Timesteps',
                    gridcolor: '#e0e0e0',
                    showgrid: true,
                    gridwidth: 1,
                    type: 'linear'
                },
                yaxis: {
                    title: 'Mean Reward',
                    gridcolor: '#e0e0e0',
                    showgrid: true,
                    gridwidth: 1
                },
                legend: {x: 0, y: 1, orientation: 'h', bgcolor: 'rgba(255,255,255,0.8)'},
                dragmode: 'pan',
                title: {
                    text: 'Model Learning Curve — Evaluation Mean Reward Over Time',
                    font: {size: 16, color: '#424242'}
                }
            });
            
            log("Learning curve loaded successfully", "success");
        })
        .catch(error => {
            log(`Error loading learning curve: ${error.message}`, "error");
            console.error("Learning curve error:", error);
        });
}

// Startup
async function init() {
    log("Initializing Trading Simulator...");
    log(`API Base: ${API_BASE}`);
    log(`WebSocket Base: ${WS_BASE}`);
    
    // Initialize empty charts
    if (!initCharts()) {
        log("Failed to initialize charts at startup", "error");
    }
    
    await loadSymbols();
    
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const health = await res.json();
        log(`Server ready - Model: ${health.model_exists ? 'Available' : 'Not found (will use heuristic)'}`, "success");
        log(`Data directory: ${health.data_dir}`);
        log(`Static directory exists: ${health.static_exists}`);
    } catch (error) {
        log(`Warning: Could not connect to server - ${error.message}`, "error");
        console.error("Health check error:", error);
    }
    
    // Load learning curve chart
    loadLearningCurve();
}

// Wait for DOM and Plotly to be ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}