"""Mock datasets for the travel agent RL system.

Contains realistic Chinese travel data (Beijing POIs, routes, weather, etc.)
used by tools during training rollouts so no real API calls are needed.
"""

# ── Beijing POIs (Points of Interest) ────────────────────────────────────

MOCK_POIS = [
    {
        "id": "B000A83M61",
        "name": "故宫博物院",
        "type": "景点",
        "address": "北京市东城区景山前街4号",
        "location": "116.397026,39.917839",
        "tel": "010-65131892",
        "rating": 4.8,
        "city": "北京",
    },
    {
        "id": "B000A83M62",
        "name": "天安门广场",
        "type": "景点",
        "address": "北京市东城区东长安街",
        "location": "116.397428,39.90923",
        "tel": "",
        "rating": 4.7,
        "city": "北京",
    },
    {
        "id": "B000A83M63",
        "name": "颐和园",
        "type": "景点",
        "address": "北京市海淀区新建宫门路19号",
        "location": "116.275041,39.999547",
        "tel": "010-62881144",
        "rating": 4.7,
        "city": "北京",
    },
    {
        "id": "B000A83M64",
        "name": "故宫餐厅",
        "type": "餐饮",
        "address": "北京市东城区景山前街4号故宫内",
        "location": "116.397526,39.918039",
        "tel": "010-65131893",
        "rating": 4.0,
        "city": "北京",
    },
    {
        "id": "B000A83M65",
        "name": "王府井小吃街",
        "type": "餐饮",
        "address": "北京市东城区王府井大街",
        "location": "116.410535,39.915085",
        "tel": "",
        "rating": 3.9,
        "city": "北京",
    },
]

# ── Mock Routes ──────────────────────────────────────────────────────────

MOCK_ROUTES = {
    ("116.397428,39.90923", "116.403414,39.914714", "walking"): {
        "distance": "1.2公里",
        "duration": "15分钟",
        "mode": "walking",
        "steps": [
            "从天安门广场向北步行200米",
            "左转进入南池子大街",
            "沿南池子大街步行800米",
            "到达目的地",
        ],
    },
    ("116.397428,39.90923", "116.275041,39.999547", "driving"): {
        "distance": "18.5公里",
        "duration": "40分钟",
        "mode": "driving",
        "steps": [
            "从天安门广场出发向西",
            "沿西长安街行驶5公里",
            "上四环向北",
            "沿北四环行驶10公里",
            "从颐和园出口驶出",
            "到达颐和园",
        ],
    },
}

# ── Mock Weather ─────────────────────────────────────────────────────────

MOCK_WEATHER = {
    ("北京", "2026-05-20"): {
        "city": "北京",
        "date": "2026-05-20",
        "weather": "晴",
        "temperature": {"high": 28, "low": 16},
        "humidity": 35,
        "wind": {"direction": "北风", "speed": 3},
        "aqi": 72,
        "aqi_level": "良",
        "suggestion": "天气晴好，适合户外活动，注意防晒。",
    },
    ("北京", "2026-05-21"): {
        "city": "北京",
        "date": "2026-05-21",
        "weather": "多云",
        "temperature": {"high": 26, "low": 15},
        "humidity": 45,
        "wind": {"direction": "南风", "speed": 2},
        "aqi": 58,
        "aqi_level": "良",
        "suggestion": "多云天气，适宜出行。",
    },
    ("上海", "2026-05-20"): {
        "city": "上海",
        "date": "2026-05-20",
        "weather": "小雨",
        "temperature": {"high": 23, "low": 18},
        "humidity": 78,
        "wind": {"direction": "东风", "speed": 4},
        "aqi": 45,
        "aqi_level": "优",
        "suggestion": "有小雨，出门请带伞。",
    },
}

# ── Mock Calendar Events ─────────────────────────────────────────────────

MOCK_CALENDAR = {
    ("user_001", "2026-05-21"): [
        {
            "event_id": "evt_001",
            "title": "故宫参观",
            "start_time": "2026-05-21 09:00",
            "end_time": "2026-05-21 12:00",
            "location": "故宫博物院",
            "description": "提前预约的故宫门票，上午场。",
        },
        {
            "event_id": "evt_002",
            "title": "午餐 - 全聚德烤鸭",
            "start_time": "2026-05-21 12:30",
            "end_time": "2026-05-21 14:00",
            "location": "全聚德(前门店)",
            "description": "北京老字号烤鸭店。",
        },
    ],
    ("user_001", "2026-05-22"): [
        {
            "event_id": "evt_003",
            "title": "颐和园游览",
            "start_time": "2026-05-22 08:30",
            "end_time": "2026-05-22 12:00",
            "location": "颐和园",
            "description": "上午游颐和园，下午自由活动。",
        },
    ],
}

# ── Mock Search Results ──────────────────────────────────────────────────

MOCK_SEARCH_RESULTS = {
    "北京旅游攻略": [
        {
            "title": "北京三日游最佳攻略 - 马蜂窝",
            "url": "https://www.mafengwo.cn/gonglve/ziyouxing/394290.html",
            "snippet": "北京三日游攻略：第一天故宫、天安门、王府井；第二天长城、十三陵；第三天颐和园、圆明园。",
        },
        {
            "title": "2026北京旅游必去景点TOP10 - 携程",
            "url": "https://www.ctrip.com/html5/you/article/detail/12345.html",
            "snippet": "故宫博物院、长城、天坛、颐和园、圆明园、南锣鼓巷、798艺术区等。",
        },
        {
            "title": "北京自由行实用攻略 - 知乎",
            "url": "https://www.zhihu.com/question/123456",
            "snippet": "建议住在地铁沿线，出行方便。故宫需要提前网上预约。",
        },
    ],
    "北京故宫门票": [
        {
            "title": "故宫博物院门票预约 - 官网",
            "url": "https://www.dpm.org.cn/visit/ticket.html",
            "snippet": "故宫门票实行全网络售票，淡季40元/人，旺季60元/人，需提前预约。",
        },
        {
            "title": "故宫参观指南 - 2026最新版",
            "url": "https://www.dpm.org.cn/visit/guide.html",
            "snippet": "故宫每周一闭馆(法定节假日除外)，建议从南门(午门)进入。",
        },
    ],
}
