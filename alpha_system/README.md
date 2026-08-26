# PEDC Alpha — Hybrid Transformer–Diffusion–RL Stock Prediction & Trading System

سیستم کامل معاملاتی کوانت ترکیبی از **DiffStock (MaTCHS + DDPM شرطی)**، **Transformer رابطه‌ای ماسک‌شده**، **Reinforcement Learning (PPO)** و **تحلیل اخبار** — پیاده‌سازی‌شده بر اساس مقالات پوشهٔ `Paper/` به‌علاوهٔ تکنیک‌های مدرن صنعت (Core-Satellite، Vol Targeting، Walk-Forward، DDIM).

> ⚠️ **سلب مسئولیت مهم:** هیچ سیستمی در جهان «قطعاً» سودده نیست. این ابزار پژوهشی-کمکی است، توصیهٔ مالی نیست. نتایج بک‌تست تضمینی برای آینده نیستند. فقط با سرمایه‌ای معامله کنید که تحمل از دست دادنش را دارید.

---

## نتایج واقعی Out-of-Sample (Walk-Forward، دورهٔ تست 2018-02 → 2026-08، شامل کرونا و بازار خرسی ۲۰۲۲)

| استراتژی | CAGR | Sharpe | Sortino | MaxDD | Calmar | برد روزانه |
|---|---|---|---|---|---|---|
| **PEDC Alpha (Core+Satellite)** | **12.9%** | **0.87** | 0.97 | **−27.1%** | 0.47 | 55.9% |
| هستهٔ روند-مدیریت‌شده | 16.8% | 0.92 | 1.06 | −31.5% | 0.53 | 56.0% |
| ماهوارهٔ آلفا (RL+Diffusion+Momentum) | −3.4% | −0.43 | −0.44 | −28.9% | −0.12 | 49.5% |
| خرید و نگه‌داری هم‌وزن | 18.6% | 0.95 | 1.11 | −32.5% | 0.57 | 56.1% |

**ارزش اصلی سیستم:** بازده نزدیک به بازار با افت سرمایهٔ (Drawdown) محسوساً کمتر و مسیر هموارتر — دقیقاً همان کاری که محصولات واقعی مدیریت سرمایه می‌کنند. آلفای خالص روزانه بعد از هزینه‌ها در این universe (مگا-کپ‌های آمریکا) تقریباً صفر است؛ سیستم این را **صادقانه** اندازه می‌گیرد و به‌جای وعده، ریسک را مدیریت می‌کند.

نمودارها: `outputs/plots/backtest_equity.png` و `outputs/plots/forecast_fan.png`

---

## معماری

```
دادهٔ ۴ بازار (nasdaq/nyse/sp500/forbes2000) ──┐
اخبار Google News + تحلیل احساسات لغوی ────────┤
                                               ▼
                    ┌─────────────────────────────────────┐
                    │  Feature Engineering (imp.pdf)       │
                    │  مومنتوم نرمال‌شده با ولتاژ، MACD،    │
                    │  ATR، حجم z-score، سنتیمنت           │
                    └──────────────┬──────────────────────┘
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
   ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐
   │ Diffusion (DDPM)│   │  RL (PPO)        │   │ مومنتوم کلاسیک  │
   │ MaTCHS Denoiser │   │ پالیسی پسماندی    │   │                 │
   │ Att-DiCEm + MRT │   │ روی مومنتوم،     │   │                 │
   │ ماسک رابطه‌ای     │   │ Vol Targeting،   │   │                 │
   │ (DiffStock)     │   │ جریمهٔ Drawdown  │   │                 │
   │ + DDIM sampling │   │ (PPO از صفر)     │   │                 │
   └────────┬────────┘   └────────┬─────────┘   └────────┬────────┘
            └──────── ترکیب سیگنال (55/35/10) ───────────┘
                               ▼
              ┌──────────────────────────────────┐
              │  Core-Satellite Portfolio        │
              │  هسته: روند شاخص + Vol Targeting │
              │  ماهواره: آلفای ترکیبی (20%)      │
              │  هزینه تراکنش + slippage در بک‌تست │
              └──────────────────────────────────┘
```

### نگاشت به مقالات

| مقاله | استفاده در سیستم |
|---|---|
| **DiffStock** (`1_C.pdf`) | DDPM شرطی با MaTCHS: Att-DiCEm (کانولوشن متسع علّی) + Masked Relational Transformer با ماسک per-relation، نویز تطبیقی مبتنی بر واریانس محلی، فرمول‌بندی masked یکپارچهٔ گذشته+آینده |
| **DiffSTG** (`2301.13629`) | نمونه‌گیری سریع DDIM، شرط‌گذاری masked روی کل پنجره |
| **Adv-ALSTM** (`2_S.pdf`) | مهندسی ویژگی‌های قیمتی نرمال‌شده، مقاوم‌سازی |
| **Oxford Benchmark** (`imp.pdf`) | بهینه‌سازی ریسک-تنظیم‌شده، EWMA Vol Targeting، سیگنال محدود [-1,1]، تحلیل Breakeven cost، انتخاب مدل روی validation Sharpe |
| **Akita ICIS 2016** | ادغام اخبار + قیمت، شبیه‌سازی معامله |
| **Large Margin / Head Pruning** (`1803.05598`, `1905.09418`) | طراحی هدهای توجه (هدهای ماسک‌دار + هدهای آزاد) |

---

## شروع سریع

```bash
cd alpha_system

# ۱) نصب پیش‌نیازها (در صورت نبود)
pip install -r requirements.txt

# ۲) ساخت کش داده (اسکن اولیه ~3 دقیقه، دفعات بعد آنی)
python3 scripts/prepare_cache.py

# ۳) آپدیت داده تا امروز (اختیاری، yfinance — در محدودیت rate ممکن است fail شود)
python3 scripts/update_data.py

# ۴) آموزش کامل (Diffusion ~12 دقیقه + RL ~4 دقیقه روی RTX 3050)
python3 scripts/run_training.py

# ۵) بک‌تست walk-forward + نمودار + گزارش
python3 scripts/run_backtest.py

# ۶) سیگنال معاملاتی برای جلسهٔ بعدی
python3 scripts/live_signal.py --capital 10000 --top 10
```

### گردش کار روزانهٔ معامله

```bash
python3 scripts/update_data.py          # دادهٔ امروز
python3 scripts/live_signal.py --capital 10000   # سیگنال جدید
# خروجی: outputs/signals_YYYYMMDD.csv + جدول LONG/SHORT با وزن و دلار
python3 scripts/run_training.py --only rl --updates 150   # تنظیم دوره‌ای RL (هفتگی)
python3 scripts/run_training.py           # آموزش کامل (ماهانه یا بعد از تغییر universe)
```

### گزینه‌های مهم

```bash
python3 scripts/run_training.py --stocks 30 --epochs 15 --updates 80   # سبک‌تر
python3 scripts/live_signal.py --no-news     # بدون اخبار
PEDC_DEVICE=cpu python3 scripts/run_backtest.py   # اجرای اجباری روی CPU
```

---

## ساختار پروژه

```
alpha_system/
├── configs/default.yaml          # تمام هایپرپارامترها
├── src/
│   ├── config.py                 # لودر YAML
│   ├── pipeline.py               # آماده‌سازی داده + فیچر + device
│   ├── data/
│   │   ├── market_data.py        # اسکن ۴ بازار، انتخاب نقدشونده‌ها، کش npz
│   │   ├── features.py           # مومنتوم/MACD/vol نرمال‌شده (imp.pdf)
│   │   ├── graph.py              # ماتریس رابطهٔ همبستگی مثبت/منفی [N,N,2]
│   │   ├── news.py               # RSS گوگل‌نیوز + امتیازدهی لغوی مالی + کش
│   │   └── dataset.py            # دیتاست دیفیوژن (کانال‌های OHLCV+sentiment)
│   ├── models/
│   │   ├── temporal.py           # Att-DiCEm، embedding سینوسی
│   │   ├── matchs.py             # MaTCHS: MRT با ماسک رابطه‌ای per-head
│   │   ├── diffusion.py          # DDPM شرطی + DDIM + نویز تطبیقی
│   │   └── rl_agent.py           # محیط پرتفوی + PPO + GAE
│   ├── training/
│   │   ├── train_diffusion.py    # آموزش با early-stopping روی val
│   │   └── train_rl.py           # آموزش RL + انتخاب بهترین بر val-Sharpe
│   ├── backtest/
│   │   ├── engine.py             # موتور بک‌تست با هزینه + Core-Satellite
│   │   ├── metrics.py            # Sharpe/Sortino/Calmar/MaxDD/Breakeven
│   │   └── walkforward.py        # ارزیابی out-of-sample بدون lookahead
│   ├── inference/
│   │   ├── predictor.py          # تولید سناریوهای احتمالاتی
│   │   └── strategy.py           # ترکیب سیگنال‌ها
│   └── utils/runtime.py          # seed/logger/device
├── scripts/                      # run_training, run_backtest, live_signal,
│                                 # prepare_cache, update_data, plot_forecast
├── tests/                        # ۳۷ تست واحد (همه پاس)
├── data_cache/                   # کش پنل قیمت + اخبار (خودکار)
└── outputs/                      # مدل‌ها، نمودارها، گزارش‌ها، سیگنال‌ها
```

---

## نحوهٔ تصمیم‌گیری سیگنال

1. **Diffusion:** ۱۶ سناریوی آینده برای هر سهم تولید می‌کند → `E[بازده فردا] / ولتاژ پیش‌بینی‌شده`
2. **RL:** پالیسی PPO روی وضعیت بازار (فیچرها + وزن فعلی + drawdown) اصلاحِ پسماندی روی مومنتوم یاد می‌گیرد
3. **مومنتوم:** z-score افق ۲۱ روزه
4. ترکیب `0.55·RL + 0.35·Diffusion + 0.10·Momentum` → EMA ۵ روزه → فیلتر سیگنال ضعیف
5. **Core (80٪ ریسک):** اگر شاخص هم‌وزن بالای MA200 باشد خرید vol-targeted، وگرنه ۳۰٪ مالتایپلایر
6. **Satellite (20٪):** پرتفوی آلفای ترکیبی
7. سقف: هر سهم ≤ ۲۵٪، اهرم ناخالص ≤ ۱۰۰٪، هزینهٔ ۵ bps + slippage ۲ bps در بک‌تست

---

## محدودیت‌ها و مسیرهای بهبود (صادقانه)

- **آلفای روزانه ضعیف است:** در مگا-کپ‌های آمریکا بعد از هزینه، edge روزانهٔ خالص نزدیک صفر است (در گزارش هم دیده می‌شود). ارزش اصلی در مدیریت ریسک هسته است.
- **اخبار تاریخی نداریم:** آرشیو اخبار فقط ۱۴ روز اخیر را پوشش می‌دهد؛ در بک‌تست سنتیمنت صفر است و در لایو فعال می‌شود. برای بک‌تستِ سنتیمنت به دیتاست تاریخی اخبار (مثلاً Kaggle financial news) نیاز است.
- **داده‌های delisted:** نمادهای حذف‌شده (مثل BSI/FPT) با قیمت فریز ادامه می‌یابند؛ با `prepare_cache.py` دوره‌ای universe تازه‌سازی کنید.
- **آموزش دوره‌ای:** بعد از ~۶ ماه دادهٔ جدید، `run_training.py` را دوباره اجرا کنید.
- **سرعت دیفیوژن در لایو:** هر سیگنال‌گیری ~۵-۱۰ ثانیه روی GPU.

## تست‌ها

```bash
python3 -m pytest tests/ -q     # ۳۷/۳۷ پاس
```
