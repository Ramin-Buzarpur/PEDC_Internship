# PEDC Internship — Trading Research & System

مونو-رپوی توسعهٔ ابزار معاملاتی مبتنی بر Diffusion + Transformer + Reinforcement
Learning برای بازارهای مالی.

## ساختار

| پوشه | توضیح |
|---|---|
| [`engine/`](./engine) | هستهٔ اصلی سیستم — تولید سیگنال، مدل‌ها، بک‌تست، آموزش. تنها بخشی که فعالانه توسعه داده می‌شود. |
| [`dashboard/`](./dashboard) | داشبورد وب (React/Vite) برای مقایسهٔ بصری مدل‌ها. |
| [`research/`](./research) | کدهای اکتشافی/آزمایشی: loss functionهای جدید، پروتوتایپ‌های اولیه. پایدار نیست، فقط برای تحقیق. |
| [`docs/papers/`](./docs/papers) | مقالات علمی مرجع پروژه. |
| [`docs/assets/`](./docs/assets) | تصاویر/نمودارهای مرجع. |
| [`archive/`](./archive) | نسخه‌های قدیمی/کنارگذاشته‌شده — فقط برای رفرنس تاریخی، روی این‌ها توسعه ندهید. |

## شروع سریع

```bash
cd engine
pip install -r requirements.txt
python3 scripts/prepare_cache.py
python3 scripts/run_training.py
python3 scripts/run_backtest.py
```

جزئیات کامل: [`engine/README.md`](./engine/README.md)

## وضعیت

- ✅ `engine/`: تست دارد، بک‌تست walk-forward واقعی دارد.
- 🧪 `research/`: کد آزمایشی، مستندسازی حداقلی.
- 🗄 `archive/`: منجمد شده، فقط رفرنس.
